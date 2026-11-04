// Included in the pinned IQ CUDA TU to reuse its independently tested weight decoders.
// Correctness experiment: CPU Q8_K inputs, CPU SwiGLU/Q8_0 intermediates.
#include "legacy_act.h"
namespace strata::kernels {
namespace {
// Decode four adjacent integer coefficients. Their common factor remains
// d_fp16 * 1/8, matching the pinned CPU IQ2 dot product exactly.
template<int TYPE>
__device__ __forceinline__ int4 mk_iq2_coeff4(const uint8_t* block,int k){
    int ib=k/32,il=(k%32)/8,j=k%8;
    uint64_t grid;int factor;uint8_t signs;
    if constexpr(TYPE==16){
        const auto* w=(const block_iq2_xxs*)block;
        const uint16_t* q=w->qs+4*ib;
        uint32_t aux=(uint32_t)q[2]|((uint32_t)q[3]<<16);
        grid=iq2xxs_grid[((const uint8_t*)q)[il]];
        factor=2*(int)(aux>>28)+1;
        signs=ksigns_iq2xs[(aux>>(7*il))&127];
    }else{
        const auto* w=(const block_iq2_s*)block;
        int code=w->qs[4*ib+il]|((w->qh[ib]<<(8-2*il))&0x300);
        grid=iq2s_grid[code];
        factor=2*((w->scales[ib]>>(4*(il/2)))&15)+1;
        signs=w->qs[32+4*ib+il];
    }
    uint32_t four=(uint32_t)(grid>>(8*j));
    int a=factor*(int)(four&255),b=factor*(int)((four>>8)&255);
    int c=factor*(int)((four>>16)&255),d=factor*(int)(four>>24);
    return make_int4((signs&(1<<j))?-a:a,(signs&(2<<j))?-b:b,
                     (signs&(4<<j))?-c:c,(signs&(8<<j))?-d:d);
}
template<int TYPE>
__global__ void mk_packed_iq2_gu(NativeExpertLayout L,const unsigned long long* ptr,
                               const int32_t* token,const int32_t* nt,const block_q8_K* x,float* gate,float* up){
    int lane=threadIdx.x&31,warp=threadIdx.x>>5,row=blockIdx.x*4+warp,entry=blockIdx.y;
    const auto* blob=(const uint8_t*)ptr[entry];
    const auto* gr=blob+row*L.gu_row;const auto* ur=blob+L.up_off+row*L.gu_row;
    const block_q8_K* a=x+(size_t)token[entry]*10;
    bool wide=nt[entry]>=2;int lanes=wide?16:8;
    constexpr int bytes=TYPE==16?66:82;float g=0,u=0;
    if(lane<lanes)for(int b=0;b<10;b++){
        const auto* gb=gr+b*bytes;const auto* ub=ur+b*bytes;
        int gi=0,ui=0;
        for(int chunk=0;chunk<256/(lanes*4);chunk++){
            int k=chunk*lanes*4+lane*4;
            int4 gw=mk_iq2_coeff4<TYPE>(gb,k),uw=mk_iq2_coeff4<TYPE>(ub,k);
            gi+=gw.x*(int)a[b].qs[k];ui+=uw.x*(int)a[b].qs[k];
            gi+=gw.y*(int)a[b].qs[k+1];ui+=uw.y*(int)a[b].qs[k+1];
            gi+=gw.z*(int)a[b].qs[k+2];ui+=uw.z*(int)a[b].qs[k+2];
            gi+=gw.w*(int)a[b].qs[k+3];ui+=uw.w*(int)a[b].qs[k+3];
        }
        float gd=__half2float(*(const __half*)gb),ud=__half2float(*(const __half*)ub);
        float gs=__fmul_rn(wide?__fmul_rn(gd,.125f):gd,a[b].d);
        float us=__fmul_rn(wide?__fmul_rn(ud,.125f):ud,a[b].d);
        g=__fmaf_rn(gs,(float)gi,g);u=__fmaf_rn(us,(float)ui,u);
    }
    for(int off=lanes/2;off;off>>=1){g=__fadd_rn(g,__shfl_xor_sync(0xffffffff,g,off));u=__fadd_rn(u,__shfl_xor_sync(0xffffffff,u,off));}
    if(lane==0){gate[entry*L.n_ff+row]=wide?g:__fmul_rn(g,.125f);up[entry*L.n_ff+row]=wide?u:__fmul_rn(u,.125f);}
}
__global__ void mk_cpu_pool_down(NativeExpertLayout L,const unsigned long long* ptr,
                                const MkLegacyAct* hq,float* out){
    int lane=threadIdx.x&31,warp=threadIdx.x>>5,row=blockIdx.x*4+warp,entry=blockIdx.y;
    const auto* w=(const block_q2_0*)((const uint8_t*)ptr[entry]+L.down_off+row*L.d_row);
    const auto& h=hq[entry];float acc=0,corr=0;
    for(int b0=0;b0<10;b0+=8){
        int nb=min(8,10-b0);
        if(lane<2*nb){float d=__half2float(w[b0+lane/2].d);corr=__fmaf_rn(d,h.hx[2*b0+lane],corr);}
        for(int i=0;i<nb;i++)if(lane<16){
            int b=b0+i,byte=w[b].qs[lane],dot=0;
            for(int j=0;j<4;j++)dot+=((byte>>(2*j))&3)*(int)h.q[b*64+lane*4+j];
            float scale=__fmul_rn(__half2float(w[b].d),h.scale[2*b+(lane>=8)]);
            acc=__fmaf_rn(scale,(float)dot,acc);
        }
    }
    for(int off=8;off;off>>=1){acc=__fadd_rn(acc,__shfl_xor_sync(0xffffffff,acc,off));corr=__fadd_rn(corr,__shfl_xor_sync(0xffffffff,corr,off));}
    if(lane==0)out[entry*L.n_embd+row]=__fsub_rn(acc,corr);
}
__global__ void mk_exact_iq2_gu(int type,NativeExpertLayout L,const unsigned long long* ptr,
                              const int32_t* token,const int32_t* nt,const block_q8_K* x,float* gate,float* up){
    __shared__ float wg[4][256],wu[4][256];
    int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    int row=blockIdx.x*4+warp,entry=blockIdx.y;
    const auto* blob=(const uint8_t*)ptr[entry];
    const auto* gr=blob+row*L.gu_row;const auto* ur=blob+L.up_off+row*L.gu_row;
    const block_q8_K* a=x+(size_t)token[entry]*10;
    bool wide=nt[entry]>=2;int lanes=wide?16:8;
    int block_bytes=type==16?66:82;
    float g=0,u=0;
    for(int b=0;b<10;b++){
        dq_dispatch<float>(type,gr,b,wg[warp],lane);dq_dispatch<float>(type,ur,b,wu[warp],lane);
        __syncthreads();
        float gd=__half2float(*(const __half*)(gr+b*block_bytes));
        float ud=__half2float(*(const __half*)(ur+b*block_bytes));
        float gb=__fmul_rn(gd,.125f),ub=__fmul_rn(ud,.125f);
        int gi=0,ui=0;
        if(lane<lanes){
            for(int chunk=0;chunk<256/(lanes*4);chunk++)for(int j=0;j<4;j++){
                int k=chunk*lanes*4+lane*4+j;
                int gw=gb==0?0:__float2int_rn(wg[warp][k]/gb);
                int uw=ub==0?0:__float2int_rn(wu[warp][k]/ub);
                gi+=gw*(int)a[b].qs[k];ui+=uw*(int)a[b].qs[k];
            }
            float gs=__fmul_rn(wide?gb:gd,a[b].d),us=__fmul_rn(wide?ub:ud,a[b].d);
            g=__fmaf_rn(gs,(float)gi,g);u=__fmaf_rn(us,(float)ui,u);
        }
        __syncthreads();
    }
    // GCC's horizontal AVX2/AVX512 reduction: high+low halves, then 2, then 1.
    for(int off=lanes/2;off;off>>=1){g=__fadd_rn(g,__shfl_xor_sync(0xffffffff,g,off));u=__fadd_rn(u,__shfl_xor_sync(0xffffffff,u,off));}
    if(lane==0){gate[entry*L.n_ff+row]=wide?g:__fmul_rn(g,.125f);up[entry*L.n_ff+row]=wide?u:__fmul_rn(u,.125f);}
}
__global__ void mk_compatible_gu(int type,NativeExpertLayout L,const unsigned long long* ptr,
                               const int32_t* token,const float* x,float* gate,float* up){
    __shared__ float wg[4][256],wu[4][256];
    int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    int row=blockIdx.x*4+warp,entry=blockIdx.y;
    const auto* blob=(const uint8_t*)ptr[entry];
    const auto* gr=blob+row*L.gu_row;const auto* ur=blob+L.up_off+row*L.gu_row;
    const float* a=x+(size_t)token[entry]*L.n_embd;
    float g=0,u=0;
    for(int b=0;b<L.n_embd/256;b++){
        dq_dispatch<float>(type,gr,b,wg[warp],lane);
        dq_dispatch<float>(type,ur,b,wu[warp],lane);
        __syncthreads();
        for(int j=lane;j<256;j+=32){g+=wg[warp][j]*a[b*256+j];u+=wu[warp][j]*a[b*256+j];}
        __syncthreads();
    }
    g=warp_sum(g);u=warp_sum(u);
    if(lane==0){gate[entry*L.n_ff+row]=g;up[entry*L.n_ff+row]=u;}
}
__global__ void mk_compatible_down(NativeExpertLayout L,const unsigned long long* ptr,
                                 const block_q8_0* hq,float* out){
    int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    int row=blockIdx.x*4+warp,entry=blockIdx.y;
    const auto* w=(const block_q2_0*)((const uint8_t*)ptr[entry]+L.down_off+row*L.d_row);
    const auto* h=hq+entry*(L.n_ff/32);
    float sum=0;
    // Q2_0 stores four consecutive codes in each byte, mapping to {-1,0,1,2}.
    // Match ggml_vec_dot_q2_0_q8_0's two Q8_0 scales per 64-value weight block.
    for(int b=0;b<L.n_ff/64;b++){
        float subtotal=0;
        for(int half=0;half<2;half++){
            int j=half*32+lane;
            int q=((w[b].qs[j/4]>>(2*(j%4)))&3)-1;
            int dot=q*int(h[2*b+half].qs[lane]);
            for(int off=16;off;off>>=1)dot+=__shfl_down_sync(0xffffffff,dot,off);
            if(lane==0)subtotal=__fadd_rn(subtotal,__fmul_rn(float(dot),__half2float(h[2*b+half].d)));
        }
        if(lane==0)sum=__fadd_rn(sum,__fmul_rn(subtotal,__half2float(w[b].d)));
    }
    if(lane==0)out[entry*L.n_embd+row]=sum;
}
}
void mk_compatible_gu_launch(int type,NativeExpertLayout L,const unsigned long long* ptr,
                             const int32_t* tok,const float* x,int entries,float* gate,float* up,void* stream){
    mk_compatible_gu<<<dim3((unsigned)(L.n_ff/4),(unsigned)entries),128,0,(cudaStream_t)stream>>>(type,L,ptr,tok,x,gate,up);
}
void mk_exact_iq2_gu_launch(int type,NativeExpertLayout L,const unsigned long long* ptr,
                          const int32_t* tok,const int32_t* nt,const void* x,int entries,float* gate,float* up,void* stream){
    if(std::getenv("MK_WORKER_PACKED_IQ2")){
        dim3 grid((unsigned)(L.n_ff/4),(unsigned)entries);
        if(type==16)mk_packed_iq2_gu<16><<<grid,128,0,(cudaStream_t)stream>>>(L,ptr,tok,nt,(const block_q8_K*)x,gate,up);
        else mk_packed_iq2_gu<22><<<grid,128,0,(cudaStream_t)stream>>>(L,ptr,tok,nt,(const block_q8_K*)x,gate,up);
        return;
    }
    mk_exact_iq2_gu<<<dim3((unsigned)(L.n_ff/4),(unsigned)entries),128,0,(cudaStream_t)stream>>>(type,L,ptr,tok,nt,(const block_q8_K*)x,gate,up);
}
void mk_cpu_pool_down_launch(NativeExpertLayout L,const unsigned long long* ptr,const void* hq,
                            int entries,float* out,void* stream){
    mk_cpu_pool_down<<<dim3((unsigned)(L.n_embd/4),(unsigned)entries),128,0,(cudaStream_t)stream>>>(L,ptr,(const MkLegacyAct*)hq,out);
}
void mk_compatible_down_launch(NativeExpertLayout L,const unsigned long long* ptr,const void* hq,
                              int entries,float* out,void* stream){
    mk_compatible_down<<<dim3((unsigned)(L.n_embd/4),(unsigned)entries),128,0,(cudaStream_t)stream>>>(L,ptr,(const block_q8_0*)hq,out);
}
}
