// Same CPU-pool integer/FMA contract, with weight decoding shared across rows.
namespace strata::kernels {
namespace {
template<int TYPE, int T>
__global__ void mk_group_gu(NativeExpertLayout L,const unsigned long long* __restrict__ ptr,
 const int32_t* __restrict__ token,const int32_t* __restrict__ starts,const int32_t* __restrict__ dst,
 const block_q8_K* __restrict__ x,float* __restrict__ gate,float* __restrict__ up){
 int lane=threadIdx.x&31,row=blockIdx.x*4+(threadIdx.x>>5),group=blockIdx.y;
 int first=starts[group],n=starts[group+1]-first;
 bool wide=n>=2;int lanes=wide?16:8;
 int entry[T]={};const block_q8_K* activation[T]={};
 #pragma unroll
 for(int t=0;t<T;t++)if(t<n){entry[t]=dst[first+t];activation[t]=x+(size_t)token[entry[t]]*10;}
 const auto* blob=(const uint8_t*)ptr[entry[0]];
 const auto* gr=blob+row*L.gu_row;const auto* ur=blob+L.up_off+row*L.gu_row;
 constexpr int bytes=TYPE==16?66:82;
 float g[T]={},u[T]={};
 for(int b=0;b<10;b++){
  int gi[T]={},ui[T]={};
  if(lane<lanes){
   for(int chunk=0;chunk<256/(lanes*4);chunk++){
    int k=chunk*lanes*4+lane*4;
    int4 gw=mk_iq2_coeff4<TYPE>(gr+b*bytes,k),uw=mk_iq2_coeff4<TYPE>(ur+b*bytes,k);
    #pragma unroll
    for(int t=0;t<T;t++)if(t<n){
     const auto& a=activation[t][b];
     gi[t]+=gw.x*(int)a.qs[k];ui[t]+=uw.x*(int)a.qs[k];
     gi[t]+=gw.y*(int)a.qs[k+1];ui[t]+=uw.y*(int)a.qs[k+1];
     gi[t]+=gw.z*(int)a.qs[k+2];ui[t]+=uw.z*(int)a.qs[k+2];
     gi[t]+=gw.w*(int)a.qs[k+3];ui[t]+=uw.w*(int)a.qs[k+3];
    }
   }
   float gd=__half2float(*(const __half*)(gr+b*bytes));
   float ud=__half2float(*(const __half*)(ur+b*bytes));
   #pragma unroll
   for(int t=0;t<T;t++)if(t<n){
    float ad=activation[t][b].d;
    float gs=__fmul_rn(wide?__fmul_rn(gd,.125f):gd,ad);
    float us=__fmul_rn(wide?__fmul_rn(ud,.125f):ud,ad);
    g[t]=__fmaf_rn(gs,(float)gi[t],g[t]);u[t]=__fmaf_rn(us,(float)ui[t],u[t]);
   }
  }
 }
 #pragma unroll
 for(int t=0;t<T;t++)if(t<n){
  for(int off=lanes/2;off;off>>=1){
   g[t]=__fadd_rn(g[t],__shfl_xor_sync(0xffffffff,g[t],off));
   u[t]=__fadd_rn(u[t],__shfl_xor_sync(0xffffffff,u[t],off));
  }
  if(lane==0){
   int e=entry[t];gate[e*L.n_ff+row]=wide?g[t]:__fmul_rn(g[t],.125f);
   up[e*L.n_ff+row]=wide?u[t]:__fmul_rn(u[t],.125f);
  }
 }
}
template<int T>
__global__ void mk_group_down(NativeExpertLayout L,const unsigned long long* __restrict__ ptr,
 const int32_t* __restrict__ starts,const int32_t* __restrict__ dst,const MkLegacyAct* __restrict__ hq,float* __restrict__ out){
 int lane=threadIdx.x&31,row=blockIdx.x*4+(threadIdx.x>>5),group=blockIdx.y;
 int first=starts[group],n=starts[group+1]-first;
 int entry[T]={};const MkLegacyAct* activation[T]={};
 #pragma unroll
 for(int t=0;t<T;t++)if(t<n){entry[t]=dst[first+t];activation[t]=hq+entry[t];}
 const auto* w=(const block_q2_0*)((const uint8_t*)ptr[entry[0]]+L.down_off+row*L.d_row);
 float acc[T]={},corr[T]={};
 for(int b0=0;b0<10;b0+=8){
  int nb=min(8,10-b0);
  if(lane<2*nb){
   float d=__half2float(w[b0+lane/2].d);
   #pragma unroll
   for(int t=0;t<T;t++)if(t<n)corr[t]=__fmaf_rn(d,activation[t]->hx[2*b0+lane],corr[t]);
  }
  for(int i=0;i<nb;i++)if(lane<16){
   int b=b0+i,byte=w[b].qs[lane];float d=__half2float(w[b].d);
   int q0=byte&3,q1=(byte>>2)&3,q2=(byte>>4)&3,q3=(byte>>6)&3;
   #pragma unroll
   for(int t=0;t<T;t++)if(t<n){
    const auto& h=*activation[t];
    int k=b*64+lane*4;
    int dot=q0*(int)h.q[k]+q1*(int)h.q[k+1]+q2*(int)h.q[k+2]+q3*(int)h.q[k+3];
    float scale=__fmul_rn(d,h.scale[2*b+(lane>=8)]);
    acc[t]=__fmaf_rn(scale,(float)dot,acc[t]);
   }
  }
 }
 #pragma unroll
 for(int t=0;t<T;t++)if(t<n){
  for(int off=8;off;off>>=1){
   acc[t]=__fadd_rn(acc[t],__shfl_xor_sync(0xffffffff,acc[t],off));
   corr[t]=__fadd_rn(corr[t],__shfl_xor_sync(0xffffffff,corr[t],off));
  }
  if(lane==0)out[entry[t]*L.n_embd+row]=__fsub_rn(acc[t],corr[t]);
 }
}
template<int T>
void launch_group_gu(int type,NativeExpertLayout L,const unsigned long long* __restrict__ ptr,const int32_t* tok,
 const int32_t* __restrict__ starts,const int32_t* __restrict__ dst,const void* x,int groups,float* __restrict__ gate,float* __restrict__ up,void* stream){
 dim3 grid((unsigned)(L.n_ff/4),(unsigned)groups);
 if(type==16)mk_group_gu<16,T><<<grid,128,0,(cudaStream_t)stream>>>(L,ptr,tok,starts,dst,(const block_q8_K*)x,gate,up);
 else mk_group_gu<22,T><<<grid,128,0,(cudaStream_t)stream>>>(L,ptr,tok,starts,dst,(const block_q8_K*)x,gate,up);
}
}
void mk_group_gu_launch(int type,NativeExpertLayout L,const unsigned long long* __restrict__ ptr,const int32_t* tok,
 const int32_t* __restrict__ starts,const int32_t* __restrict__ dst,const void* x,int groups,int max_rows,float* __restrict__ gate,float* __restrict__ up,void* stream){
 if(max_rows<=2)launch_group_gu<2>(type,L,ptr,tok,starts,dst,x,groups,gate,up,stream);
 else if(max_rows<=4)launch_group_gu<4>(type,L,ptr,tok,starts,dst,x,groups,gate,up,stream);
 else launch_group_gu<8>(type,L,ptr,tok,starts,dst,x,groups,gate,up,stream);
}
void mk_group_down_launch(NativeExpertLayout L,const unsigned long long* __restrict__ ptr,const int32_t* __restrict__ starts,
 const int32_t* __restrict__ dst,const void* hq,int groups,int max_rows,float* __restrict__ out,void* stream){
 dim3 grid((unsigned)(L.n_embd/4),(unsigned)groups);
 if(max_rows<=2)mk_group_down<2><<<grid,128,0,(cudaStream_t)stream>>>(L,ptr,starts,dst,(const MkLegacyAct*)hq,out);
 else if(max_rows<=4)mk_group_down<4><<<grid,128,0,(cudaStream_t)stream>>>(L,ptr,starts,dst,(const MkLegacyAct*)hq,out);
 else mk_group_down<8><<<grid,128,0,(cudaStream_t)stream>>>(L,ptr,starts,dst,(const MkLegacyAct*)hq,out);
}
}
