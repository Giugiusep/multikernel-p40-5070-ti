// Packed IQ2 weights; CPU-compatible integer grouping and eight FP32 accumulation lanes.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"
namespace {
__device__ __forceinline__ int2 coeff4(int type,const uint8_t*b,int k){
 int ib=k/32,il=(k%32)/8,j=k%8;uint64_t grid;int factor;uint8_t signs;
 if(type==16){const auto*w=(const block_iq2_xxs*)b;const auto*q=w->qs+4*ib;uint32_t aux=(uint32_t)q[2]|((uint32_t)q[3]<<16);grid=iq2xxs_grid[((const uint8_t*)q)[il]];factor=2*(int)(aux>>28)+1;signs=ksigns_iq2xs[(aux>>(7*il))&127];}
 else if(type==17){const auto*w=(const block_iq2_xs*)b;unsigned q=w->qs[4*ib+il];grid=iq2xs_grid[q&511];factor=2*((w->scales[ib]>>(4*(il/2)))&15)+1;signs=ksigns_iq2xs[q>>9];}
 else{const auto*w=(const block_iq2_s*)b;int index=w->qs[4*ib+il]|((w->qh[ib]<<(8-2*il))&0x300);grid=iq2s_grid[index];factor=2*((w->scales[ib]>>(4*(il/2)))&15)+1;signs=w->qs[32+4*ib+il];}
 uint32_t q=(uint32_t)(grid>>(8*j));int a=int(q&255),c=int((q>>8)&255),d=int((q>>16)&255),e=int(q>>24);
 a=signs&(1<<j)?-a:a;c=signs&(2<<j)?-c:c;d=signs&(4<<j)?-d:d;e=signs&(8<<j)?-e:e;unsigned packed=(uint8_t)a|((unsigned)(uint8_t)c<<8)|((unsigned)(uint8_t)d<<16)|((unsigned)(uint8_t)e<<24);return make_int2((int)packed,factor);
}
__global__ void exact_down(int type,const uint8_t*w,int n,int nout,const int8_t*q,const float*sc,float*out){
 int lane=threadIdx.x&7,warp=threadIdx.x>>5,row=blockIdx.x*16+warp*4+((threadIdx.x&31)>>3),t=blockIdx.y;int bytes=type==16?66:type==17?74:82;
 const uint8_t*wr=w+(size_t)row*(n/256)*bytes;float acc=0;
 if(lane<8)for(int b=0;b<n/256;b++){
  int sum=0;const int8_t*x=q+(size_t)t*n+b*256;
  for(int k=lane*4;k<256;k+=32){int2 v=coeff4(type,wr+b*bytes,k);sum+=v.y*__dp4a(v.x,*(const int*)(x+k),0);}
  float wd=__half2float(*(const __half*)(wr+b*bytes));float scale=__fmul_rn(wd,sc[(size_t)t*(n/32)+b*8]);acc=__fmaf_rn(scale,(float)sum,acc);
 }
 for(int off=4;off;off>>=1)acc=__fadd_rn(acc,__shfl_xor_sync(0xffffffff,acc,off));
 if(lane==0)out[(size_t)t*8*nout+row]=__fmul_rn(acc,.125f);
}
}
extern "C" void ds_exact_down(int type,const uint8_t*w,int n,int nout,int rows,const int8_t*q,const float*sc,float*out,cudaStream_t s){exact_down<<<dim3(nout/16,rows),128,0,s>>>(type,w,n,nout,q,sc,out);}
