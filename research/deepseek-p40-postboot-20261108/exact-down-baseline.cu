// Reuse pinned GGML weight decoders; emulate the CPU's eight accumulation lanes.
#include "/home/shiba/multikernel-experiment/research/Strata-dsv41-experiment/src/kernels/cuda/iq_kernels.cu"
namespace strata::kernels {
namespace {
__global__ void ds_exact_down_kernel(int type,const uint8_t*w,int n,int nout,const int8_t*q,const float*sc,float*out){
 __shared__ float decoded[4][256];
 int lane=threadIdx.x&31,warp=threadIdx.x>>5,row=blockIdx.x*4+warp,t=blockIdx.y;
 int bytes=type==16?66:type==17?74:82;
 const uint8_t*wr=w+(size_t)row*(n/256)*bytes;
 float acc=0;
 for(int b=0;b<n/256;b++){
  dq_dispatch<float>(type,wr,b,decoded[warp],lane);__syncthreads();
  float wd=__half2float(*(const __half*)(wr+b*bytes));float base=__fmul_rn(wd,.125f);int sum=0;
  if(lane<8){for(int k=lane*4;k<256;k+=32)for(int j=0;j<4;j++){
   int wi=base==0?0:__float2int_rn(decoded[warp][k+j]/base);
   sum+=wi*(int)q[(size_t)t*n+b*256+k+j];
  }float scale=__fmul_rn(wd,sc[(size_t)t*(n/32)+b*8]);acc=__fmaf_rn(scale,(float)sum,acc);}
  __syncthreads();
 }
 for(int off=4;off;off>>=1)acc=__fadd_rn(acc,__shfl_xor_sync(0xffffffff,acc,off));
 if(lane==0)out[(size_t)t*8*nout+row]=__fmul_rn(acc,.125f);
}
}
}
extern "C" void ds_exact_down(int type,const uint8_t*w,int n,int nout,int rows,const int8_t*q,const float*sc,float*out,cudaStream_t s){
 strata::kernels::ds_exact_down_kernel<<<dim3(nout/4,rows),128,0,s>>>(type,w,n,nout,q,sc,out);
}
