// Isolated SM61 timing harness. Synthetic packed weights at checkpoint matrix shapes.
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#define CK(x) do { auto e=(x); if(e!=cudaSuccess) throw std::runtime_error(std::string(#x)+": "+cudaGetErrorString(e)); } while(0)
using namespace strata::kernels;
static constexpr int COPIES=64, SAMPLES=50;
static void one(const std::string &path, cudaStream_t stream) {
 std::ifstream bin(path+".bin",std::ios::binary),refin(path+".f32",std::ios::binary);
 if(!bin||!refin) throw std::runtime_error("missing fixture: "+path);
 int hdr[3];bin.read((char*)hdr,sizeof hdr);if(!bin)throw std::runtime_error("fixture header");
 const int type=hdr[0],rows=hdr[1],cols=hdr[2];
 const size_t bytes=native_mmvq_weight_bytes(type,cols,rows);
 std::vector<unsigned char>w(bytes);bin.read((char*)w.data(),bytes);if(!bin)throw std::runtime_error("weights truncated");
 std::vector<float> ref((size_t)rows*cols);refin.read((char*)ref.data(),ref.size()*4);if(!refin)throw std::runtime_error("reference truncated");
 unsigned char *dw;float *dx,*dy;void *dq;
 CK(cudaMalloc(&dw,bytes*COPIES));CK(cudaMalloc(&dx,(size_t)cols*8*4));CK(cudaMalloc(&dy,(size_t)rows*8*4));CK(cudaMalloc(&dq,(size_t)cols/32*36*8));
 for(int i=0;i<COPIES;i++)CK(cudaMemcpy(dw+i*bytes,w.data(),bytes,cudaMemcpyHostToDevice));
 std::mt19937 rng(7);std::normal_distribution<float> normal(0,1);std::vector<float>x((size_t)cols*8);
 for(auto &v:x)v=normal(rng);
 CK(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice));quantize_q8_1_rows(dx,8,cols,dq,stream);CK(cudaStreamSynchronize(stream));
 std::vector<float> single((size_t)rows*8),out(single.size());
 for(int j=0;j<8;j++)native_mmvq(type,dw,(unsigned char*)dq+(size_t)j*cols/32*36,dy+(size_t)j*rows,cols,rows,1,stream);
 CK(cudaStreamSynchronize(stream));CK(cudaMemcpy(single.data(),dy,single.size()*4,cudaMemcpyDeviceToHost));
 double error=0,denom=0;
 for(int j=0;j<8;j++)for(int r=0;r<rows;r++){
  double expected=0;for(int k=0;k<cols;k++)expected+=(double)ref[(size_t)r*cols+k]*x[(size_t)j*cols+k];
  float got=single[(size_t)j*rows+r];if(!std::isfinite(got)||!std::isfinite(expected))throw std::runtime_error("nonfinite");
  error+=std::abs(got-expected);denom+=std::abs(expected);
 }
 const double rel=error/(denom+1e-30);if(!(rel<0.02))throw std::runtime_error("float reference mismatch");
 auto launch=[&](int m,int copy){native_mmvq(type,dw+(copy%COPIES)*bytes,dq,dy,cols,rows,m,stream);};
 for(int m=1;m<=8;m++){
  launch(m,0);CK(cudaStreamSynchronize(stream));CK(cudaMemcpy(out.data(),dy,(size_t)rows*m*4,cudaMemcpyDeviceToHost));
  if(std::memcmp(out.data(),single.data(),(size_t)rows*m*4))throw std::runtime_error("multi-column mismatch");
  for(int i=0;i<512;i++)launch(m,i);CK(cudaStreamSynchronize(stream));
  cudaGraph_t graph;cudaGraphExec_t exec;
  CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
  for(int i=0;i<COPIES;i++)launch(m,i);
  CK(cudaStreamEndCapture(stream,&graph));CK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
  cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));
  for(int mode=0;mode<2;mode++){
   std::vector<double> us;
   for(int sample=0;sample<SAMPLES;sample++){
    CK(cudaEventRecord(a,stream));
    if(mode)CK(cudaGraphLaunch(exec,stream));else for(int i=0;i<COPIES;i++)launch(m,i);
    CK(cudaEventRecord(b,stream));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));us.push_back(ms*1000/COPIES);
   }
   CK(cudaMemcpy(out.data(),dy,(size_t)rows*m*4,cudaMemcpyDeviceToHost));if(std::memcmp(out.data(),single.data(),(size_t)rows*m*4))throw std::runtime_error("timed output mismatch");
   std::sort(us.begin(),us.end());double med=(us[24]+us[25])/2,p95=us[47];
   std::printf("%s,%d,%d,%d,%d,%s,%.6f,%.6f,%.6f,%.8f,%zu,%d,%d\n",path.c_str(),type,rows,cols,m,mode?"graph":"enqueue",med,p95,bytes/med/1000,rel,bytes,COPIES,SAMPLES);std::fflush(stdout);
  }
  CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(graph));
 }
 CK(cudaFree(dw));CK(cudaFree(dx));CK(cudaFree(dy));CK(cudaFree(dq));
}
int main(int argc,char **argv){try{
 if(argc<2)throw std::runtime_error("usage: p40_iq_speed FIXTURE_PREFIX ...");
 cudaDeviceProp prop;CK(cudaGetDeviceProperties(&prop,0));if(prop.major!=6||prop.minor!=1)throw std::runtime_error("SM61 only");
 std::printf("# device=%s L2=%d bytes; rotating weight storage=64*weight_bytes; activation quantization and RPC excluded\n",prop.name,prop.l2CacheSize);
 std::puts("case,type,rows,cols,M,mode,median_us,p95_us,effective_weight_GBs,relative_L1,weight_bytes,copies,samples");
 cudaStream_t stream;CK(cudaStreamCreate(&stream));for(int i=1;i<argc;i++)one(argv[i],stream);CK(cudaStreamDestroy(stream));return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}}
