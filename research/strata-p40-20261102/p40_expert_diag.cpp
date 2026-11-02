// Actual Strata grouped expert path, mode8, synthetic weights at Qwen H=2560 FF=640.
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
#define CK(x) do{auto e=(x);if(e!=cudaSuccess)throw std::runtime_error(std::string(#x)+": "+cudaGetErrorString(e));}while(0)
using namespace strata::kernels;
constexpr int H=2560,FF=640,TOP=10,MAXM=8,SAMPLES=50,REPS=16;
struct Mat{int type,rows,cols;std::vector<unsigned char>raw;std::vector<float>ref;};
Mat load(std::string path){Mat m;std::ifstream a(path+".bin",std::ios::binary),b(path+".f32",std::ios::binary);int h[3];a.read((char*)h,12);if(!a||!b)throw std::runtime_error("fixture open");m.type=h[0];m.rows=h[1];m.cols=h[2];size_t size=iq_row_bytes(m.type,m.cols)*m.rows;m.raw.resize(size);m.ref.resize((size_t)m.rows*m.cols);a.read((char*)m.raw.data(),size);b.read((char*)m.ref.data(),m.ref.size()*4);if(!a||!b)throw std::runtime_error("fixture truncated");return m;}
static void test(std::string path,std::string downpath,cudaStream_t s){
 Mat g=load(path),d=load(downpath);if(g.rows!=FF||g.cols!=H||d.rows!=H||d.cols!=FF)throw std::runtime_error("shape");
 auto L=native_expert_layout(g.type,d.type,H,FF);if(!native_expert_supported(g.type,d.type,H,FF))throw std::runtime_error("unsupported expert");
 std::vector<unsigned char>blob(L.bytes);std::memcpy(blob.data(),g.raw.data(),g.raw.size());
 size_t row=iq_row_bytes(g.type,H);for(int r=0;r<FF;r++)std::memcpy(blob.data()+L.up_off+r*row,g.raw.data()+((r+1)%FF)*row,row);std::memcpy(blob.data()+L.down_off,d.raw.data(),d.raw.size());
 unsigned char*weights;CK(cudaMalloc(&weights,L.bytes*TOP*MAXM));for(int i=0;i<TOP*MAXM;i++)CK(cudaMemcpy(weights+i*L.bytes,blob.data(),L.bytes,cudaMemcpyHostToDevice));
 std::mt19937 rng(11);std::normal_distribution<float>normal(0,.02f);std::vector<float>x(MAXM*H);for(auto&v:x)v=normal(rng);
 std::vector<double>ref(MAXM*H);for(int m=0;m<MAXM;m++){
  std::vector<double>h(FF);for(int f=0;f<FF;f++){double a=0,b=0;for(int k=0;k<H;k++){a+=(double)g.ref[(size_t)f*H+k]*x[m*H+k];b+=(double)g.ref[(size_t)((f+1)%FF)*H+k]*x[m*H+k];}h[f]=a/(1+std::exp(-a))*b;}
  for(int r=0;r<H;r++){double sum=0;for(int k=0;k<FF;k++)sum+=(double)d.ref[(size_t)r*FF+k]*h[k];ref[m*H+r]=sum;}
 }
 float *dx,*dy;void *dq,*scratch;unsigned long long*ptr;int32_t *starts,*count,*dst,*tok;
 CK(cudaMalloc(&dx,x.size()*4));CK(cudaMalloc(&dq,MAXM*H/32*36));CK(cudaMalloc(&dy,TOP*MAXM*H*4));CK(cudaMalloc(&scratch,native_expert_scratch_bytes(TOP*MAXM,FF)));
 CK(cudaMalloc(&ptr,TOP*MAXM*8));CK(cudaMalloc(&starts,(TOP*MAXM+1)*4));CK(cudaMalloc(&count,4));CK(cudaMalloc(&dst,TOP*MAXM*4));CK(cudaMalloc(&tok,TOP*MAXM*4));CK(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice));quantize_q8_1_rows(dx,MAXM,H,dq,s);CK(cudaStreamSynchronize(s));native_expert_set_mode(std::getenv("P40_EXPERT_MODE")?std::atoi(std::getenv("P40_EXPERT_MODE")):8,0);
 std::vector<float>one(TOP*MAXM*H),out(one.size());
 for(int m=1;m<=MAXM;m++)for(int diversity=0;diversity<2;diversity++){
  int groups=diversity?TOP*m:TOP,entries=TOP*m;
  std::vector<unsigned long long>hp(groups);std::vector<int32_t>hs(groups+1),hd(entries),ht(entries);
  for(int e=0;e<groups;e++){hp[e]=(unsigned long long)(weights+e*L.bytes);hs[e]=diversity?e:e*m;}hs[groups]=entries;
  for(int i=0;i<entries;i++){hd[i]=i;ht[i]=i%m;}
  CK(cudaMemcpy(ptr,hp.data(),hp.size()*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(starts,hs.data(),hs.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(count,&groups,4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dst,hd.data(),hd.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(tok,ht.data(),ht.size()*4,cudaMemcpyHostToDevice));
  auto launch=[&](){native_expert_grouped(L,ptr,starts,count,dst,tok,groups,entries,dq,scratch,dy,s,groups);};
  launch();CK(cudaStreamSynchronize(s));CK(cudaMemcpy(out.data(),dy,entries*H*4,cudaMemcpyDeviceToHost));double err=0,den=0;
  for(int i=0;i<entries;i++)for(int k=0;k<H;k++){float got=out[i*H+k];double expected=ref[(i%m)*H+k];if(!std::isfinite(got)||!std::isfinite(expected))throw std::runtime_error("nonfinite");err+=std::abs(got-expected);den+=std::abs(expected);}
  double rel=err/(den+1e-30);std::fprintf(stderr,"CHECK type=%d M=%d diversity=%d rel=%.9f\n",g.type,m,diversity,rel);if(!(rel<.03))throw std::runtime_error("independent float expert reference");if(std::getenv("P40_CORRECTNESS_ONLY"))continue;
  // Save the exact result and require every timed result to match it.
  std::copy(out.begin(),out.begin()+entries*H,one.begin());for(int i=0;i<128;i++)launch();CK(cudaStreamSynchronize(s));
  cudaGraph_t graph;cudaGraphExec_t exec;CK(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));for(int i=0;i<REPS;i++)launch();CK(cudaStreamEndCapture(s,&graph));CK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));
  for(int mode=0;mode<2;mode++){
   std::vector<double>us;for(int i=0;i<SAMPLES;i++){CK(cudaEventRecord(a,s));if(mode)CK(cudaGraphLaunch(exec,s));else for(int j=0;j<REPS;j++)launch();CK(cudaEventRecord(b,s));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));us.push_back(ms*1000/REPS);}
   CK(cudaMemcpy(out.data(),dy,entries*H*4,cudaMemcpyDeviceToHost));if(std::memcmp(out.data(),one.data(),entries*H*4))throw std::runtime_error("timed output changed");std::sort(us.begin(),us.end());
   std::printf("%s,%d,%d,%s,%d,%d,%s,%.6f,%.6f,%.8f,%zu\n",path.c_str(),g.type,m,diversity?"distinct":"shared",groups,entries,mode?"graph":"enqueue",(us[24]+us[25])/2,us[47],rel,L.bytes*groups);std::fflush(stdout);
  }
  CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(graph));
 }
 CK(cudaFree(weights));CK(cudaFree(dx));CK(cudaFree(dy));CK(cudaFree(dq));CK(cudaFree(scratch));CK(cudaFree(ptr));CK(cudaFree(starts));CK(cudaFree(count));CK(cudaFree(dst));CK(cudaFree(tok));
}
int main(int argc,char**argv){try{if(argc<3)throw std::runtime_error("usage: p40_expert_speed DOWN_PREFIX GU_PREFIX...");cudaDeviceProp p;CK(cudaGetDeviceProperties(&p,0));if(p.major!=6||p.minor!=1)throw std::runtime_error("SM61 only");std::printf("# %s; mode8 grouped expert; synthetic; 10 experts/row; H2560 FF640; no routing combine, RPC or input quantization timing\n",p.name);std::puts("case,gu_type,M,routing,groups,entries,mode,median_us,p95_us,relative_L1,distinct_weight_bytes");cudaStream_t s;CK(cudaStreamCreate(&s));for(int i=2;i<argc;i++)test(argv[i],argv[1],s);CK(cudaStreamDestroy(s));return 0;}catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}}
