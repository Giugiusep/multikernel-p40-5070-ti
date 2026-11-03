// Deterministic host reproduction of the experimental CUDA GU accumulation.
#include <cmath>
#include <fstream>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
extern "C" int mk_cpu_swiglu_hq(int,const float*,const float*,int,void*);
extern "C" int mk_cpu_down(int,const unsigned char*,const void*,int,float*);
template<class T> std::vector<T> read(const std::string&name,size_t n){std::vector<T>v(n);std::ifstream f(name,std::ios::binary);f.read((char*)v.data(),n*sizeof(T));if(!f)throw std::runtime_error("input "+name);return v;}
template<class T> void write(const std::string&name,const std::vector<T>&v){std::ofstream f(name,std::ios::binary);f.write((char*)v.data(),v.size()*sizeof(T));}
int main(int argc,char**argv){try{
 if(argc!=2)return 2;std::string p=argv[1];constexpr int rows=4,H=2560,FF=640;
 auto g=read<float>(p+"/gate.f32",FF*H),u=read<float>(p+"/up.f32",FF*H),x=read<float>(p+"/cpu-M4.f32.qx",rows*H);
 auto blob=read<unsigned char>(p+"/weights.bin",1305600);
 std::vector<float>gate(rows*FF),up(rows*FF);
 for(int t=0;t<rows;t++)for(int r=0;r<FF;r++){
  float a[32]={},b[32]={},ta[32],tb[32];
  for(int block=0;block<10;block++)for(int j=0;j<8;j++)for(int lane=0;lane<32;lane++){
   int k=block*256+j*32+lane;a[lane]=std::fma(g[r*H+k],x[t*H+k],a[lane]);b[lane]=std::fma(u[r*H+k],x[t*H+k],b[lane]);
  }
  for(int off=16;off;off>>=1){for(int l=0;l<32;l++){ta[l]=a[l]+a[l^off];tb[l]=b[l]+b[l^off];}for(int l=0;l<32;l++){a[l]=ta[l];b[l]=tb[l];}}
  gate[t*FF+r]=a[0];up[t*FF+r]=b[0];
 }
 std::vector<unsigned char>hq(rows*680);std::vector<float>out(rows*H);
 if(mk_cpu_swiglu_hq(16,gate.data(),up.data(),rows,hq.data())||mk_cpu_down(16,blob.data(),hq.data(),rows,out.data()))return 1;
 write(p+"/emulated.hq",hq);write(p+"/emulated.out",out);return 0;
 }catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
