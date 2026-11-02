#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>
#include <algorithm>
int main(int argc,char**argv){if(argc!=2)return 2;const int H=2560,F=640;std::ifstream in(argv[1],std::ios::binary);std::vector<float>w(H*F),x(H*8);in.read((char*)w.data(),w.size()*4);if(!in)return 3;std::mt19937 rng(11);std::normal_distribution<float>dist(0,.02f);for(auto&v:x)v=dist(rng);double maxh=0;for(int m=0;m<8;m++)for(int r=0;r<F;r++){double g=0,u=0;for(int k=0;k<H;k++){g+=(double)w[r*H+k]*x[m*H+k];u+=(double)w[((r+1)%F)*H+k]*x[m*H+k];}maxh=std::max(maxh,std::abs(g/(1+std::exp(-g))*u));}double scale=maxh/127;_Float16 half=(_Float16)scale;std::printf("max_float_reference_h=%.12g max_q81_scale=%.12g rounded_half_scale=%.12g\n",maxh,scale,(double)half);return half==0?0:1;}
