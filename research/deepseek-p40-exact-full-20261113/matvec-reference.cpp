#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
int main(int argc,char**argv){if(argc!=8)return 2;int ty=atoi(argv[1]),n=atoi(argv[2]),rows=atoi(argv[3]),m=atoi(argv[4]);if(n%256||rows%16||m<1||m>8)return 3;ggml_cpu_init();auto*dot=ggml_get_type_traits_cpu((ggml_type)ty);auto*q=ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);if(!dot||!dot->vec_dot||!q||!q->from_float)return 4;size_t rb=ggml_row_size((ggml_type)ty,n),ab=ggml_row_size(GGML_TYPE_Q8_K,n);std::vector<unsigned char>w(rows*rb),a(m*ab);std::vector<float>x(m*n),y(m*rows);FILE*f=fopen(argv[5],"rb");if(!f||fread(w.data(),1,w.size(),f)!=w.size())return 5;fclose(f);f=fopen(argv[6],"rb");if(!f||fread(x.data(),4,x.size(),f)!=x.size())return 6;fclose(f);for(int t=0;t<m;t++){q->from_float(x.data()+t*n,a.data()+t*ab,n);for(int r=0;r<rows;r++)dot->vec_dot(n,&y[t*rows+r],0,w.data()+r*rb,0,a.data()+t*ab,0,1);}f=fopen(argv[7],"wb");fwrite(y.data(),4,y.size(),f);fclose(f);f=fopen((std::string(argv[7])+".q8k").c_str(),"wb");fwrite(a.data(),1,a.size(),f);return fclose(f);}
