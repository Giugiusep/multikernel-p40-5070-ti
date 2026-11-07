// Independent ggml decoder for bounded expert fixtures, not GPU arithmetic.
#include "ggml.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc,char**argv){if(argc!=6)return 2;int t=atoi(argv[1]),r=atoi(argv[2]),k=atoi(argv[3]);if(r<1||r>5120||k<1||k>5120)return 2;auto*tr=ggml_get_type_traits((ggml_type)t);if(!tr||!tr->to_float)return 3;FILE*in=fopen(argv[4],"rb"),*out=fopen(argv[5],"wb");if(!in||!out)return 4;size_t rb=ggml_row_size((ggml_type)t,k);std::vector<unsigned char>b(rb);std::vector<float>x(k);for(int i=0;i<r;i++){if(fread(b.data(),1,rb,in)!=rb)return 5;tr->to_float(b.data(),x.data(),k);if(fwrite(x.data(),4,k,out)!=(size_t)k)return 6;}fclose(in);return fclose(out)?7:0;}
