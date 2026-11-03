#include <fstream>
#include <vector>
#include <cstdio>
#include <cstdlib>
extern "C" int mk_cpu_reference(int,const unsigned char*,const float*,int,float*);
extern "C" int mk_cpu_pool_reference(int,const unsigned char*,const float*,int,float*);
extern "C" int mk_cpu_quantized_x(int,const float*,int,float*);
int main(int argc,char**argv){
 if(argc!=6){fprintf(stderr,"usage: cpu_reference GU ROWS BLOB X OUTPUT\n");return 2;}
 int gu=atoi(argv[1]),rows=atoi(argv[2]);if(rows<1||rows>8)return 2;
 std::ifstream wf(argv[3],std::ios::binary|std::ios::ate),xf(argv[4],std::ios::binary);
 if(!wf||!xf)return 2;auto n=wf.tellg();if(n<1||n>2000000)return 2;
 std::vector<unsigned char>w((size_t)n);wf.seekg(0);wf.read((char*)w.data(),n);
 std::vector<float>x(rows*2560),out(x.size()),decoded(x.size());xf.read((char*)x.data(),x.size()*4);if(!wf||!xf)return 2;
 bool pool=getenv("MK_CPU_POOL_ORACLE")||std::string(argv[0]).find("cpu_reference_pool")!=std::string::npos;
 int code=pool?mk_cpu_pool_reference(gu,w.data(),x.data(),rows,out.data()):mk_cpu_reference(gu,w.data(),x.data(),rows,out.data());
 if(code||mk_cpu_quantized_x(gu,x.data(),rows,decoded.data()))return 1;
 std::ofstream f(argv[5],std::ios::binary);f.write((char*)out.data(),out.size()*4);
 std::ofstream q(std::string(argv[5])+".qx",std::ios::binary);q.write((char*)decoded.data(),decoded.size()*4);return f&&q?0:1;
}
