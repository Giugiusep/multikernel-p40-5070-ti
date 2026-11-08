// Bounded DeepSeek expert probe. No model filesystem, driver or kernel ownership.
#include "ds4_iq.cuh"
#include <cuda_runtime.h>
#include <linux/vm_sockets.h>
#include <sys/socket.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>
extern "C" void ds_exact_down(int,const uint8_t*,int,int,int,const int8_t*,const float*,float*,cudaStream_t);
namespace iq=strata::ds4::gguf::iq;
constexpr unsigned MAGIC=0x4458454d,VER=2,PORT=6011,MAX_BODY=16*1024*1024;
constexpr int D=5120,F=2304,M=8,STRIDE=8;
struct Header{unsigned magic,version,op,status,sequence,bytes;};
static_assert(sizeof(Header)==24);
struct CudaFault:std::runtime_error{using std::runtime_error::runtime_error;};
void ck(cudaError_t e){if(e!=cudaSuccess)throw CudaFault(cudaGetErrorString(e));}
void io(int fd,void* data,size_t n,bool tx){auto*p=(char*)data;unsigned retries=0;while(n){auto count=std::min(n,size_t(32768));ssize_t k=tx?send(fd,p,count,MSG_NOSIGNAL):recv(fd,p,count,0);if(k<0&&errno==EINTR)continue;if(k<0&&tx&&(errno==ENOSPC||errno==EAGAIN)&&++retries<=1000){usleep(100);continue;}if(k==0)throw std::runtime_error("peer EOF");if(k<0)throw std::runtime_error(strerror(errno));p+=k;n-=k;retries=0;if(tx&&n)usleep(100);}}
unsigned word(const std::vector<unsigned char>&b,size_t i){unsigned v;if((i+1)*4>b.size())throw std::runtime_error("short payload");memcpy(&v,b.data()+4*i,4);return v;}
void append(std::vector<unsigned char>&b,const void*p,size_t n){auto*q=(const unsigned char*)p;b.insert(b.end(),q,q+n);}
template<class T>struct Device{T*p=nullptr;explicit Device(size_t n){ck(cudaMalloc(&p,n*sizeof(T)));}~Device(){if(p)cudaFree(p);}Device(const Device&)=delete;};
struct Expert{unsigned char*p=nullptr;unsigned tg=0,td=0;size_t bytes=0,gu=0;bool compact=false;~Expert(){if(p)cudaFree(p);}};
__global__ void quant_h(const float*h,int8_t*q,float*sc,int rows){
 int row=blockIdx.y,b=blockIdx.x,j=threadIdx.x;
 __shared__ float scale,inv;
 size_t base=size_t(row*STRIDE)*F+b*256;
 if(j==0){float amax=0,m=0;for(int i=0;i<256;i++){float v=h[base+i];if(fabsf(v)>amax){amax=fabsf(v);m=v;}}inv=amax?-127.f/m:0.f;scale=amax?1.f/inv:0.f;}
 __syncthreads();
 for(int i=j;i<256;i+=32)q[base+i]=inv?min(127,__float2int_rn(inv*h[base+i])):0;
 if(j<8)sc[size_t(row*STRIDE)*(F/32)+b*8+j]=scale;
}

class Worker{
 std::map<std::pair<unsigned,unsigned>,std::unique_ptr<Expert>> cache;
 size_t resident=0;static constexpr size_t LIMIT=1024ull*1024*1024;
 Device<int8_t>x{M*D},hq{M*STRIDE*F};Device<float>xd{M*D/32},h{M*STRIDE*F},hd{M*STRIDE*F/32},y{M*STRIDE*D};
 Device<unsigned long long>ptr{1};Device<int>start{2},ent{M},ng{1};
 cudaStream_t stream{};cudaEvent_t events[4]{};
public:
 Worker(){ck(cudaStreamCreate(&stream));for(auto&e:events)ck(cudaEventCreate(&e));}
 ~Worker(){for(auto e:events)cudaEventDestroy(e);cudaStreamDestroy(stream);}
 std::vector<unsigned char> run(unsigned op,const std::vector<unsigned char>&b){
 if(op==0){if(!b.empty())throw std::runtime_error("INFO payload must be empty");const char*s="{\"contract\":\"deepseek-iq-q8k-v2\",\"hidden\":5120,\"ffn\":2304,\"max_rows\":8,\"cache_bytes\":1073741824,\"model_io\":false}";return {s,s+strlen(s)};}
 if(op==5){if(b.size()<32)throw std::runtime_error("short LOAD_DOWN");unsigned l=word(b,0),e=word(b,1),tg=word(b,2),td=word(b,3);if(l>=40||e>=384||word(b,4)!=D||word(b,5)!=F||word(b,6)!=0||!(td==16||td==17||td==22))throw std::runtime_error("unsupported down layout");size_t bytes=D*iq::row_bytes(td,F);if(word(b,7)!=bytes||b.size()!=32+bytes)throw std::runtime_error("LOAD_DOWN byte count");if(cache.count({l,e}))throw std::runtime_error("duplicate expert");if(resident+bytes>LIMIT)throw std::runtime_error("resident byte limit");auto p=std::make_unique<Expert>();p->td=td;p->tg=tg;p->bytes=bytes;p->compact=true;ck(cudaMalloc(&p->p,bytes));ck(cudaMemcpy(p->p,b.data()+32,bytes,cudaMemcpyHostToDevice));cache.emplace(std::make_pair(l,e),std::move(p));resident+=bytes;return {};}
 if(op==1){if(b.size()<32)throw std::runtime_error("short LOAD");unsigned l=word(b,0),e=word(b,1),tg=word(b,2),td=word(b,3);if(l>=40||e>=384||word(b,4)!=D||word(b,5)!=F||(tg!=16&&tg!=17&&tg!=22)||(td!=16&&td!=17&&td!=22&&td!=18))throw std::runtime_error("unsupported expert layout");size_t gu=F*iq::row_bytes(tg,D),dn=D*iq::row_bytes(td,F),bytes=gu*2+dn;if(word(b,6)!=gu||word(b,7)!=dn||b.size()!=32+bytes)throw std::runtime_error("LOAD byte count");if(cache.count({l,e}))throw std::runtime_error("duplicate expert");if(resident+bytes>LIMIT)throw std::runtime_error("resident byte limit");auto p=std::make_unique<Expert>();p->tg=tg;p->td=td;p->gu=gu;p->bytes=bytes;ck(cudaMalloc(&p->p,bytes));ck(cudaMemcpy(p->p,b.data()+32,bytes,cudaMemcpyHostToDevice));cache.emplace(std::make_pair(l,e),std::move(p));resident+=bytes;return {};}
 if(op==3){if(!b.empty())throw std::runtime_error("CLEAR payload must be empty");cache.clear();resident=0;return {};}
 if(op!=2&&op!=4)throw std::runtime_error("unknown operation");if(b.size()<16)throw std::runtime_error("short RUN");unsigned l=word(b,0),e=word(b,1),rows=word(b,2),debug=word(b,3);int input_dim=op==4?F:D;if(rows<1||rows>M||debug>1||b.size()!=16+rows*(input_dim+input_dim/32*4))throw std::runtime_error("RUN dimensions/bytes");auto it=cache.find({l,e});if(it==cache.end())throw std::runtime_error("expert not resident");auto&ex=*it->second;if(ex.compact&&op!=4)throw std::runtime_error("down-only expert contract");
 const auto*q=(const int8_t*)(b.data()+16);std::vector<float>sc(rows*input_dim/32);memcpy(sc.data(),b.data()+16+rows*input_dim,sc.size()*4);for(float s:sc)if(!std::isfinite(s))throw std::runtime_error("invalid activation scale");for(unsigned i=0;i<rows*input_dim;i++)if(q[i]==-128)throw std::runtime_error("invalid activation code");
 int st[2]={0,(int)rows},idx[M];for(unsigned i=0;i<rows;i++)idx[i]=i*STRIDE;int one=1;auto address=(unsigned long long)ex.p;
 ck(cudaMemcpyAsync(x.p,q,rows*input_dim,cudaMemcpyHostToDevice,stream));ck(cudaMemcpyAsync(xd.p,sc.data(),sc.size()*4,cudaMemcpyHostToDevice,stream));ck(cudaMemcpyAsync(start.p,st,sizeof(st),cudaMemcpyHostToDevice,stream));ck(cudaMemcpyAsync(ent.p,idx,rows*4,cudaMemcpyHostToDevice,stream));ck(cudaMemcpyAsync(ng.p,&one,4,cudaMemcpyHostToDevice,stream));ck(cudaMemcpyAsync(ptr.p,&address,8,cudaMemcpyHostToDevice,stream));
 iq::Args a{};a.grp_ptr=ptr.p;a.grp_start=start.p;a.grp_ent=ent.p;a.n_groups=ng.p;a.tr[0]=0;a.tr[1]=ex.gu;a.tr[2]=2*ex.gu;a.n_embd=D;a.n_ff=F;a.limit=10.f;a.tg=ex.tg;a.td=ex.td;a.xq=x.p;a.xd=xd.p;a.h=h.p;a.hq=hq.p;a.hd=hd.p;a.part=y.p;
 if(op==4){if(debug||!(ex.td==16||ex.td==17||ex.td==22))throw std::runtime_error("exact down contract");ck(cudaEventRecord(events[0],stream));ds_exact_down(ex.td,ex.p+2*ex.gu,F,D,rows,x.p,xd.p,y.p,stream);ck(cudaGetLastError());for(int k=1;k<4;k++)ck(cudaEventRecord(events[k],stream));}else{ ck(cudaEventRecord(events[0],stream));if(!iq::gate_up(a,1,rows,stream))throw std::runtime_error("GU refused");ck(cudaEventRecord(events[1],stream));quant_h<<<dim3(F/256,rows),32,0,stream>>>(h.p,hq.p,hd.p,rows);ck(cudaGetLastError());ck(cudaEventRecord(events[2],stream));if(!iq::down(a,1,rows,stream))throw std::runtime_error("down refused");ck(cudaEventRecord(events[3],stream));}ck(cudaEventSynchronize(events[3]));float us[3];for(int i=0;i<3;i++){ck(cudaEventElapsedTime(us+i,events[i],events[i+1]));us[i]*=1000;}
 std::vector<float>out(rows*D),host_h(debug?rows*F:0),host_hd(debug?rows*F/32:0);std::vector<int8_t>host_hq(debug?rows*F:0);
 for(unsigned i=0;i<rows;i++){ck(cudaMemcpy(out.data()+size_t(i)*D,y.p+size_t(i*STRIDE)*D,D*4,cudaMemcpyDeviceToHost));if(debug){ck(cudaMemcpy(host_h.data()+size_t(i)*F,h.p+size_t(i*STRIDE)*F,F*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(host_hq.data()+size_t(i)*F,hq.p+size_t(i*STRIDE)*F,F,cudaMemcpyDeviceToHost));ck(cudaMemcpy(host_hd.data()+size_t(i)*F/32,hd.p+size_t(i*STRIDE)*F/32,F/32*4,cudaMemcpyDeviceToHost));}}
 for(float v:out)if(!std::isfinite(v))throw CudaFault("nonfinite expert result");
 std::vector<unsigned char>reply;unsigned meta[4]={rows,debug,D,F};append(reply,meta,16);append(reply,us,12);append(reply,out.data(),out.size()*4);if(debug){append(reply,host_h.data(),host_h.size()*4);append(reply,host_hq.data(),host_hq.size());append(reply,host_hd.data(),host_hd.size()*4);}return reply;
 }
};
int main(){try{cudaDeviceProp p{};ck(cudaGetDeviceProperties(&p,0));if(p.major!=6||p.minor!=1)throw std::runtime_error("SM61 device required");int fd=socket(AF_VSOCK,SOCK_STREAM,0),type=1;if(fd<0||setsockopt(fd,AF_VSOCK,9,&type,4))throw std::runtime_error("Multikernel socket selection");sockaddr_vm a{};a.svm_family=AF_VSOCK;a.svm_cid=VMADDR_CID_ANY;a.svm_port=PORT;if(bind(fd,(sockaddr*)&a,sizeof(a))||listen(fd,1))throw std::runtime_error("listen6011");printf("DEEPSEEK_WORKER_READY port=6011 GPU=%s cache=1GiB\n",p.name);fflush(stdout);
 for(;;){int peer=accept(fd,nullptr,nullptr);if(peer<0){if(errno==EINTR)continue;throw std::runtime_error("accept");}timeval t{120,0};setsockopt(peer,SOL_SOCKET,SO_RCVTIMEO,&t,sizeof(t));setsockopt(peer,SOL_SOCKET,SO_SNDTIMEO,&t,sizeof(t));try{Worker w;for(;;){Header h{};io(peer,&h,24,false);if(h.magic!=MAGIC||h.version!=VER||h.status||h.bytes>MAX_BODY)throw std::runtime_error("invalid frame");std::vector<unsigned char>b(h.bytes),r;io(peer,b.data(),b.size(),false);if(getenv("DS4_MK_TRACE"))fprintf(stderr,"worker begin seq=%u op=%u bytes=%u\n",h.sequence,h.op,h.bytes);unsigned status=0;try{r=w.run(h.op,b);}catch(const CudaFault&){throw;}catch(const std::exception&e){status=1;append(r,e.what(),strlen(e.what()));}Header reply{MAGIC,VER,h.op,status,h.sequence,(unsigned)r.size()};io(peer,&reply,24,true);io(peer,r.data(),r.size(),true);if(getenv("DS4_MK_TRACE"))fprintf(stderr,"worker end seq=%u op=%u status=%u bytes=%zu\n",h.sequence,h.op,status,r.size());}}catch(const std::exception&e){fprintf(stderr,"connection: %s\n",e.what());fflush(stderr);}close(peer);}
 }catch(const std::exception&e){fprintf(stderr,"FATAL %s\n",e.what());return 1;}}
