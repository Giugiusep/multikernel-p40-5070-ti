// Experimental Qwen packed-expert worker. No model filesystem or PCI changes.
#include "strata/kernels/iq_kernels.hpp"
#include <cuda_runtime.h>
#include <linux/vm_sockets.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>
#include <chrono>
#include <algorithm>
using namespace strata::kernels;
constexpr uint32_t MAGIC=0x5058454d, VERSION=1, PORT=6005, MAX_BODY=2*1024*1024;
constexpr int H=2560, FF=640, MAX_ROWS=8, MAX_ENTRIES=80;
constexpr size_t CACHE_LIMIT=64*1024*1024;
struct Header {uint32_t magic,version,op,status,request,size;};
static_assert(sizeof(Header)==24);
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> struct Device {
 T* p=nullptr;
 explicit Device(size_t count){ck(cudaMalloc(&p,count*sizeof(T)));}
 ~Device(){cudaFree(p);}
 Device(const Device&)=delete; Device& operator=(const Device&)=delete;
};
struct Expert {
 uint8_t* p=nullptr; NativeExpertLayout layout{};
 ~Expert(){if(p)cudaFree(p);}
};
static void io(int fd,void* data,size_t n,bool send_data){
 auto* p=static_cast<char*>(data); unsigned retries=0;
 while(n){size_t chunk=std::min(n,size_t(32768));
  ssize_t got=send_data?send(fd,p,chunk,MSG_NOSIGNAL):recv(fd,p,chunk,0);
  if(got<0&&errno==EINTR)continue;
  if(got<0&&send_data&&(errno==ENOSPC||errno==EAGAIN)&&++retries<=1000){usleep(100);continue;}
  if(got<=0)throw std::runtime_error("socket IO closed or timed out: "+std::string(strerror(errno)));
  retries=0;p+=got;n-=got;
  if(send_data&&n)usleep(100); // conservative known-good pacing in both directions
 }
}
static uint32_t word(const std::vector<uint8_t>& b,size_t i){uint32_t v; if((i+1)*4>b.size())throw std::runtime_error("short payload");memcpy(&v,b.data()+i*4,4);return v;}
class Worker {
 std::map<std::pair<int,int>,Expert> cache; size_t resident=0;
 Device<float> x{MAX_ROWS*H}, out{MAX_ENTRIES*H};
 Device<uint8_t> q{MAX_ROWS*H/32*36};
 Device<uint8_t> scratch{native_expert_scratch_bytes(MAX_ENTRIES,FF)};
 Device<unsigned long long> ptr{MAX_ENTRIES};
 Device<int32_t> starts{MAX_ENTRIES+1}, count{1}, dst{MAX_ENTRIES}, tok{MAX_ENTRIES};
 cudaStream_t stream=nullptr;
public:
 Worker(){ck(cudaStreamCreate(&stream)); native_expert_set_mode(8,0);}
 ~Worker(){cudaStreamDestroy(stream);}
 std::vector<uint8_t> dispatch(uint32_t op,const std::vector<uint8_t>& b){
  if(op==1){ // Load exactly one checkpoint-native packed expert.
   if(b.size()<28)throw std::runtime_error("short load");
   int layer=word(b,0), id=word(b,1),gu=word(b,2),down=word(b,3);
   if(layer<0||layer>=48||id<0||id>=512||word(b,4)!=H||word(b,5)!=FF||
      (gu!=16&&gu!=22&&gu!=29)||down!=42)throw std::runtime_error("unsupported Qwen shape/format/id");
   auto L=native_expert_layout(gu,down,H,FF);
   if(!native_expert_supported(gu,down,H,FF)||word(b,6)!=L.bytes||b.size()!=28+L.bytes)throw std::runtime_error("expert byte count");
   if(cache.count({layer,id}))throw std::runtime_error("duplicate expert");
   if(resident+L.bytes>CACHE_LIMIT)throw std::runtime_error("64 MiB experimental cache limit");
   auto& e=cache[{layer,id}];e.layout=L;
   ck(cudaMalloc(&e.p,L.bytes));ck(cudaMemcpy(e.p,b.data()+28,L.bytes,cudaMemcpyHostToDevice));resident+=L.bytes;
   return {};
  }
  if(op==2){
   if(b.size()<16)throw std::runtime_error("short run");
   int layer=word(b,0), rows=word(b,1),k=word(b,2);
   if(rows<1||rows>MAX_ROWS||k<1||k>10||word(b,3)!=0)throw std::runtime_error("run dimensions");
   int entries=rows*k;
   if(b.size()!=16+entries*4+rows*H*4)throw std::runtime_error("run byte count");
   std::vector<int32_t> ids(entries);memcpy(ids.data(),b.data()+16,entries*4);
   std::vector<int32_t> experts,hs,hd,ht;std::vector<unsigned long long>hp;
   NativeExpertLayout L{};bool have=false;
   for(int e:ids){auto it=cache.find({layer,e});if(it==cache.end())throw std::runtime_error("expert not resident");
    if(!have){L=it->second.layout;have=true;}else if(L.gu_type!=it->second.layout.gu_type)throw std::runtime_error("mixed layer formats");
    if(std::find(experts.begin(),experts.end(),e)==experts.end())experts.push_back(e);
   }
   for(int e:experts){hs.push_back(hd.size());hp.push_back((unsigned long long)cache.at({layer,e}).p);
    for(int i=0;i<entries;i++)if(ids[i]==e){hd.push_back(i);ht.push_back(i/k);}}
   hs.push_back(entries);int groups=experts.size();
   ck(cudaMemcpyAsync(x.p,b.data()+16+entries*4,rows*H*4,cudaMemcpyHostToDevice,stream));
   ck(cudaMemcpyAsync(ptr.p,hp.data(),hp.size()*8,cudaMemcpyHostToDevice,stream));
   ck(cudaMemcpyAsync(starts.p,hs.data(),hs.size()*4,cudaMemcpyHostToDevice,stream));
   ck(cudaMemcpyAsync(dst.p,hd.data(),hd.size()*4,cudaMemcpyHostToDevice,stream));
   ck(cudaMemcpyAsync(tok.p,ht.data(),ht.size()*4,cudaMemcpyHostToDevice,stream));
   ck(cudaMemcpyAsync(count.p,&groups,4,cudaMemcpyHostToDevice,stream));
   quantize_q8_1_rows(x.p,rows,H,q.p,stream);
   native_expert_grouped(L,ptr.p,starts.p,count.p,dst.p,tok.p,groups,entries,q.p,scratch.p,out.p,stream,groups);
   ck(cudaGetLastError());std::vector<uint8_t> result(entries*H*4);
   ck(cudaMemcpyAsync(result.data(),out.p,result.size(),cudaMemcpyDeviceToHost,stream));
   ck(cudaStreamSynchronize(stream));return result;
  }
  if(op==3){if(!b.empty())throw std::runtime_error("info payload");
   cudaDeviceProp p{};ck(cudaGetDeviceProperties(&p,0));size_t free,total;ck(cudaMemGetInfo(&free,&total));
   std::string info="{\"device\":\""+std::string(p.name)+"\",\"sm\":"+std::to_string(p.major*10+p.minor)+",\"experts\":"+std::to_string(cache.size())+",\"resident_bytes\":"+std::to_string(resident)+",\"free_bytes\":"+std::to_string(free)+",\"total_bytes\":"+std::to_string(total)+"}";
   return {info.begin(),info.end()};
  }
  if(op==4){if(!b.empty())throw std::runtime_error("clear payload");cache.clear();resident=0;return {};}
  throw std::runtime_error("unknown operation");
 }
};
int main(){try{
 cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,0));
 if(prop.major!=6||prop.minor!=1)throw std::runtime_error("this build requires SM61");
 int listener=socket(AF_VSOCK,SOCK_STREAM,0),type=1;
 if(listener<0||setsockopt(listener,AF_VSOCK,9,&type,sizeof(type)))throw std::runtime_error("Multikernel socket selection");
 sockaddr_vm addr{};addr.svm_family=AF_VSOCK;addr.svm_cid=VMADDR_CID_ANY;addr.svm_port=PORT;
 if(bind(listener,(sockaddr*)&addr,sizeof(addr))||listen(listener,1))throw std::runtime_error("listen");
 puts("MK_EXPERT_WORKER_READY port=6005 cache_limit=64MiB");fflush(stdout);
 for(;;){int peer=accept(listener,nullptr,nullptr);if(peer<0){if(errno==EINTR)continue;throw std::runtime_error("accept");}
  timeval timeout{10,0};setsockopt(peer,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));setsockopt(peer,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout));
  try {Worker worker;for(;;){Header h{};io(peer,&h,sizeof(h),false);
   if(h.magic!=MAGIC||h.version!=VERSION||h.status!=0||h.size>MAX_BODY)throw std::runtime_error("invalid frame");
   std::vector<uint8_t>b(h.size);io(peer,b.data(),b.size(),false);
   auto t=std::chrono::steady_clock::now();std::vector<uint8_t> result;uint32_t status=0;
   try{result=worker.dispatch(h.op,b);}catch(const std::exception& e){status=1;std::string s=e.what();result.assign(s.begin(),s.end());}
   Header reply{MAGIC,VERSION,h.op,status,h.request,(uint32_t)result.size()};io(peer,&reply,sizeof(reply),true);io(peer,result.data(),result.size(),true);
   fprintf(stderr,"request=%u op=%u in=%u out=%zu status=%u wall_us=%.3f\n",h.request,h.op,h.size,result.size(),status,std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t).count());fflush(stderr);
  }}catch(const std::exception&e){fprintf(stderr,"connection closed: %s\n",e.what());fflush(stderr);}close(peer);
 }
 }catch(const std::exception&e){fprintf(stderr,"FATAL: %s\n",e.what());return 1;}}
