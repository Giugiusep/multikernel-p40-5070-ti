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
#if MK_ENABLE_CPU_COMPAT
extern "C" int mk_cpu_quantized_x(int,const float*,int,float*);
extern "C" int mk_cpu_quantized_raw(int,const float*,int,void*);
extern "C" int mk_cpu_swiglu_hq(int,const float*,const float*,int,void*);
extern "C" int mk_cpu_pool_hq(int,const float*,const float*,int,void*);
namespace strata::kernels {
void mk_compatible_gu_launch(int,NativeExpertLayout,const unsigned long long*,const int32_t*,const float*,int,float*,float*,void*);
void mk_compatible_down_launch(NativeExpertLayout,const unsigned long long*,const void*,int,float*,void*);
void mk_exact_iq2_gu_launch(int,NativeExpertLayout,const unsigned long long*,const int32_t*,const int32_t*,const void*,int,float*,float*,void*);
void mk_cpu_pool_down_launch(NativeExpertLayout,const unsigned long long*,const void*,int,float*,void*);
}
#endif
constexpr uint32_t MAGIC=0x5058454d, VERSION=1, PORT=6005, MAX_BODY=2*1024*1024;
constexpr int H=2560, FF=640, MAX_ROWS=8, MAX_ENTRIES=80;
static size_t cache_limit(){
 const char* env=getenv("MK_WORKER_CACHE_MIB");if(!env)return 64*1024*1024;
 char* end=nullptr;unsigned long value=strtoul(env,&end,10);
 if(!*env||*end||value<64||value>512)throw std::runtime_error("cache limit must be 64..512 MiB");
 return size_t(value)*1024*1024;
}
struct Header {uint32_t magic,version,op,status,request,size;};
static_assert(sizeof(Header)==24);
struct CudaFault : std::runtime_error {using std::runtime_error::runtime_error;};
static void ck(cudaError_t e){if(e!=cudaSuccess)throw CudaFault(cudaGetErrorString(e));}
struct WorkerProfile {
 bool enabled=getenv("MK_WORKER_PROFILE")!=nullptr;
 cudaEvent_t event[8]{};
 WorkerProfile(){if(enabled)for(auto& e:event)ck(cudaEventCreate(&e));}
 ~WorkerProfile(){if(enabled)for(auto e:event)cudaEventDestroy(e);}
 void mark(int n,cudaStream_t s){if(enabled)ck(cudaEventRecord(event[n],s));}
 double us(int a,int b){float ms=0;if(enabled)ck(cudaEventElapsedTime(&ms,event[a],event[b]));return ms*1000.;}
};
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
#if MK_ENABLE_CPU_COMPAT
 Device<float> compat_gate{MAX_ENTRIES*FF},compat_up{MAX_ENTRIES*FF};
 Device<uint8_t> compat_hq{MAX_ENTRIES*880};
 Device<uint8_t> compat_aq{MAX_ROWS*2920};
#endif
 cudaStream_t stream=nullptr;
 WorkerProfile profile;
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
#if MK_ENABLE_CPU_COMPAT
   if(getenv("MK_WORKER_CPU_EXACT_IQ2")&&gu==29)throw std::runtime_error("exact IQ2 worker does not offload IQ1_M");
#endif
   if(resident+L.bytes>cache_limit())throw std::runtime_error("experimental cache byte limit");
   auto& e=cache[{layer,id}];e.layout=L;
   ck(cudaMalloc(&e.p,L.bytes));ck(cudaMemcpy(e.p,b.data()+28,L.bytes,cudaMemcpyHostToDevice));resident+=L.bytes;
   return {};
  }
  if(op==2||op==5){
   if(b.size()<16)throw std::runtime_error("short run");
   int layer=word(b,0), rows=word(b,1),k=word(b,2);
   if(rows<1||rows>MAX_ROWS||k<1||(op==2?k>10:k>MAX_ENTRIES)||word(b,3)!=0)throw std::runtime_error("run dimensions");
   int entries=op==2?rows*k:k;
   size_t meta_bytes=entries*(op==2?4:8);
   if(b.size()!=16+meta_bytes+rows*H*4)throw std::runtime_error("run byte count");
   std::vector<int32_t> ids(entries),tokens(entries);
   for(int i=0;i<entries;i++){
    ids[i]=word(b,4+i*(op==2?1:2));tokens[i]=op==2?i/k:word(b,5+i*2);
    if(tokens[i]<0||tokens[i]>=rows)throw std::runtime_error("token index");
   }
   std::vector<int32_t> experts,hs,hd,ht;std::vector<unsigned long long>hp;
   NativeExpertLayout L{};bool have=false;
   for(int e:ids){auto it=cache.find({layer,e});if(it==cache.end())throw std::runtime_error("expert not resident");
    if(!have){L=it->second.layout;have=true;}else if(L.gu_type!=it->second.layout.gu_type)throw std::runtime_error("mixed layer formats");
    if(std::find(experts.begin(),experts.end(),e)==experts.end())experts.push_back(e);
   }
   for(int e:experts){hs.push_back(hd.size());hp.push_back((unsigned long long)cache.at({layer,e}).p);
    for(int i=0;i<entries;i++)if(ids[i]==e){hd.push_back(i);ht.push_back(tokens[i]);}}
   hs.push_back(entries);int groups=experts.size();
#if MK_ENABLE_CPU_COMPAT
   if(getenv("MK_WORKER_CPU_COMPAT")){
    auto profile_start=std::chrono::steady_clock::now();
    auto profile_gu_start=profile_start;
    std::vector<float> input(rows*H),decoded(rows*H),hg(entries*FF),hu(entries*FF);
    memcpy(input.data(),b.data()+16+meta_bytes,input.size()*4);
    if(mk_cpu_quantized_x(L.gu_type,input.data(),rows,decoded.data()))throw std::runtime_error("CPU input quantizer");
    std::vector<unsigned long long> entry_ptr(entries);
    for(int i=0;i<entries;i++)entry_ptr[i]=(unsigned long long)cache.at({layer,ids[i]}).p;
    ck(cudaMemcpyAsync(x.p,decoded.data(),decoded.size()*4,cudaMemcpyHostToDevice,stream));
    ck(cudaMemcpyAsync(ptr.p,entry_ptr.data(),entries*8,cudaMemcpyHostToDevice,stream));
    ck(cudaMemcpyAsync(tok.p,tokens.data(),entries*4,cudaMemcpyHostToDevice,stream));
    if(getenv("MK_WORKER_CPU_EXACT_IQ2")){
     std::vector<uint8_t> packed(rows*2920);std::vector<int32_t> nt(entries);
     if(mk_cpu_quantized_raw(L.gu_type,input.data(),rows,packed.data()))throw std::runtime_error("CPU packed input quantizer");
     for(int i=0;i<entries;i++){bool seen[MAX_ROWS]={};for(int j=0;j<entries;j++)if(ids[j]==ids[i])seen[tokens[j]]=true;for(bool v:seen)nt[i]+=v;}
     ck(cudaMemcpyAsync(compat_aq.p,packed.data(),packed.size(),cudaMemcpyHostToDevice,stream));
     ck(cudaMemcpyAsync(starts.p,nt.data(),entries*4,cudaMemcpyHostToDevice,stream));
     profile_gu_start=std::chrono::steady_clock::now();profile.mark(0,stream);
     mk_exact_iq2_gu_launch(L.gu_type,L,ptr.p,tok.p,starts.p,compat_aq.p,entries,compat_gate.p,compat_up.p,stream);
     profile.mark(1,stream);
     ck(cudaStreamSynchronize(stream)); // packed/nt host buffers must survive their async copies
    }else {profile_gu_start=std::chrono::steady_clock::now();profile.mark(0,stream);mk_compatible_gu_launch(L.gu_type,L,ptr.p,tok.p,x.p,entries,compat_gate.p,compat_up.p,stream);profile.mark(1,stream);}
    auto profile_gu_wait_done=std::chrono::steady_clock::now();
    ck(cudaGetLastError());
    profile.mark(2,stream);
    ck(cudaMemcpyAsync(hg.data(),compat_gate.p,hg.size()*4,cudaMemcpyDeviceToHost,stream));
    ck(cudaMemcpyAsync(hu.data(),compat_up.p,hu.size()*4,cudaMemcpyDeviceToHost,stream));
    profile.mark(3,stream);
    ck(cudaStreamSynchronize(stream));
    auto profile_read_done=std::chrono::steady_clock::now();
    bool pool=getenv("MK_WORKER_CPU_POOL")!=nullptr;
    std::vector<uint8_t> hq(entries*(pool?880:680));
    int quant_status=pool?mk_cpu_pool_hq(L.gu_type,hg.data(),hu.data(),entries,hq.data()):mk_cpu_swiglu_hq(L.gu_type,hg.data(),hu.data(),entries,hq.data());
    if(quant_status)throw std::runtime_error("CPU hidden quantizer");
    auto profile_quant_done=std::chrono::steady_clock::now();profile.mark(4,stream);
    ck(cudaMemcpyAsync(compat_hq.p,hq.data(),hq.size(),cudaMemcpyHostToDevice,stream));
    profile.mark(5,stream);
    if(pool)mk_cpu_pool_down_launch(L,ptr.p,compat_hq.p,entries,out.p,stream);
    else mk_compatible_down_launch(L,ptr.p,compat_hq.p,entries,out.p,stream);
    ck(cudaGetLastError());
    profile.mark(6,stream);
    std::vector<uint8_t> result(entries*H*4);
    ck(cudaMemcpyAsync(result.data(),out.p,result.size(),cudaMemcpyDeviceToHost,stream));
    profile.mark(7,stream);
    ck(cudaStreamSynchronize(stream));
    if(profile.enabled){
     auto now=std::chrono::steady_clock::now();
     auto wall=[](auto a,auto b){return std::chrono::duration<double,std::micro>(b-a).count();};
     fprintf(stderr,"PROFILE type=%d M=%d entries=%d prep_wall_us=%.3f gu_gpu_us=%.3f gu_wait_wall_us=%.3f gu_d2h_gpu_us=%.3f gu_read_wall_us=%.3f hidden_cpu_us=%.3f hidden_h2d_gpu_us=%.3f down_gpu_us=%.3f output_d2h_gpu_us=%.3f down_wall_us=%.3f dispatch_wall_us=%.3f\n",L.gu_type,rows,entries,wall(profile_start,profile_gu_start),profile.us(0,1),wall(profile_gu_start,profile_gu_wait_done),profile.us(2,3),wall(profile_gu_wait_done,profile_read_done),wall(profile_read_done,profile_quant_done),profile.us(4,5),profile.us(5,6),profile.us(6,7),wall(profile_quant_done,now),wall(profile_start,now));fflush(stderr);
    }
    return result;
   }
#endif
   ck(cudaMemcpyAsync(x.p,b.data()+16+meta_bytes,rows*H*4,cudaMemcpyHostToDevice,stream));
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
   const char* contract="experimental-other";
#if MK_ENABLE_CPU_COMPAT
   if(getenv("MK_WORKER_CPU_COMPAT")&&getenv("MK_WORKER_CPU_EXACT_IQ2")&&getenv("MK_WORKER_CPU_POOL"))contract="strata-pool-iq2-v1";
#endif
   info.pop_back();info+=",\"math_contract\":\""+std::string(contract)+"\"}";
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
 uint32_t port=getenv("MK_WORKER_PORT")?(uint32_t)strtoul(getenv("MK_WORKER_PORT"),nullptr,10):PORT;
 if(port<6005||port>6008)throw std::runtime_error("experimental worker port must be 6005..6008");
 sockaddr_vm addr{};addr.svm_family=AF_VSOCK;addr.svm_cid=VMADDR_CID_ANY;addr.svm_port=port;
 if(bind(listener,(sockaddr*)&addr,sizeof(addr))||listen(listener,1))throw std::runtime_error("listen");
 printf("MK_EXPERT_WORKER_READY port=%u cache_limit=%zuMiB cpu_compat=%d\n",port,cache_limit()/1024/1024,getenv("MK_WORKER_CPU_COMPAT")?1:0);fflush(stdout);
 for(;;){int peer=accept(listener,nullptr,nullptr);if(peer<0){if(errno==EINTR)continue;throw std::runtime_error("accept");}
  timeval timeout{10,0};setsockopt(peer,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));setsockopt(peer,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout));
  try {Worker worker;for(;;){Header h{};io(peer,&h,sizeof(h),false);
   if(h.magic!=MAGIC||h.version!=VERSION||h.status!=0||h.size>MAX_BODY)throw std::runtime_error("invalid frame");
   std::vector<uint8_t>b(h.size);io(peer,b.data(),b.size(),false);
   auto t=std::chrono::steady_clock::now();std::vector<uint8_t> result;uint32_t status=0;
   try{result=worker.dispatch(h.op,b);}catch(const CudaFault&){throw;}catch(const std::exception& e){status=1;std::string s=e.what();result.assign(s.begin(),s.end());}
   Header reply{MAGIC,VERSION,h.op,status,h.request,(uint32_t)result.size()};io(peer,&reply,sizeof(reply),true);io(peer,result.data(),result.size(),true);
   fprintf(stderr,"request=%u op=%u in=%u out=%zu status=%u wall_us=%.3f\n",h.request,h.op,h.size,result.size(),status,std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t).count());fflush(stderr);
  }}catch(const std::exception&e){fprintf(stderr,"connection closed: %s\n",e.what());fflush(stderr);}close(peer);
 }
 }catch(const std::exception&e){fprintf(stderr,"FATAL: %s\n",e.what());return 1;}}
