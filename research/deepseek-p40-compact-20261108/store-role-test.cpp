#include "ds4_store.hpp"
#include <cassert>
#include <filesystem>
#include <thread>
using namespace strata::ds4;
int main(){
 char name[]="/tmp/ds4-role-test-XXXXXX";int fd=mkstemp(name);assert(fd>=0);std::vector<uint8_t> data(1024*1024);for(size_t i=0;i<data.size();i++)data[i]=(i*37+i/997)%251;assert(write(fd,data.data(),data.size())==(ssize_t)data.size());close(fd);
 auto pfd=os::open_par(name),dfd=os::open_direct(name);assert(pfd!=os::kBadFd);
 ExpertStore::LayerGeo g;g.n_exp=8;uint64_t cursor=0;for(int r=0;r<3;r++){g.file[r]=0;g.off[r]=123+20000*r;g.stride[r]=100000;g.bytes[r]=5003+1004*r;g.tr[r]=cursor;cursor+=((g.bytes[r]+255)&~uint64_t(255))+ExpertStore::kSlack;}g.blob=cursor;
 size_t slot=(g.blob+2*ExpertStore::kSlack+4095)&~size_t(4095);std::atomic<bool> remote{true};
 {
  ExpertStore::Cfg cfg;cfg.budget_bytes=slot*2;cfg.floor_bytes=0;cfg.threads=2;cfg.ram_avail=[](){return ~uint64_t(0);};cfg.map_chunk=[](void*p,size_t){return p;};cfg.defer_down=[&](int l,int e){assert(l==0);return remote.load()&&(e%2)==0;};
  ExpertStore st;assert(st.init(cfg,{g},{dfd},{pfd}));
  auto load=[&](int e){std::vector<char>ok;assert(st.ensure(0,&e,1,ok));assert(ok[0]);return st.find(0,e);};
  auto release=[&](int e){st.release(0,&e,1,std::vector<char>{1});};
  auto check=[&](int s,int e,int r){assert(!memcmp(st.mat(s,r),data.data()+g.off[r]+e*g.stride[r],g.bytes[r]));};
  int s=load(0);assert(st.stats().bytes_read==g.bytes[0]+g.bytes[1]);assert(st.stats().down_deferred_bytes==g.bytes[2]);assert(st.dev_blob(s)==0);check(s,0,0);check(s,0,1);
  // Loss of remote residency: concurrent CPU fallback must perform exactly one down read.
  remote=false;std::vector<std::thread> readers;for(int i=0;i<12;i++)readers.emplace_back([&]{check(s,0,2);});for(auto&t:readers)t.join();assert(st.stats().down_refill_bytes==g.bytes[2]);assert(st.stats().bytes_read==g.bytes[0]+g.bytes[1]+g.bytes[2]);assert(st.dev_blob(s)!=0);check(s,0,0);check(s,0,1);release(0);
  int b=load(1);for(int r=0;r<3;r++)check(b,1,r);assert(st.dev_blob(b));release(1);
  remote=true;int c=load(2);assert(st.dev_blob(c)==0);const auto*blob=st.blob(c);for(int r=0;r<3;r++)assert(!memcmp(blob+g.tr[r],data.data()+g.off[r]+2*g.stride[r],g.bytes[r]));assert(st.stats().down_refill_bytes==2*g.bytes[2]);assert(st.dev_blob(c));release(2);
  // Reusing an evicted slot must replace its role-validity mask.
  int d=load(4);assert(st.dev_blob(d)==0);check(d,4,0);check(d,4,1);check(d,4,2);release(4);
  assert(st.stats().bytes_read==4*(g.bytes[0]+g.bytes[1]+g.bytes[2]));assert(st.stats().down_refill_bytes==3*g.bytes[2]);assert(st.stats().fails==0);printf("deferred roles, exact bytes, concurrent fallback, full-blob repair, GPU visibility and eviction: PASS\n");
 }
 os::close_fd(pfd);if(dfd!=os::kBadFd)os::close_fd(dfd);unlink(name);
}
