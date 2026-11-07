// Independent byte oracle for two-thread raw/split-scale/MX8 reads.
#include "ds4_engram_io.hpp"
#include <stdexcept>
#include <array>
using namespace strata::ds4;
int main(){try{
 std::array<std::string,2> paths;std::array<std::vector<uint8_t>,2> data;
 for(int f=0;f<2;f++){char name[]="/tmp/ds4-eg-test-XXXXXX";int fd=mkstemp(name);if(fd<0)throw std::runtime_error("tempfile");paths[f]=name;data[f].resize(1024*1024);uint64_t s=20261107+f;for(auto&v:data[f]){s^=s<<13;s^=s>>7;s^=s<<17;v=(uint8_t)s;}size_t done=0;while(done<data[f].size()){ssize_t n=write(fd,data[f].data()+done,data[f].size()-done);if(n<=0)throw std::runtime_error("write");done+=n;}close(fd);}
 for(int hint:{0,1}){setenv("DS4_ENGRAM_RANDOM",hint?"1":"0",1);EgReader reader;reader.start({paths[0],paths[1]},2);
 for(int pass=0;pass<40;pass++){
  EgBatch batch;std::vector<std::vector<uint8_t>> out(64,std::vector<uint8_t>(264)),scales(64,std::vector<uint8_t>(8));
  for(int i=0;i<64;i++){EgRead r;r.file=i%2;r.off=(uint64_t(pass)*25193+i*7919)%(data[0].size()-264);r.n=264;r.dst=out[i].data();r.scales=scales[i].data();r.mx8=i%3==0;if(!r.mx8&&i%3==1){r.file2=1-r.file;r.off2=(r.off+12345)%(data[0].size()-8);r.n2=8;}batch.reads.push_back(r);}
  reader.submit(batch);reader.wait(batch);
  for(int i=0;i<64;i++){const auto&r=batch.reads[i];if(!batch.ok[i])throw std::runtime_error("read failed");if(r.mx8){for(int b=0;b<8;b++){if(scales[i][b]!=data[r.file][r.off+33*b]||memcmp(out[i].data()+32*b,data[r.file].data()+r.off+33*b+1,32))throw std::runtime_error("MX8 mismatch");}}else if(memcmp(out[i].data(),data[r.file].data()+r.off,264)||(r.n2&&memcmp(scales[i].data(),data[r.file2].data()+r.off2,8)))throw std::runtime_error("raw/split mismatch");}
 }reader.stop();printf("hint=%d:2560 concurrent rows exact\n",hint);}
 for(auto&p:paths)unlink(p.c_str());return 0;
 }catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
