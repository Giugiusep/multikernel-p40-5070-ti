// Same CPU quantizers as the deployed Qwen reference; C ABI avoids format ABI mixing.
#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml-quants.h"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "legacy_act.h"
#include <cstring>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
using namespace strata::kernels::cpu;
extern "C" int mk_cpu_pool_reference(int gu,const unsigned char* blob,const float* x,int rows,float* out){
    NativeFmt f;std::string error;if(!native_fmt(gu,42,2560,640,f,error)||rows<1||rows>8)return 1;
    std::vector<unsigned char> aq(rows*kNativeActBytes);std::vector<ActQ> hq(rows);
    std::vector<float> h(rows*640);std::vector<const void*> acts(rows);std::vector<const ActQ*> hs(rows);std::vector<float*> ff(rows),ys(rows);
    for(int t=0;t<rows;t++){native_quant_act(f,x+t*2560,aq.data()+t*kNativeActBytes);acts[t]=aq.data()+t*kNativeActBytes;ff[t]=h.data()+t*640;ys[t]=out+t*2560;}
    native_gu_rows(f,blob,acts.data(),rows,ff.data(),0,640);
    for(int t=0;t<rows;t++){act_quant_any(h.data()+t*640,640,hq[t]);hs[t]=&hq[t];}
    q2_rows_any(blob+f.down_off,f.d_row,10,hs.data(),rows,ys.data(),0,2560);return 0;
}
extern "C" int mk_cpu_pool_hq(int,const float* gate,const float* up,int entries,void* packed){
    if(!cpu_avx512_ok()||expert_oracle_q8_0_enabled())return 1;
    float h[640];ActQ a;
    for(int e=0;e<entries;e++){
        for(int i=0;i<640;i++){float g=gate[e*640+i];h[i]=(g/(1.f+std::exp(-g)))*up[e*640+i];}
        act_quant_any(h,640,a);auto& p=((MkLegacyAct*)packed)[e];
        memcpy(p.q,a.q,640);memcpy(p.scale,a.scale,80);memcpy(p.sum,a.sum,80);memcpy(p.hx,a.hx,80);
    }
    return 0;
}
extern "C" int mk_cpu_quantized_raw(int gu,const float* x,int rows,void* packed){
    NativeFmt f;std::string error;if(!native_fmt(gu,42,2560,640,f,error)||f.gu_act!=GGML_TYPE_Q8_K||f.act_bytes!=2920)return 1;
    for(int t=0;t<rows;t++)native_quant_act(f,x+t*2560,(unsigned char*)packed+t*2920);return 0;
}
extern "C" int mk_cpu_reference(int gu,const unsigned char* blob,const float* x,int rows,float* out){
    NativeFmt f;std::string error;if(!native_fmt(gu,42,2560,640,f,error)||rows<1||rows>8)return 1;
    std::vector<unsigned char> aq(rows*kNativeActBytes),hq(rows*kNativeHBytes);
    std::vector<float> h(rows*640);std::vector<const void*> acts(rows),hs(rows);std::vector<float*> ff(rows),ys(rows);
    for(int t=0;t<rows;t++){native_quant_act(f,x+t*2560,aq.data()+t*kNativeActBytes);acts[t]=aq.data()+t*kNativeActBytes;ff[t]=h.data()+t*640;ys[t]=out+t*2560;}
    native_gu_rows(f,blob,acts.data(),rows,ff.data(),0,640);
    for(int t=0;t<rows;t++){native_quant_h(f,h.data()+t*640,hq.data()+t*kNativeHBytes);hs[t]=hq.data()+t*kNativeHBytes;}
    if(const char* prefix=getenv("MK_CPU_REF_CAPTURE_PREFIX")){
        std::ofstream hf(std::string(prefix)+".h",std::ios::binary);hf.write((char*)h.data(),h.size()*4);
        std::ofstream qf(std::string(prefix)+".hq",std::ios::binary);qf.write((char*)hq.data(),hq.size());
    }
    native_down_rows(f,blob,hs.data(),rows,ys.data(),0,2560);return 0;
}
extern "C" int mk_cpu_down(int gu,const unsigned char* blob,const void* hq,int rows,float* out){
    NativeFmt f;std::string error;if(!native_fmt(gu,42,2560,640,f,error))return 1;
    std::vector<const void*> hs(rows);std::vector<float*> ys(rows);
    for(int t=0;t<rows;t++){hs[t]=(const unsigned char*)hq+t*680;ys[t]=out+t*2560;}
    native_down_rows(f,blob,hs.data(),rows,ys.data(),0,2560);return 0;
}
extern "C" int mk_cpu_quantized_x(int gu,const float* x,int rows,float* decoded){
    NativeFmt f;std::string error;
    if(!native_fmt(gu,42,2560,640,f,error)||f.gu_act!=GGML_TYPE_Q8_K)return 1;
    alignas(64) unsigned char quant[kNativeActBytes];
    for(int t=0;t<rows;t++){
        native_quant_act(f,x+t*2560,quant);
        dequantize_row_q8_K((const block_q8_K*)quant,decoded+t*2560,2560);
    }
    return 0;
}
extern "C" int mk_cpu_swiglu_hq(int gu,const float* gate,const float* up,int entries,void* hq){
    NativeFmt f;std::string error;
    if(!native_fmt(gu,42,2560,640,f,error)||f.d_act!=GGML_TYPE_Q8_0||f.h_bytes!=680)return 1;
    float h[640];
    for(int e=0;e<entries;e++){
        for(int i=0;i<640;i++){float g=gate[e*640+i];h[i]=(g/(1.f+std::exp(-g)))*up[e*640+i];}
        native_quant_h(f,h,(unsigned char*)hq+e*680);
    }
    return 0;
}
