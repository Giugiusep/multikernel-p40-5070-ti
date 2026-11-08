#include "ds4_gguf_cpu.hpp"
#include "ds4_mk_expert.hpp"
#include <cassert>
struct Entry{int e,n,slot0,tok[8];};
int main(){Entry a{7,2,0,{1,2}},b{7,2,2,{3,4}},c=a;assert(strata_mk::same_entry(a,c));assert(!strata_mk::same_entry(a,b));c.tok[1]=3;assert(!strata_mk::same_entry(a,c));c=a;c.n=1;assert(!strata_mk::same_entry(a,c));puts("duplicate expert IDs preserve distinct slots/token rows: PASS");}
