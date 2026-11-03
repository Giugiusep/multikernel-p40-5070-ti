#pragma once
#include <stdint.h>
struct MkLegacyAct {
    int8_t q[640];
    float scale[20];
    int32_t sum[20];
    float hx[20];
};
static_assert(sizeof(MkLegacyAct)==880);
