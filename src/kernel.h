#ifndef MKERNEL_H_
#define MKERNEL_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "kernel_params.h"
#include "defines.h"

// Prototype of top level function for C-synthesis
void kernel(
    const bigdata_t *in, // Read-Only Vector
    const bigdata_t *initial, // Read-Only Vector
    weight12_t w12[786432],
    recurrent_weight12_t wr12[786432],
    bigdata_t *out       // Output Result
);

#endif
