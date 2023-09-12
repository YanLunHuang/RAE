#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void myproject(
    hls::stream<input10_t> input_2[1], hls::stream<layer11_t> layer11_out[1],
    hls::stream<layer12_t> layer12_out[1],
    weight12_t w12[786432],
    recurrent_weight12_t wr12[786432]
);

#endif
