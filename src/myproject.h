#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void myproject(
    hls::stream<input_t> input_1[1],
    hls::stream<layer2_t> layer2_out[1],
    forward_weight2_t fw2_v1[37632],
    forward_weight2_t fw2_v2[37632],
    backward_weight2_t bw2_v1[37632],
    backward_weight2_t bw2_v2[37632],
    forward_recurrent_weight2_t fwr2_v1[393216],
    forward_recurrent_weight2_t fwr2_v2[393216],
    backward_recurrent_weight2_t bwr2_v1[393216],
    backward_recurrent_weight2_t bwr2_v2[393216]
);

#endif
