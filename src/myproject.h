#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void myproject(
    hls::stream<input_t> input_1[1], hls::stream<input10_t> input_2[1],
    hls::stream<result_t> layer16_out[1],
    forward_weight2_t fw2_v1[25088],
    forward_weight2_t fw2_v2[25088],
    forward_weight2_t fw2_v3[25088],
    backward_weight2_t bw2_v1[25088],
    backward_weight2_t bw2_v2[25088],
    backward_weight2_t bw2_v3[25088],
    forward_recurrent_weight2_t fwr2_v1[262144],
    forward_recurrent_weight2_t fwr2_v2[262144],
    forward_recurrent_weight2_t fwr2_v3[262144],
    backward_recurrent_weight2_t bwr2_v1[262144],
    backward_recurrent_weight2_t bwr2_v2[262144],
    backward_recurrent_weight2_t bwr2_v3[262144],

    weight4_t w4[524288],
    weight6_t w6[524288],

    weight12_t w12_v1[262144],
    weight12_t w12_v2[262144],
    weight12_t w12_v3[262144],
    recurrent_weight12_t wr12_v1[262144],
    recurrent_weight12_t wr12_v2[262144],
    recurrent_weight12_t wr12_v3[262144]
);

#endif
