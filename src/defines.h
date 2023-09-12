#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 73
#define N_INPUT_2_1 49
#define N_OUT_2 1024
#define N_OUT_2 1024
#define N_LAYER_4 512
#define N_LAYER_6 512
#define N_LAYER_4 512
#define N_LAYER_6 512
#define N_INPUT_1_10 73
#define N_INPUT_2_10 512
#define N_LAYER_4 512
#define N_TIME_STEPS_12 73
#define N_OUT_12 512
#define N_TIME_STEPS_12 73
#define N_OUT_12 512
#define N_OUTPUTS_17 73
#define N_FILT_17 49
#define N_LAYER_1_14 73
#define N_LAYER_2_14 49

// hls-fpga-machine-learning insert layer-precision

typedef ap_fixed<8,1> active_bits_table_t;
typedef ap_fixed<8,1> input10_t;
typedef ap_fixed<8,1> layer11_t;
typedef ap_fixed<8,1> accum12_t;
typedef ap_fixed<8,1> weight12_t;
typedef ap_fixed<8,1> recurrent_weight12_t;
typedef ap_fixed<8,1> bias12_t;
typedef ap_fixed<8,1> recurrent_bias12_t;
typedef ap_fixed<8,1,AP_RND_CONV,AP_SAT> act12_t;
typedef ap_ufixed<8,1,AP_RND_CONV,AP_SAT> recr_act12_t;
typedef ap_fixed<8,1,AP_RND_CONV,AP_SAT> state12_t;
typedef ap_ufixed<8,1> slope12_t;
typedef ap_ufixed<8,1> shift12_t;
typedef ap_fixed<8,1> layer12_t;
typedef ap_fixed<8,1> accum_dense12_t;

#endif
