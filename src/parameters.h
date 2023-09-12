#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_array_stream.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_bidirectional.h"
#include "nnet_utils/nnet_conv1d.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_array_stream.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_merge.h"
#include "nnet_utils/nnet_merge_array_stream.h"
#include "nnet_utils/nnet_merge_stream.h"
#include "nnet_utils/nnet_recurrent.h"
#include "nnet_utils/nnet_recurrent_array_stream.h"
#include "nnet_utils/nnet_sepconv1d_array_stream.h"
#include "nnet_utils/nnet_sepconv1d_stream.h"

// hls-fpga-machine-learning insert weights
//#include "weights/w12.h"
//#include "weights/wr12.h"
#include "weights/b12.h"
#include "weights/br12.h"

// hls-fpga-machine-learning insert layer-config
// DecoderGRU
struct config12_1 : nnet::dense_config {
    static const unsigned n_in = N_INPUT_2_10;
    static const unsigned n_out = N_OUT_12 * 3/2;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = N_INPUT_2_10;
    static const unsigned n_zeros = 56941;
    static const unsigned n_nonzeros = 729491;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef accum_dense12_t accum_t;
    typedef bias12_t bias_t;
    typedef weight12_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config12_2 : nnet::dense_config {
    static const unsigned n_in = N_OUT_12;
    static const unsigned n_out = N_OUT_12 * 3/2;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = N_OUT_12;
    static const unsigned n_zeros = 95838;
    static const unsigned n_nonzeros = 690594;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef accum_dense12_t accum_t;
    typedef bias12_t bias_t;
    typedef weight12_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct hard_sigmoid_config12_recr  : nnet::hard_activ_config {
    static const unsigned n_in = N_OUT_12 * 2;
    static const slope12_t slope;
    static const shift12_t shift;
    static const unsigned io_type = nnet::io_array_stream;
    static const unsigned reuse_factor = 1;
};
const slope12_t hard_sigmoid_config12_recr::slope = 0.5;
const shift12_t hard_sigmoid_config12_recr::shift = 0.5;

struct hard_tanh_config12 : nnet::hard_activ_config{
    static const unsigned n_in = N_OUT_12;
    static const slope12_t slope;
    static const shift12_t shift;
    static const unsigned io_type = nnet::io_array_stream;
    static const unsigned reuse_factor = 1;
};
const slope12_t hard_tanh_config12::slope = 0.5;
const shift12_t hard_tanh_config12::shift = 0.5;

struct config12 : nnet::gru_config {
    typedef accum_dense12_t accum_dense_t;
    typedef accum12_t accum_t;
    typedef weight12_t weight_t;  // Matrix
    typedef bias12_t bias_t;  // Vector
    typedef config12_1 mult_config1;
    typedef config12_2 mult_config2;
    typedef hard_sigmoid_config12_recr ACT_CONFIG_GRU;
    template<class x_T, class y_T, class config_T>
    using activation_recr = nnet::activation::hard_sigmoid<x_T, y_T, config_T>;
    typedef hard_tanh_config12 ACT_CONFIG_T;
    template<class x_T, class y_T, class config_T>
    using activation = nnet::activation::hard_tanh<x_T, y_T, config_T>;
    static const unsigned n_in  = N_INPUT_2_10;//512
    static const unsigned n_out = N_OUT_12;//512
    static const unsigned n_state = N_OUT_12;//512
    static const unsigned n_sequence = N_INPUT_1_10;
    static const unsigned n_sequence_out = N_TIME_STEPS_12;
    static const unsigned io_type = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const bool use_static = true;
    static const bool use_initial = 1;
    typedef state12_t state_t;
    typedef act12_t act_t;
    typedef recr_act12_t recr_act_t;
};

#endif
