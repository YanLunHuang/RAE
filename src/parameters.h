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
//#include "weights/fw2.h"
//#include "weights/bw2.h"
//#include "weights/fwr2.h"
//#include "weights/bwr2.h"
#include "weights/fb2.h"
#include "weights/fbr2.h"
#include "weights/bb2.h"
#include "weights/bbr2.h"
//#include "weights/w4.h"
#include "weights/b4.h"
//#include "weights/w6.h"
#include "weights/b6.h"
//#include "weights/w12.h"
//#include "weights/wr12.h"
#include "weights/b12.h"
#include "weights/br12.h"
#include "weights/w17.h"
#include "weights/b17.h"

// hls-fpga-machine-learning insert layer-config
// Encoder_BidirectionalGRU
struct config2_1 : nnet::dense_config {
    static const unsigned n_in = N_INPUT_2_1;
    static const unsigned n_out = N_OUT_2 /2;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = N_INPUT_2_1;
    static const unsigned n_zeros = 4747;
    static const unsigned n_nonzeros = 70517;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef accum_dense2_t accum_t;
    typedef forward_bias2_t bias_t;
    typedef forward_weight2_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config2_2 : nnet::dense_config {
    static const unsigned n_in = N_OUT_2/2;
    static const unsigned n_out = N_OUT_2 /2;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = N_OUT_2/2;
    static const unsigned n_zeros = 95434;
    static const unsigned n_nonzeros = 690998;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef accum_dense2_t accum_t;
    typedef forward_bias2_t bias_t;
    typedef forward_weight2_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct hard_sigmoid_config2_recr  : nnet::hard_activ_config {
    static const unsigned n_in = N_OUT_2 * 2 /2;
    static const slope2_t slope;
    static const shift2_t shift;
    static const unsigned io_type = nnet::io_array_stream;
    static const unsigned reuse_factor = 1;
};
const slope2_t hard_sigmoid_config2_recr::slope = 0.5;
const shift2_t hard_sigmoid_config2_recr::shift = 0.5;

struct hard_tanh_config2 : nnet::hard_activ_config{
    static const unsigned n_in = N_OUT_2/2;
    static const slope2_t slope;
    static const shift2_t shift;
    static const unsigned io_type = nnet::io_array_stream;
    static const unsigned reuse_factor = 1;
};
const slope2_t hard_tanh_config2::slope = 0.5;
const shift2_t hard_tanh_config2::shift = 0.5;

struct config2_f : nnet::gru_config {
    typedef accum_dense2_t accum_dense_t;
    typedef accum2_t accum_t;
    typedef forward_weight2_t weight_t;  // Matrix
    typedef forward_bias2_t bias_t;  // Vector
    typedef config2_1 mult_config1;
    typedef config2_2 mult_config2;
    typedef hard_sigmoid_config2_recr ACT_CONFIG_GRU;
    template<class x_T, class y_T, class config_T>
    using activation_recr = nnet::activation::hard_sigmoid<x_T, y_T, config_T>;
    typedef hard_tanh_config2 ACT_CONFIG_T;
    template<class x_T, class y_T, class config_T>
    using activation = nnet::activation::hard_tanh<x_T, y_T, config_T>;
    static const unsigned n_in  = N_INPUT_2_1;
    static const unsigned n_out = 512;
    static const unsigned n_state = 512;
    static const unsigned n_sequence = N_INPUT_1_1;
    static const unsigned n_sequence_out = 1;
    static const unsigned io_type = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const bool use_static = true;
    typedef state2_t state_t;
    typedef act2_t act_t;
    typedef recr_act2_t recr_act_t;
    static const unsigned merge_mode = nnet::concat;
};

struct config2_b : nnet::gru_config {
    typedef accum_dense2_t accum_dense_t;
    typedef accum2_t accum_t;
    typedef backward_weight2_t weight_t;  // Matrix
    typedef backward_bias2_t bias_t;  // Vector
    typedef config2_1 mult_config1;
    typedef config2_2 mult_config2;
    typedef hard_sigmoid_config2_recr ACT_CONFIG_GRU;
    template<class x_T, class y_T, class config_T>
    using activation_recr = nnet::activation::hard_sigmoid<x_T, y_T, config_T>;
    typedef hard_tanh_config2 ACT_CONFIG_T;
    template<class x_T, class y_T, class config_T>
    using activation = nnet::activation::hard_tanh<x_T, y_T, config_T>;
    static const unsigned n_in  = N_INPUT_2_1;
    static const unsigned n_out = 512;
    static const unsigned n_state = 512;
    static const unsigned n_sequence = N_INPUT_1_1;
    static const unsigned n_sequence_out = 1;
    static const unsigned io_type = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const bool use_static = true;
    typedef state2_t state_t;
    typedef act2_t act_t;
    typedef recr_act2_t recr_act_t;
    static const unsigned merge_mode = nnet::concat;
};

struct config2 : nnet::bidirectional_config {
    typedef accum_dense2_t accum_dense_t;
    typedef accum2_t accum_t;
    typedef backward_weight2_t weight_t;  // Matrix
    typedef backward_bias2_t bias_t;  // Vector

    typedef config2_f config_rnn_layer_f;
    typedef config2_b config_rnn_layer_b;
    static const unsigned n_in  = N_INPUT_2_1;
    static const unsigned n_out = N_OUT_2;
    static const unsigned n_state = N_OUT_2/2;
    static const unsigned n_sequence = N_INPUT_1_1;
    static const unsigned n_sequence_out = 1;
    static const unsigned io_type = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const unsigned merge_mode = nnet::concat;
};

// active_bits0
struct linear_config3 : nnet::activ_config {
    static const unsigned n_in = 1024;
    static const unsigned n_chan = N_OUT_2;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_array_stream;
    static const unsigned reuse_factor = 1;
    typedef active_bits0_table_t table_t;
};

// dense_mean
struct config4 : nnet::dense_config {
    static const unsigned n_in = 1024;
    static const unsigned n_out = 512;
    static const unsigned io_type = nnet::io_array_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 34363;
    static const unsigned n_nonzeros = 489925;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias4_t bias_t;
    typedef weight4_t weight_t;
    typedef layer4_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// dense_latent2
struct config6 : nnet::dense_config {
    static const unsigned n_in = 1024;
    static const unsigned n_out = 512;
    static const unsigned io_type = nnet::io_array_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 34127;
    static const unsigned n_nonzeros = 490161;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias6_t bias_t;
    typedef weight6_t weight_t;
    typedef layer6_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// active_bits1
struct linear_config8 : nnet::activ_config {
    static const unsigned n_in = 512;
    static const unsigned n_chan = N_LAYER_4;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_array_stream;
    static const unsigned reuse_factor = 1;
    typedef active_bits1_table_t table_t;
};

// active_bits
struct linear_config9 : nnet::activ_config {
    static const unsigned n_in = 512;
    static const unsigned n_chan = N_LAYER_6;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_array_stream;
    static const unsigned reuse_factor = 1;
    typedef active_bits_table_t table_t;
};

// add
struct config11 : nnet::merge_config {
    static const unsigned n_elem = N_LAYER_4;
};

// DecoderGRU
struct config12_1 : nnet::dense_config {
    static const unsigned n_in = N_INPUT_2_10;
    static const unsigned n_out = N_OUT_12;
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
    static const unsigned n_out = N_OUT_12;
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
    static const unsigned n_in  = N_INPUT_2_10;
    static const unsigned n_out = N_OUT_12;
    static const unsigned n_state = N_OUT_12;
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

// active_bits2
struct linear_config13 : nnet::activ_config {
    static const unsigned n_in = 37376;
    static const unsigned n_chan = N_OUT_12;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_array_stream;
    static const unsigned reuse_factor = 1;
    typedef active_bits2_table_t table_t;
};

// nerual_dense
struct config17_mult : nnet::dense_config {
    static const unsigned n_in = 512;
    static const unsigned n_out = 49;
    static const unsigned reuse_factor = 512;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef model_default_t accum_t;
    typedef bias17_t bias_t;
    typedef nerual_dense_weight_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config17 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 73;
    static const unsigned n_chan = 512;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 49;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 73;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 73;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 73;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv1DBuffer<data_T, CONFIG_T>;
    typedef model_default_t accum_t;
    typedef bias17_t bias_t;
    typedef nerual_dense_weight_t weight_t;
    typedef config17_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_unscaled<K, S, W>;
};
const ap_uint<config17::filt_width> config17::pixels[] = {0};

// active_bits4
struct linear_config16 : nnet::activ_config {
    static const unsigned n_in = 3577;
    static const unsigned n_chan = N_FILT_17;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_array_stream;
    static const unsigned reuse_factor = 1;
    typedef active_bits4_table_t table_t;
};


#endif
