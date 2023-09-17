#include <iostream>

#include "myproject.h"
#include "parameters.h"

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
) {

    //hls-fpga-machine-learning insert IO
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        //nnet::load_weights_from_txt<forward_weight2_t, 75264>(fw2, "fw2.txt");
        //nnet::load_weights_from_txt<backward_weight2_t, 75264>(bw2, "bw2.txt");
        //nnet::load_weights_from_txt<forward_recurrent_weight2_t, 786432>(fwr2, "fwr2.txt");
        //nnet::load_weights_from_txt<backward_recurrent_weight2_t, 786432>(bwr2, "bwr2.txt");
        nnet::load_weights_from_txt<forward_bias2_t, 1536>(fb2, "fb2.txt");
        nnet::load_weights_from_txt<forward_recurrent_bias2_t, 1536>(fbr2, "fbr2.txt");
        nnet::load_weights_from_txt<backward_bias2_t, 1536>(bb2, "bb2.txt");
        nnet::load_weights_from_txt<backward_recurrent_bias2_t, 1536>(bbr2, "bbr2.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    nnet::bidirectional_single<input_t, layer2_t, config2>(input_1, layer2_out, bw2_v1,bw2_v2, bwr2_v1,bwr2_v2, bb2, bbr2, fw2_v1,fw2_v2, fwr2_v1,fwr2_v2, fb2, fbr2); // Encoder_BidirectionalGRU
}
