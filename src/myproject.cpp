#include <iostream>

#include "myproject.h"
#include "parameters.h"

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
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=input_1,input_2,layer16_out
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
        //nnet::load_weights_from_txt<weight4_t, 524288>(w4, "w4.txt");
        nnet::load_weights_from_txt<bias4_t, 512>(b4, "b4.txt");
        //nnet::load_weights_from_txt<weight6_t, 524288>(w6, "w6.txt");
        nnet::load_weights_from_txt<bias6_t, 512>(b6, "b6.txt");
        //nnet::load_weights_from_txt<weight12_t, 786432>(w12, "w12.txt");
        //nnet::load_weights_from_txt<recurrent_weight12_t, 786432>(wr12, "wr12.txt");
        nnet::load_weights_from_txt<bias12_t, 1536>(b12, "b12.txt");
        nnet::load_weights_from_txt<recurrent_bias12_t, 1536>(br12, "br12.txt");
        nnet::load_weights_from_txt<nerual_dense_weight_t, 25088>(w17, "w17.txt");
        nnet::load_weights_from_txt<bias17_t, 49>(b17, "b17.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers
std::cout<<"1"<<std::endl;
    hls::stream<layer2_t> layer2_out[1];
    #pragma HLS STREAM variable=layer2_out depth=1
    //nnet::bidirectional_single<input_t, layer2_t, config2>(input_1, layer2_out, bw2, bwr2, bb2, bbr2, fw2, fwr2, fb2, fbr2); // Encoder_BidirectionalGRU
    nnet::bidirectional_single<input_t, layer2_t, config2>(input_1, layer2_out, bw2_v1,bw2_v2,bw2_v3, bwr2_v1, bwr2_v2, bwr2_v3, bb2, bbr2, fw2_v1, fw2_v2, fw2_v3, fwr2_v1, fwr2_v2, fwr2_v3, fb2, fbr2); // Encoder_BidirectionalGRU

std::cout<<"2"<<std::endl;
    hls::stream<layer3_t> layer3_out[1];
    #pragma HLS STREAM variable=layer3_out depth=1
    nnet::linear_single<layer2_t, layer3_t, linear_config3>(layer2_out, layer3_out); // active_bits0
std::cout<<"33"<<std::endl;
    hls::stream<layer33_t> layer3_cpy1[1];
    hls::stream<layer33_t> layer3_cpy2[1];
    #pragma HLS STREAM variable=layer3_cpy1 depth=1
    #pragma HLS STREAM variable=layer3_cpy2 depth=1
    nnet::clone_stream_single<layer3_t, layer33_t, 1024>(layer3_out, layer3_cpy1, layer3_cpy2); // clone_dense	

std::cout<<"3"<<std::endl;
    hls::stream<layer4_t> layer4_out[1];
    #pragma HLS STREAM variable=layer4_out depth=1
    nnet::dense_single<layer33_t, layer4_t, config4>(layer3_cpy1, layer4_out, w4, b4); // dense_mean
std::cout<<"4"<<std::endl;
    hls::stream<layer6_t> layer6_out[1];
    #pragma HLS STREAM variable=layer6_out depth=1
    nnet::dense_single<layer33_t, layer6_t, config6>(layer3_cpy2, layer6_out, w6, b6); // dense_latent2
std::cout<<"5"<<std::endl;
    hls::stream<layer8_t> layer8_out[1];
    #pragma HLS STREAM variable=layer8_out depth=1
    nnet::linear_single<layer4_t, layer8_t, linear_config8>(layer4_out, layer8_out); // active_bits1
std::cout<<"6"<<std::endl;
    hls::stream<layer9_t> layer9_out[1];
    #pragma HLS STREAM variable=layer9_out depth=1
    nnet::linear_single<layer6_t, layer9_t, linear_config9>(layer6_out, layer9_out); // active_bits
std::cout<<"7"<<std::endl;
    hls::stream<layer11_t> layer11_out[1];
    #pragma HLS STREAM variable=layer11_out depth=1
    nnet::add_single<layer8_t, layer9_t, layer11_t, config11>(layer8_out, layer9_out, layer11_out); // add
std::cout<<"8"<<std::endl;
    hls::stream<layer12_t> layer12_out[1];
    #pragma HLS STREAM variable=layer12_out depth=73
    nnet::gru_stack_single<input10_t, layer11_t, layer12_t, config12>(input_2, layer11_out, layer12_out, w12_v1,w12_v2,w12_v3, wr12_v1,wr12_v2,wr12_v3, b12, br12); // DecoderGRU
std::cout<<"9"<<std::endl;
    hls::stream<layer13_t> layer13_out[1];
    #pragma HLS STREAM variable=layer13_out depth=73
    nnet::linear_single<layer12_t, layer13_t, linear_config13>(layer12_out, layer13_out); // active_bits2
std::cout<<"10"<<std::endl;
    hls::stream<layer17_t> layer17_out[1];
    #pragma HLS STREAM variable=layer17_out depth=73
    nnet::pointwise_conv_1d_cl_single<layer13_t, layer17_t, config17>(layer13_out, layer17_out, w17, b17); // nerual_dense
std::cout<<"11"<<std::endl;
    nnet::linear_single<layer17_t, result_t, linear_config16>(layer17_out, layer16_out); // active_bits4

}
