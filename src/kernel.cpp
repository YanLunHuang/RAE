/**********
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

/*******************************************************************************
Description:
    HLS pragmas can be used to optimize the design : improve throughput, reduce latency and 
    device resource utilization of the resulting RTL code
    This is a wrapper to be used with an hls4ml project to enable proper handling by SDAccel
*******************************************************************************/
#include <iostream>
#include "kernel.h"
#include "myproject.h"


void kernel(
    const bigdata_t *in,
    const bigdata_t *in2,
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
    recurrent_weight12_t wr12_v3[262144],
    bigdata_t *out       // Output Result
    )
{

    #pragma HLS DATAFLOW

    hls::stream<input_t> in_buf[1];
    hls::stream<input10_t> in2_buf[1];
    hls::stream<result_t> out_buf[1];

    //If input or output variable is array
    //#pragma HLS ARRAY_PARTITION   variable=in_buf  complete dim=0
    //#pragma HLS ARRAY_PARTITION   variable=out_buf complete dim=0
    #pragma HLS STREAM   variable=in_buf  depth=73
    #pragma HLS STREAM   variable=in2_buf  depth=73
    #pragma HLS STREAM   variable=out_buf depth=1024

    //=============================================
    //Input
    //=============================================
    //Get data from DRAM
    for (int i = 0; i < N_INPUT_1_1*N_INPUT_2_1; i++) {
        #pragma HLS PIPELINE II=1
        input_t tmp = in[i];
        in_buf[0].write(tmp);
    }
    for (int i = 0; i < N_INPUT_1_10*N_INPUT_2_10; i++) {
        #pragma HLS PIPELINE II=1
        input10_t tmp2 = in2[i];
        in2_buf[0].write(tmp2);
    }
    //=============================================
    //Start computation
    //=============================================

    std::cout<<"inf start"<<std::endl;
    myproject(in_buf,in2_buf,out_buf,fw2_v1,fw2_v2,fw2_v3,bw2_v1,bw2_v2,bw2_v3,fwr2_v1,fwr2_v2,fwr2_v3,
              bwr2_v1,bwr2_v2,bwr2_v3,w4,w6,w12_v1,w12_v2,w12_v3,wr12_v1,wr12_v2,wr12_v3);
    std::cout<<"inf end"<<std::endl;

    //=============================================
    //Output
    //=============================================
    for(int i0 = 0; i0 < N_LAYER_1_14*N_LAYER_2_14; i0++) {
        #pragma HLS PIPELINE II=1
        result_t tmp3 = out_buf[0].read();
        out[i0] = tmp3;
    }

}

