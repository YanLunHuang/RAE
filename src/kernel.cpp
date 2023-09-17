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
    forward_weight2_t fw2_v1[37632],
    forward_weight2_t fw2_v2[37632],
    backward_weight2_t bw2_v1[37632],
    backward_weight2_t bw2_v2[37632],
    forward_recurrent_weight2_t fwr2_v1[393216],
    forward_recurrent_weight2_t fwr2_v2[393216],
    backward_recurrent_weight2_t bwr2_v1[393216],
    backward_recurrent_weight2_t bwr2_v2[393216],
    bigdata_t *out       // Output Result
    )
{

    #pragma HLS DATAFLOW

    hls::stream<input_t> in_buf[1];
    hls::stream<layer2_t> out_buf[1];

    //If input or output variable is array
    //#pragma HLS ARRAY_PARTITION   variable=in_buf  complete dim=0
    //#pragma HLS ARRAY_PARTITION   variable=out_buf complete dim=0
    #pragma HLS STREAM   variable=in_buf  depth=3000
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
    //=============================================
    //Start computation
    //=============================================

    std::cout<<"inf start"<<std::endl;
    myproject(in_buf,out_buf,fw2_v1,fw2_v2,bw2_v1,bw2_v2,fwr2_v1,fwr2_v2,bwr2_v1,bwr2_v2);
    std::cout<<"inf end"<<std::endl;

    //=============================================
    //Output
    //=============================================
    for(int i0 = 0; i0 < 1024; i0++) {
        #pragma HLS PIPELINE II=1
        layer2_t tmp2 = out_buf[0].read();
        out[i0] = tmp2;
    }

}

