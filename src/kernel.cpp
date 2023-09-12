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
    const bigdata_t *in, // Read-Only Vector
    const bigdata_t *initial, // Read-Only Vector
    weight12_t w12_v1[393216],
    weight12_t w12_v2[393216],
    recurrent_weight12_t wr12_v1[393216],
    recurrent_weight12_t wr12_v2[393216],
    bigdata_t *out       // Output Result
    )
{

    #pragma HLS DATAFLOW

    input10_t in_bigbuf[N_INPUT_1_10*N_INPUT_2_10];
    layer11_t initial_bigbuf[512];
    layer12_t out_bigbuf;
    
    hls::stream<input10_t> in_buf[1];
    hls::stream<layer11_t> initial_buf[1];
    hls::stream<layer12_t> out_buf[1];

    //If input or output variable is array
    //#pragma HLS ARRAY_PARTITION   variable=in_buf  complete dim=0
    //#pragma HLS ARRAY_PARTITION   variable=out_buf complete dim=0
    #pragma HLS STREAM   variable=in_buf  depth=73
    #pragma HLS STREAM   variable=initial_buf  depth=512
    #pragma HLS STREAM   variable=out_buf depth=73
    
    //Get data from DRAM
    for (int i = 0; i < N_INPUT_1_10*N_INPUT_2_10; i++) {
        #pragma HLS PIPELINE II=1
        in_bigbuf[i] = in[i];
    }
    
    for (int i = 0; i < 512; i++) {
        #pragma HLS PIPELINE II=1
        initial_bigbuf[i] = initial[i];
    }
    //=============================================
    //Input
    //=============================================
    for(int i0 = 0; i0 < N_INPUT_1_10*N_INPUT_2_10; i0++) { 
        #pragma HLS PIPELINE II=1
        input10_t tmp = in_bigbuf[i0];
        in_buf[0].write(tmp);
    }

    for(int i1 = 0; i1 < 512; i1++) { 
        #pragma HLS PIPELINE II=1
        layer11_t tmp1 = initial_bigbuf[i1];
        initial_buf[0].write(tmp1);
    }

    //=============================================
    //Start computation
    //=============================================

    std::cout<<"inf start"<<std::endl;
    myproject(in_buf,initial_buf,out_buf,w12_v1,w12_v2,wr12_v1,wr12_v2);
    std::cout<<"inf end"<<std::endl;

    //=============================================
    //Output
    //=============================================
    for(int i0 = 0; i0 < 37376; i0++) {
        #pragma HLS PIPELINE II=1
        layer12_t tmp2 = out_buf[0].read();
        out[i0] = tmp2;
    }
}

