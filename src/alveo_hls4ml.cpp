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
#include "kernel_params.h"

template<class T, unsigned N> 
void fillWeights(const bigdata_t iWeightsIn[N], T *weights) { 
  for(int i0 = 0; i0 < N; i0++) { 
    #pragma HLS PIPELINE II=1
    weights[i0] = iWeightsIn[i0];
    //std::cout <<(double)weights[i0]<<" ";
  }
  //std::cout <<" \n\n\n";
}


extern "C" {

void alveo_hls4ml(
    const int con,
    const bigdata_t *in,
    const bigdata_t *in2,
    const bigdata_t *in_fw2_v1,
    const bigdata_t *in_fw2_v2,
    const bigdata_t *in_fw2_v3,
    const bigdata_t *in_bw2_v1,
    const bigdata_t *in_bw2_v2,
    const bigdata_t *in_bw2_v3,
    const bigdata_t *in_fwr2_v1,
    const bigdata_t *in_fwr2_v2,
    const bigdata_t *in_fwr2_v3,
    const bigdata_t *in_bwr2_v1,
    const bigdata_t *in_bwr2_v2,
    const bigdata_t *in_bwr2_v3,
    
    const bigdata_t *in_w4,
    const bigdata_t *in_w6,
    
    const bigdata_t *in_w12_v1,
    const bigdata_t *in_w12_v2,
    const bigdata_t *in_w12_v3,
    const bigdata_t *in_wr12_v1,
    const bigdata_t *in_wr12_v2,
    const bigdata_t *in_wr12_v3,
    
    bigdata_t *out       // Output Result
    )
{
    #pragma HLS INTERFACE m_axi port=in  offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=in2  offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=in_fw2_v1  offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi port=in_fw2_v2  offset=slave bundle=gmem3
    #pragma HLS INTERFACE m_axi port=in_fw2_v3  offset=slave bundle=gmem4
    #pragma HLS INTERFACE m_axi port=in_bw2_v1  offset=slave bundle=gmem5
    #pragma HLS INTERFACE m_axi port=in_bw2_v2  offset=slave bundle=gmem6
    #pragma HLS INTERFACE m_axi port=in_bw2_v3  offset=slave bundle=gmem7
    #pragma HLS INTERFACE m_axi port=in_fwr2_v1  offset=slave bundle=gmem8
    #pragma HLS INTERFACE m_axi port=in_fwr2_v2  offset=slave bundle=gmem9
    #pragma HLS INTERFACE m_axi port=in_fwr2_v3  offset=slave bundle=gmem10
    #pragma HLS INTERFACE m_axi port=in_bwr2_v1  offset=slave bundle=gmem11
    #pragma HLS INTERFACE m_axi port=in_bwr2_v2  offset=slave bundle=gmem12
    #pragma HLS INTERFACE m_axi port=in_bwr2_v3  offset=slave bundle=gmem13

    #pragma HLS INTERFACE m_axi port=in_w4  offset=slave bundle=gmem14
    #pragma HLS INTERFACE m_axi port=in_w6  offset=slave bundle=gmem15

    #pragma HLS INTERFACE m_axi port=in_w12_v1  offset=slave bundle=gmem16
    #pragma HLS INTERFACE m_axi port=in_w12_v2  offset=slave bundle=gmem17
    #pragma HLS INTERFACE m_axi port=in_w12_v3  offset=slave bundle=gmem18
    #pragma HLS INTERFACE m_axi port=in_wr12_v1  offset=slave bundle=gmem19
    #pragma HLS INTERFACE m_axi port=in_wr12_v2  offset=slave bundle=gmem20
    #pragma HLS INTERFACE m_axi port=in_wr12_v3  offset=slave bundle=gmem21

    #pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem22


    #pragma HLS INTERFACE s_axilite port=con   bundle=control
    #pragma HLS INTERFACE s_axilite port=in   bundle=control 
    #pragma HLS INTERFACE s_axilite port=in2   bundle=control 

    #pragma HLS INTERFACE s_axilite port=in_fw2_v1   bundle=control
    #pragma HLS INTERFACE s_axilite port=in_fw2_v2   bundle=control
    #pragma HLS INTERFACE s_axilite port=in_fw2_v3   bundle=control
    #pragma HLS INTERFACE s_axilite port=in_bw2_v1   bundle=control
    #pragma HLS INTERFACE s_axilite port=in_bw2_v2   bundle=control
    #pragma HLS INTERFACE s_axilite port=in_bw2_v3   bundle=control
    #pragma HLS INTERFACE s_axilite port=in_fwr2_v1   bundle=control
    #pragma HLS INTERFACE s_axilite port=in_fwr2_v2   bundle=control
    #pragma HLS INTERFACE s_axilite port=in_fwr2_v3   bundle=control
    #pragma HLS INTERFACE s_axilite port=in_bwr2_v1   bundle=control
    #pragma HLS INTERFACE s_axilite port=in_bwr2_v2   bundle=control
    #pragma HLS INTERFACE s_axilite port=in_bwr2_v3   bundle=control

    #pragma HLS INTERFACE s_axilite port=in_w4   bundle=control
    #pragma HLS INTERFACE s_axilite port=in_w6   bundle=control

    #pragma HLS INTERFACE s_axilite port=in_w12_v1   bundle=control
    #pragma HLS INTERFACE s_axilite port=in_w12_v2   bundle=control
    #pragma HLS INTERFACE s_axilite port=in_w12_v3   bundle=control
    #pragma HLS INTERFACE s_axilite port=in_wr12_v1   bundle=control
    #pragma HLS INTERFACE s_axilite port=in_wr12_v2   bundle=control
    #pragma HLS INTERFACE s_axilite port=in_wr12_v3   bundle=control

    #pragma HLS INTERFACE s_axilite port=out  bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    //weight file, which are stored in URAM
    static forward_weight2_t fw2_v1[25088];
    static forward_weight2_t fw2_v2[25088];
    static forward_weight2_t fw2_v3[25088];
    static backward_weight2_t bw2_v1[25088];
    static backward_weight2_t bw2_v2[25088];
    static backward_weight2_t bw2_v3[25088];
    static forward_recurrent_weight2_t fwr2_v1[262144];
    static forward_recurrent_weight2_t fwr2_v2[262144];
    static forward_recurrent_weight2_t fwr2_v3[262144];
    static backward_recurrent_weight2_t bwr2_v1[262144];
    static backward_recurrent_weight2_t bwr2_v2[262144];
    static backward_recurrent_weight2_t bwr2_v3[262144];

    static weight4_t w4[524288];
    static weight6_t w6[524288];

    static weight12_t w12_v1[262144];
    static weight12_t w12_v2[262144];
    static weight12_t w12_v3[262144];
    static recurrent_weight12_t wr12_v1[262144];
    static recurrent_weight12_t wr12_v2[262144];
    static recurrent_weight12_t wr12_v3[262144];


    #pragma HLS bind_storage variable=w12_v1 type=RAM_T2P impl=uram
    #pragma HLS bind_storage variable=w12_v2 type=RAM_T2P impl=uram
    #pragma HLS bind_storage variable=w12_v3 type=RAM_T2P impl=uram
    #pragma HLS bind_storage variable=wr12_v1 type=RAM_T2P impl=uram
    #pragma HLS bind_storage variable=wr12_v2 type=RAM_T2P impl=uram
    #pragma HLS bind_storage variable=wr12_v3 type=RAM_T2P impl=uram

    if(con==0) {
        fillWeights<forward_weight2_t,25088>(in_fw2_v1,fw2_v1);
        fillWeights<forward_weight2_t,25088>(in_fw2_v2,fw2_v2);
        fillWeights<forward_weight2_t,25088>(in_fw2_v3,fw2_v3);
        fillWeights<backward_weight2_t,25088>(in_bw2_v1,bw2_v1);
        fillWeights<backward_weight2_t,25088>(in_bw2_v2,bw2_v2);
        fillWeights<backward_weight2_t,25088>(in_bw2_v3,bw2_v3);
        fillWeights<forward_recurrent_weight2_t,262144>(in_fwr2_v1,fwr2_v1);
        fillWeights<forward_recurrent_weight2_t,262144>(in_fwr2_v2,fwr2_v2);
        fillWeights<forward_recurrent_weight2_t,262144>(in_fwr2_v3,fwr2_v3);
        fillWeights<backward_recurrent_weight2_t,262144>(in_bwr2_v1,bwr2_v1);
        fillWeights<backward_recurrent_weight2_t,262144>(in_bwr2_v2,bwr2_v2);
        fillWeights<backward_recurrent_weight2_t,262144>(in_bwr2_v3,bwr2_v3);

        fillWeights<weight4_t,524288>(in_w4,w4);
        fillWeights<weight4_t,524288>(in_w6,w6);

        fillWeights<weight12_t,262144>(in_w12_v1,w12_v1);
        fillWeights<weight12_t,262144>(in_w12_v2,w12_v2);
        fillWeights<weight12_t,262144>(in_w12_v3,w12_v3);
        fillWeights<recurrent_weight12_t,262144>(in_wr12_v1,wr12_v1);
        fillWeights<recurrent_weight12_t,262144>(in_wr12_v2,wr12_v2);
        fillWeights<recurrent_weight12_t,262144>(in_wr12_v3,wr12_v3);
    }
    else {
        kernel(in,in2,fw2_v1,fw2_v2,fw2_v3,bw2_v1,bw2_v2,bw2_v3,fwr2_v1,fwr2_v2,
        fwr2_v3,bwr2_v1,bwr2_v2,bwr2_v3,w4,w6,w12_v1,w12_v2,w12_v3,wr12_v1,wr12_v2,wr12_v3,out);
    }
}
}
