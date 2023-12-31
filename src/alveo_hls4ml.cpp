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

template<unsigned N> 
void fillWeights(const bigdata_t iWeightsIn[N], weight12_t weights[N]) { 
  for(int i0 = 0; i0 < N; i0++) { 
    weights[i0] = iWeightsIn[i0];
  }
}

template<unsigned N> 
void fillWeights1(const bigdata_t iWeightsIn[N], recurrent_weight12_t weights[N]) { 
  for(int i0 = 0; i0 < N; i0++) { 
    weights[i0] = iWeightsIn[i0];
  }
}

extern "C" {

void alveo_hls4ml(
    const bigdata_t *in, // Read-Only Vector
    const bigdata_t *initial, // Read-Only Vector
	const bigdata_t *in_w12,
	const bigdata_t *in_wr12,
    bigdata_t *out       // Output Result
    )
{
    #pragma HLS INTERFACE m_axi port=in  offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=initial  offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=in_w12  offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi port=in_wr12  offset=slave bundle=gmem3
    #pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem4
    #pragma HLS INTERFACE s_axilite port=in   bundle=control
    #pragma HLS INTERFACE s_axilite port=initial   bundle=control
    #pragma HLS INTERFACE s_axilite port=in_w12   bundle=control
    #pragma HLS INTERFACE s_axilite port=in_wr12   bundle=control
    #pragma HLS INTERFACE s_axilite port=out  bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    //weight file, which are stored in URAM
    static weight12_t w12[786432];
    static recurrent_weight12_t wr12[786432];
    static bool fillWeights_ = false;

    if(!fillWeights_) {
      fillWeights<786432>(in_w12,w12);
      fillWeights1<786432>(in_wr12,wr12);
      fillWeights_ = true;
    }else {
        kernel(in,initial,w12,wr12,out);
    }
}
}
