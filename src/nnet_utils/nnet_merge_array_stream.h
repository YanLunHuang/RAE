#ifndef NNET_MERGE_ARRAY_STREAM_H_
#define NNET_MERGE_ARRAY_STREAM_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_mult.h"
#include <math.h>

namespace nnet {

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void add_single(hls::stream<input1_T> data1[1], hls::stream<input2_T> data2[1], hls::stream<res_T> res[1]) {
    for (int ii = 0; ii < CONFIG_T::n_elem; ii++) {
        #pragma HLS PIPELINE
        res_T tmp1 = data1[0].read();
        res_T tmp2 = data2[0].read();
        res[0].write(tmp1+tmp2);
    }
}


template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void add_array(hls::stream<input1_T> data1[CONFIG_T::n_elem], hls::stream<input2_T> data2[CONFIG_T::n_elem], hls::stream<res_T> res[CONFIG_T::n_elem]) {
    #pragma HLS PIPELINE

    for (int ii = 0; ii < CONFIG_T::n_elem; ii++) {
        #pragma HLS UNROLL
        res_T tmp1 = data1[ii].read();
        res_T tmp2 = data2[ii].read();
        res[ii].write(tmp1+tmp2);
    }
}

} // namespace nnet

#endif
