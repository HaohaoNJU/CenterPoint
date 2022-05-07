/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//headers in local files
#include "scatter_cuda.h"
#include <stdio.h>
// For (N * channel) tensors ===>  (channel, BEV_H, BEV_W) tensors
__global__ void scatter_kernel( int *coors, float *pfe_output, float *scattered_feature,
                                const int FEATURE_NUM, const int BEV_H, const int BEV_W)
{
    int i_pillar = blockIdx.x;
    int i_feature = threadIdx.x;
    int bev_ind = coors[i_pillar];
    // if (i_feature ==60 && i_pillar % 100 ==0)
    //     printf("Block %d / %d, Thread %d / %d,  bev_ind %d \n", i_pillar, gridDim.x, i_feature, blockDim.x,bev_ind);

    if(bev_ind >= 0) {
    // // int x_ind = x_coors[i_pillar];
    // // int y_ind = y_coors[i_pillar];
    // // pfe_output : N * 64, get  current feature value ;
    float feature = pfe_output[i_pillar*FEATURE_NUM + i_feature];
    // scattered_feature[i_feature*BEV_H*BEV_W + y_ind * BEV_W + x_ind] = feature;
    scattered_feature[i_feature * BEV_H * BEV_W + bev_ind] = feature;
    }

}


ScatterCuda::ScatterCuda(const int NUM_THREADS, const int FEATURE_NUM, const int GRID_X_SIZE, const int GRID_Y_SIZE):
NUM_THREADS_(NUM_THREADS),
FEATURE_NUM_(FEATURE_NUM),
GRID_X_SIZE_(GRID_X_SIZE),
GRID_Y_SIZE_(GRID_Y_SIZE)
{
}

// MAX_PILLARS, dev_coors_,  static_cast<float*>(buffers.getHostBuffer("47")), dev_scattered_feature_)
// NUM_THREADS_ need to be consistent with channels of pfe output , default is 64
void ScatterCuda::doScatterCuda(const int pillar_count, int *coors, float *pfe_output, float *scattered_feature)
{
  scatter_kernel<<<pillar_count, NUM_THREADS_>>>(coors, pfe_output, scattered_feature,
                                                FEATURE_NUM_, GRID_X_SIZE_, GRID_Y_SIZE_);
}
