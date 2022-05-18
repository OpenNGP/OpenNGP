#include <gtest/gtest.h>
#include <torch/torch.h>

#include "ongp/base/tensor.h"

TEST(TENSOR, ARRAY2TENSOR) {
    std::vector<std::vector<float>> vec{
        {1,2,3,4},
        {1,2,3,4},
        {1,2,3,4},
        {1,2,3,4},
    };

    ongp::Array2dToTensor(vec);
}