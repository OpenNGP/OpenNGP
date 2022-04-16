#ifndef ONGP_PRIMITIVE_COMPONENT_GEOMETRY_H_
#define ONGP_PRIMITIVE_COMPONENT_GEOMETRY_H_

#include <torch/torch.h>

namespace ongp
{
    struct LinearImpl : torch::nn::Module {
    LinearImpl(int64_t in, int64_t out);

    torch::Tensor forward(const torch::Tensor& input);

    torch::Tensor weight, bias;
    };

    TORCH_MODULE(Linear); // Linear is now a wrapper over std::shared_ptr<LinearImpl>.
}

#endif // ONGP_PRIMITIVE_COMPONENT_GEOMETRY_H_
