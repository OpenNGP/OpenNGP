#pragma one

#include <torch/torch.h>

#include "ongp/primitive/component/module/encoder/frequency.h"
#include "ongp/primitive/component/module/regressor/mlp.h"

namespace ongp
{
    struct SigmaOptions{
    SigmaOptions(int64_t n_emb, int64_t d, int64_t w, int64_t final_w)
    : d_(d), w_(w), final_w_(final_w), n_emb_(n_emb)
    {}
        TORCH_ARG(int64_t, n_emb);
        TORCH_ARG(int64_t, d);
        TORCH_ARG(int64_t, w);
        TORCH_ARG(int64_t, final_w);
    };

    struct SigmaImpl : torch::nn::Module {
    explicit SigmaImpl(const SigmaOptions& options): options(options)
    {
        encoder = Frequency(options.n_emb());
        regressor = MLP(MLPOptions(options.d(), options.w(), 3*(1+2*options.n_emb()), options.final_w()+1));
    }

    // TODO: return more tensors
    torch::Tensor forward(const torch::Tensor& input)
    {
        auto feats = input;
        feats = encoder->forward(feats);
        auto outputs = regressor->forward(feats);
        auto sigmas = regressor->options.activation()(outputs);
        return sigmas;
    }

    Frequency encoder;
    MLP regressor;

    SigmaOptions options;
    };

    TORCH_MODULE(Sigma); 
}
