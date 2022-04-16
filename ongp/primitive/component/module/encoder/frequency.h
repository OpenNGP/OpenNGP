#ifndef ONGP_PRIMITIVE_MODULE_ENCODER_FREQUENCY_H_
#define ONGP_PRIMITIVE_MODULE_ENCODER_FREQUENCY_H_

#include <torch/torch.h>

namespace ongp
{
    struct FrequencyImpl : torch::nn::Module {
        FrequencyImpl(int n_freqs, bool logscale = true)
        : n_freqs(n_freqs)
        {
            if (logscale)
                freq_bands = torch::exp2(torch::linspace(0, n_freqs-1, n_freqs));
            else
                freq_bands = torch::linspace(1, (n_freqs-1)*(n_freqs-1), n_freqs);
        }

        torch::Tensor forward(const torch::Tensor& input)
        {
            std::vector<torch::Tensor> outputs{input};
            auto freqs = freq_bands.accessor<int, 1>();
            for (int64_t i = 0; i < freqs.size(0); ++ i)
            {
                auto sin_x = torch::sin(freqs[i]*input);
                auto cos_x = torch::cos(freqs[i]*input);
                outputs.push_back(sin_x);
                outputs.push_back(cos_x);
            }

            return torch::concat(torch::TensorList(outputs), -1);
        }

        int n_freqs;
        torch::Tensor freq_bands;
    };

    TORCH_MODULE(Frequency); 
}

#endif // ONGP_PRIMITIVE_MODULE_ENCODER_FREQUENCY_H_
