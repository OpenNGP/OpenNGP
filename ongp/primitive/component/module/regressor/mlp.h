#ifndef ONGP_PRIMITIVE_MODULE_REGRESSOR_MLP_H_
#define ONGP_PRIMITIVE_MODULE_REGRESSOR_MLP_H_

#include <torch/torch.h>

namespace ongp
{
    struct MLPOptions {
    MLPOptions(int64_t d, int64_t w, int64_t input_ch, int64_t output_ch)
    : d_(d), w_(w), input_ch_(input_ch), output_ch_(output_ch)
    {}

        TORCH_ARG(int64_t, d);
        TORCH_ARG(int64_t, w);
        TORCH_ARG(int64_t, input_ch);
        TORCH_ARG(int64_t, output_ch);

        TORCH_ARG(bool, act_on_last_layer) = false;
        TORCH_ARG(bool, bias) = false;
        TORCH_ARG(std::vector<int64_t>, skip_connections) = {};
        TORCH_ARG(torch::nn::Functional, activation) = torch::nn::Functional(torch::relu);
    };

    struct MLPImpl : torch::nn::Module {
        explicit MLPImpl(const MLPOptions& options): options(options)
        {
            torch::nn::Linear first_layer(torch::nn::LinearOptions(options.input_ch(), options.w()).bias(options.bias()));
            linears->push_back(first_layer);
            for (int64_t i = 0; i < options.d()-1; ++ i)
            {
                int64_t input_dim = 0;
                if (std::find(options.skip_connections().begin(), options.skip_connections().end(), i) == options.skip_connections().end())
                    input_dim = options.w();
                else 
                    input_dim = options.w()+options.input_ch();
                torch::nn::Linear hidden_layer(torch::nn::LinearOptions(input_dim, options.w()).bias(options.bias()));
                linears->push_back(hidden_layer);
            }
            
            last_layer = torch::nn::Linear(torch::nn::LinearOptions(options.w(), options.output_ch()).bias(options.bias()));

            register_module("linears", linears);
            register_module("last_layer", last_layer);
        }

        torch::Tensor forward(const torch::Tensor& input)
        {
            torch::Tensor h = input;
            for (int64_t i = 0; i < linears->size(); ++ i)
            {
                h = linears[i]->as<torch::nn::Linear>()->forward(h);
                h = options.activation()(h);
                if (std::find(options.skip_connections().begin(), options.skip_connections().end(), i) != options.skip_connections().end())
                    h = torch::concat({input, h}, -1);
            }

            if (options.act_on_last_layer())
                return options.activation()(last_layer->forward(h));
            else
                return last_layer->forward(h);
        }

        torch::nn::ModuleList linears;
        torch::nn::Linear last_layer;

        MLPOptions options;

    };

    TORCH_MODULE(MLP); 
}

#endif // ONGP_PRIMITIVE_MODULE_REGRESSOR_MLP_H_
