#pragma one

#include <torch/torch.h>
#include "ongp/trainer/trainer.h"

namespace ongp
{
    class NeRFTrainer: public Trainer
    {
    public:
        NeRFTrainer(Primitive* primitive, torch::optim::Optimizer* optimizer);

        void Train() override;
    };
}

