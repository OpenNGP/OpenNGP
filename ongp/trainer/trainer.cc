#include "ongp/trainer/trainer.h"

namespace ongp
{
      Trainer::Trainer(Primitive* primitive, torch::optim::Optimizer* optimizer)
      :primitive_(primitive), opt_(optimizer), epoch_(100), frame_dataset_(nullptr)
      {}
}

