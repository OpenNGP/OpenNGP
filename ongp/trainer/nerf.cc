#include "ongp/trainer/nerf.h"

namespace ongp
{
    NeRFTrainer::NeRFTrainer(Primitive *primitive, torch::optim::Optimizer *optimizer) : Trainer(primitive, optimizer)
    {
    }

    void NeRFTrainer::Train()
    {
        // Prepare dataloader
        auto train_data_loader = torch::data::make_data_loader(
            *ray_dataset_, torch::data::DataLoaderOptions()
                                 .batch_size(256)
                                 .workers(8)
                                 .enforce_ordering(true));

        for (size_t epoch = 0; epoch < epoch_; ++epoch)
        {
            size_t batch_index = 0;
            // Iterate the data loader to yield batches from the dataset.
            for (auto &batch : *train_data_loader)
            {
                // Reset gradients.
                opt_->zero_grad();

                // Execute the model on the input data.
                //torch::Tensor prediction = primitive_->forward(batch);
                // TODO: Renderer needed
                torch::Tensor prediction; // from renderer
                torch::Tensor gt; // from batch data
                // Compute a loss value to judge the prediction of our model.
                torch::Tensor loss = torch::huber_loss(prediction, gt);

                // Compute gradients of the loss w.r.t. the parameters of our model.
                loss.backward();

                // Update the parameters based on the calculated gradients.
                opt_->step();
            }
        }
    }
}
