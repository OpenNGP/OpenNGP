#include "ongp/trainer/nerf.h"

namespace ongp
{
     NeRFTrainer::NeRFTrainer(Primitive* primitive, torch::optim::Optimizer* optimizer):
     Trainer(primitive, optimizer)
     {}

     void NeRFTrainer::Train()
     {
        for (size_t epoch = 1; epoch <= epoch_; ++epoch) {
            size_t batch_index = 0;
            // Iterate the data loader to yield batches from the dataset.
            for (auto& batch : *data_loader) {
                // Reset gradients.
                opt_->zero_grad();
                // Execute the model on the input data.
                torch::Tensor prediction = primitive_->forward(batch.data);
                // Compute a loss value to judge the prediction of our model.
                torch::Tensor loss = torch::nll_loss(prediction, batch.target);
                // Compute gradients of the loss w.r.t. the parameters of our model.
                loss.backward();
                // Update the parameters based on the calculated gradients.
                opt_->step();
               // // Output the loss and checkpoint every 100 batches.
               // if (++batch_index % 100 == 0) {
               //     std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
               //             << " | Loss: " << loss.item<float>() << std::endl;
               //     // Serialize your model periodically as a checkpoint.
               //     torch::save(net, "net.pt");
                }
            }
        }
     }

}

