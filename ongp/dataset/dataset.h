#ifndef ONGP_DATASET_DATASET_H_
#define ONGP_DATASET_DATASET_H_

#include <torch/torch.h>

#include "ongp/base/frame.h"
#include "ongp/dataset/frame_data.h"

namespace ongp
{
    class FrameDataset: public torch::data::Dataset<FrameDataset, Frame>
    {
    public:
        FrameDataset(const std::string& root_path, const std::string& dataset);

        Frame get(size_t index) override;

        torch::optional<size_t> size() const override; 

    protected:
        Image LoadImage(const std::string& img_file);

    protected:
        FrameDataList fd_list_;
        std::string root_path_;
    };
}

#endif // ONGP_DATASET_DATASET_H_
