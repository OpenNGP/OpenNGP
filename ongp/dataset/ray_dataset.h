#pragma one

#include <torch/torch.h>

#include "ongp/base/ray.h"
#include "ongp/base/frame.h"
#include "ongp/dataset/frame_data.h"

namespace ongp
{
    struct RayWithPixel
    {
        Ray ray;
        torch::Tensor pixel;
    };

    class RayDataset: public torch::data::Dataset<RayDataset, RayWithPixel>
    {
    public:
        RayDataset(const std::string& root_path, const std::string& dataset);

        RayWithPixel get(size_t index) override;

        torch::optional<size_t> size() const override; 

    protected:
        FrameDataList fd_list_;
        std::string root_path_;

        std::vector<Frame> frames_;
    };
}
