#pragma once

#include <torch/torch.h>

#include "ongp/base/json.h"

namespace ongp
{
    struct FrameData: public Json
    {
        void Serialize() override;
        void Deserialize() override;

        torch::Tensor mat44;
        std::string img_path;
    };

    struct FrameDataList: public Json
    {
        void Serialize() override;
        void Deserialize() override;

        std::vector<FrameData> frames;
    };
}
