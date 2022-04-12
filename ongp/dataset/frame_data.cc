#include "ongp/dataset/frame_data.h"
#include "ongp/base/tensor.h"

namespace ongp
{
    void FrameData::Serialize()
    {
        j_["file_path"] = img_path; 
        // TODO
    }

    void FrameData::Deserialize()
    {
        img_path = j_["file_path"];
        mat44 = Array2dToTensor(j_["transform_matrix"].get<Array2d<float>>());
        std::cout << j_["transform_matrix"].get<Array2d<float>>()[0] << std::endl;
        std::cout << mat44 << std::endl;
    }

    void FrameDataList::Serialize()
    {
        j_.clear();
        for (auto& frame : frames)
        {
            frame.Serialize();
            j_.push_back(frame.j());
        }
    }

    void FrameDataList::Deserialize()
    {
        frames.clear();
        for (json::iterator it = j_.begin(); it != j_.end(); ++it) {
            FrameData fd;
            fd.set_j(*it);
            fd.Deserialize();
            frames.push_back(fd);
        }
    }
}
