#include "ongp/dataset/frame_dataset.h"
#include "ongp/dataset/image_data.h"

namespace ongp
{
    FrameDataset::FrameDataset(const std::string& root_path, const std::string& dataset): root_path_(root_path)
    {
        // TODO: our dataset format TBD
        std::ifstream ifs(root_path+dataset);
        json cur_j;
        ifs >> cur_j;
        fd_list_.set_j(cur_j["frames"]);
        fd_list_.Deserialize();
    }

    Frame FrameDataset::get(size_t index) {

        auto img_path = fd_list_.frames[index].img_path + ".png"; // TODO
        auto mat44 = fd_list_.frames[index].mat44;

        img_path = root_path_ + img_path;
        std::cout << img_path << std::endl;
        std::cout << mat44 << std::endl;

        // Intrinsics
        torch::Tensor k_mat;

        return Frame(Camera(k_mat, Pose(mat44)), LoadImage(img_path));
    }

    torch::optional<size_t> FrameDataset::size() const {
        std::cout <<fd_list_.frames.size() << std::endl; 
        return fd_list_.frames.size();
    }
}
