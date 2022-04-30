#include "ongp/dataset/ray_dataset.h"

#define STB_IMAGE_IMPLEMENTATION
#include "ongp/external/stb_image.h"

namespace ongp
{
    RayDataset::RayDataset(const std::string& root_path, const std::string& dataset): root_path_(root_path)
    {
        // TODO: our dataset format TBD
        std::ifstream ifs(root_path+dataset);
        json cur_j;
        ifs >> cur_j;
        fd_list_.set_j(cur_j["frames"]);
        fd_list_.Deserialize();

        // Load images
        for (auto& frame_data : fd_list_.frames)
        {
            auto img_path = root_path_ + frame_data.img_path + ".png"; // TODO
            frames_.push_back({Camera(frame_data.mat44), LoadImage(img_path)});
        }
    }

    Image RayDataset::LoadImage(const std::string& img_file)
    {
        int x,y,n;
        unsigned char *img_data = stbi_load(img_file.c_str(), &x, &y, &n, 0);

        torch::Tensor img_tensor = torch::from_blob(img_data, {x, y, n}, torch::kByte).clone();
        img_tensor = img_tensor.permute({2, 0, 1}); // convert to CxHxW

        stbi_image_free(img_data);

        return Image(img_tensor);

    }

    RayWithPixel RayDataset::get(size_t index) {

        auto img = frames_[index].img();
        auto r_idx = torch::randint(img.Height(), 1);
        auto c_idx = torch::randint(img.Width(), 1);

        // Generate ray from camera
        RayWithPixel rp;
        rp.ray = frames_[index].cam().GenerateRay(r_idx, c_idx);
        rp.pixel = img.data().index({r_idx, c_idx});
        return rp;
    }

    torch::optional<size_t> RayDataset::size() const {
        return fd_list_.frames.size();
    }
}
