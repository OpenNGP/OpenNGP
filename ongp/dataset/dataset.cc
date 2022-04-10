#include "ongp/dataset/dataset.h"
#include "ongp/external/stb_image.h"

namespace ongp
{
    FrameDataset::FrameDataset(const std::string& file_path)
    {
        // TODO: our format TBD
        std::ifstream ifs(file_path);
        json cur_j;
        ifs >> cur_j;
        fd_list_.set_j(cur_j["frames"]);
        fd_list_.Deserialize();
    }

    Frame FrameDataset::get(size_t index) {

        auto img_path = fd_list_.frames[index].img_path;
        auto mat44 = fd_list_.frames[index].mat44;

        int x,y,n;
        unsigned char *img_data = stbi_load(img_path.c_str(), &x, &y, &n, 0);

        // Convert the image and label to a tensor.
        // Here we need to clone the data, as from_blob does not change the ownership of the underlying memory,
        // which, therefore, still belongs to OpenCV. If we did not clone the data at this point, the memory
        // would be deallocated after leaving the scope of this get method, which results in undefined behavior.
        torch::Tensor img_tensor = torch::from_blob(img_data, {x, y, n}, torch::kByte).clone();
        img_tensor = img_tensor.permute({2, 0, 1}); // convert to CxHxW

        stbi_image_free(img_data);

        return Frame(Camera(Pose(mat44)), Image(img_tensor));
    }

    torch::optional<size_t> FrameDataset::size() const {
        return fd_list_.frames.size();
    }
}
