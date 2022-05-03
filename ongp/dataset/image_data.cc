
#include "ongp/dataset/image_data.h"
#define STB_IMAGE_IMPLEMENTATION
#include "ongp/external/stb_image.h"

namespace ongp
{
    Image LoadImage(const std::string& img_file)
    {
        int x,y,n;
        unsigned char *img_data = stbi_load(img_file.c_str(), &x, &y, &n, 0);

        torch::Tensor img_tensor = torch::from_blob(img_data, {x, y, n}, torch::kByte).clone();
        img_tensor = img_tensor.permute({2, 0, 1}); // convert to CxHxW

        stbi_image_free(img_data);

        return Image(img_tensor);
    }
} 