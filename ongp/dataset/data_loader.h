#ifndef ONGP_DATASET_DATA_LOADER_H_
#define ONGP_DATASET_DATA_LOADER_H_

#include "ongp/base/image.h"
#include "ongp/dataset/frame_data.h"

namespace ongp
{
    class DataLoader
    {
    public:
        explicit DataLoader(const std::string& file_path);

        static Image LoadImage(const std::string& img_file);

    protected:
        FrameDataList fd_list_;
    };
}

#endif // ONGP_DATASET_DATA_LOADER_H_
