#include "ongp/dataset/data_loader.h"

namespace ongp
{
    DataLoader::DataLoader(const std::string& file_path)
    {
        // TODO: our format TBD
        std::ifstream ifs(file_path);
        json cur_j;
        ifs >> cur_j;
        fd_list_.set_j(cur_j["frames"]);
        fd_list_.Deserialize();
    }
}
