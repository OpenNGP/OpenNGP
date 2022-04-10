#include "ongp/base/json.h"

#include <fstream>

namespace ongp
{
    void Json::Dump(const std::string& json_file)
    {
        this->Serialize();
        std::ofstream ofs(json_file);
        ofs << std::setw(4) << j_ << std::endl;
    }

    void Json::Load(const std::string& json_file)
    {
        std::ifstream ifs(json_file);
        ifs >> j_;
        this->Deserialize();
    }
}

