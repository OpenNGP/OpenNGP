#pragma once

#include <torch/torch.h>
#include "ongp/base/macros.h"
#include "ongp/external/json.hpp"

namespace ongp
{
    using json = nlohmann::json;
    class Json
    {
    public:
        virtual void Serialize() = 0;
        virtual void Deserialize() = 0;

        SET_MEMBER_FUNC(json, j);
        GET_MEMBER_FUNC(json, j);

        void Dump(const std::string& json_file);
        void Load(const std::string& json_file);

    protected:
        json j_;
    };
}
