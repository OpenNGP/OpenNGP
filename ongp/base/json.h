#ifndef ONGP_BASE_JSON_H_
#define ONGP_BASE_JSON_H_

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

#endif // ONGP_BASE_JSON_H_
