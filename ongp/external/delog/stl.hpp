#ifndef DELOG_STL_HPP
#define DELOG_STL_HPP

#include "delog/delog.hpp"

namespace delog
{
namespace stl
{
namespace basics
{
namespace formats
{
template <typename Type1, typename Type2>
string_t format_pair(const char_t* log_prefix, const char_t* log_suffix, const char_t* name, const std::pair<Type1, Type2>& type, const Parameters& type2_args)
{
    auto format_simple = [&]()
    {
        std::stringstream ss;                               
        ss << LEFT_PARENTHESIS_STR << delog::message(NULL_STR, NULL_STR, "pair.first", type.first, {}) << COMMA_STR; 
        ss << delog::message(NULL_STR, NULL_STR, "pair.second", type.second, {}) << RIGHT_PARENTHESIS_STR; 
        return ss.str();
    };

    auto format_complex = [&]()
    {
        string_t type_str = GET_VARIABLE_TYPE(type);            
        std::stringstream ss;                               
        ss << log_prefix;
        ss << MAGENTA(type_str) + SPACE_STR + GREEN(name) + SPACE_STR+EQUAL_STR+SPACE_STR;
        ss << LEFT_PARENTHESIS_STR << delog::message(NULL_STR, NULL_STR, "pair.first", type.first, {}) << COMMA_STR; 
        ss << delog::message(NULL_STR, NULL_STR, "pair.second", type.second, {}) << RIGHT_PARENTHESIS_STR; 
        ss << log_suffix;
        return ss.str();
    };

    auto format_verbose = [&]()
    {
        string_t type_str = GET_VARIABLE_TYPE(type);            
        std::stringstream ss;                               
        ss << log_prefix << string_t("Name: ") << GREEN(name) << log_suffix;        
        ss << log_prefix << string_t("Type: ") << MAGENTA(type_str) << log_suffix;         
        ss << log_prefix << string_t("First:") << log_suffix;            
        ss << log_prefix << LEFT_BRACE_STR << log_suffix; 
        ss << delog::message(log_prefix, log_suffix, "pair.first", type.first, {}); 
        ss << log_prefix << RIGHT_BRACE_STR << log_suffix; 
        ss << log_prefix << string_t("Second:") << log_suffix;            
        ss << log_prefix << LEFT_BRACE_STR << log_suffix; 
        ss << delog::message(log_prefix, log_suffix, "pair.second", type.second, type2_args); 
        ss << log_prefix << RIGHT_BRACE_STR << log_suffix; 
        return ss.str();
    };
    
    if (default_basic_types.find(typeid(Type1).name()) != default_basic_types.end() && 
        default_basic_types.find(typeid(Type2).name()) != default_basic_types.end())
        {
            if (!string_t(log_prefix).empty() && !string_t(log_suffix).empty())
                return format_complex();
            else
                return format_simple();
        }
    else
        return format_verbose();
}

} // formats


class Primitive
{
public:
    Primitive(const string_t& log_prefix, const string_t& log_suffix): log_prefix_(log_prefix), log_suffix_(log_suffix){}
    // pair
    template <typename Type1, typename Type2>
    string_t generate(const char_t* name, const std::pair<Type1, Type2>& value, const Parameters& type2_args={})
    {
        return build(log_prefix_.c_str(), log_suffix_.c_str(), name, value, type2_args);
    }
private:
    template <typename Type1, typename Type2>
    string_t build(const char_t* log_prefix, const char_t* log_suffix, const char_t* name, const std::pair<Type1, Type2>& type, const Parameters& type2_args)
    {
        return formats::format_pair(log_prefix, log_suffix, name, type, type2_args);
    }
private:
    string_t log_prefix_;
    string_t log_suffix_;
};

} // basics 
} // stl


#define REGISTER_STL_BASICS_TWO_PARAMETER(ContainerType)                                                                    \
template <typename T1, typename T2, typename... Args>                                                                       \
string_t message(const string_t& prefix, const string_t& suffix, const char_t* name, const ContainerType<T1,T2>& type, const std::initializer_list<Args>&... args)          \
{                                                                                                                           \
    return delog::stl::basics::Primitive(prefix, suffix).generate(name, type, args...);                                                   \
}                                                                                                                           \
template <typename T1, typename T2>                                                                                         \
string_t message(const string_t& prefix, const string_t& suffix, const char_t* name, const ContainerType<T1,T2>& type, const Parameters& type2_args)                        \
{                                                                                                                           \
    return delog::stl::basics::Primitive(prefix, suffix).generate(name, type, type2_args);                                                \
}                                                                                  

REGISTER_STL_BASICS_TWO_PARAMETER(std::pair)

namespace stl
{
namespace container
{
namespace formats
{
template <typename Type>
string_t format_range(const char_t* log_prefix, const char_t* log_suffix,const char_t* name, const Type& type, const ParameterList& container_args, const Parameters& type_args)
{
    auto format_simple = [&]()
    {
        string_t type_str = GET_VARIABLE_TYPE(type);            
        std::stringstream ss;                               
        ss << log_prefix;
        ss << MAGENTA(type_str) + SPACE_STR + GREEN(name) + SPACE_STR+EQUAL_STR+SPACE_STR;
        ss << LEFT_BRACKET_STR;
        size_t start = container_args[0];                   
        size_t end = container_args[1];                     
        for (size_t i = start; i <= end; ++ i)              
        {                                                   
            if (i < 0 || i >= type.size()) continue;
            ss << delog::message(NULL_STR, NULL_STR, (string_t("var")+LEFT_BRACKET_STR+std::to_string(i)+RIGHT_BRACKET_STR).c_str(), type[i], type_args); 
            if (i != end) ss << SPACE_STR;
        }                                                   
        ss << RIGHT_BRACKET_STR << log_suffix;
        return ss.str();                                    
    };

    auto format_verbose = [&]()
    {
        string_t type_str = GET_VARIABLE_TYPE(type);            
        std::stringstream ss;                               
        ss << log_prefix << string_t("Name: ") << GREEN(name) << log_suffix;        
        ss << log_prefix << string_t("Type: ") << MAGENTA(type_str) << log_suffix;         
        ss << log_prefix << string_t("Length: ") << type.size() << log_suffix;              
        size_t start = container_args[0];                   
        size_t end = container_args[1];                     
        for (size_t i = start; i <= end; ++ i)              
        {                                                   
            if (i < 0 || i >= type.size()) continue;
            ss << log_prefix << "--------[" << i << "]--------" << log_suffix;                  
            ss << log_prefix << LEFT_BRACE_STR << log_suffix;
            ss << delog::message(log_prefix, log_suffix, (string_t("var")+LEFT_BRACKET_STR+std::to_string(i)+RIGHT_BRACKET_STR).c_str(), type[i], type_args); 
            ss << log_prefix << RIGHT_BRACE_STR << log_suffix;
        }                                                   
        return ss.str();                                    
    };

    if (default_basic_types.find(typeid(type[0]).name()) != default_basic_types.end())
        return format_simple();
    else
        return format_verbose();
}

template <typename Type1, typename Type2>
string_t format_iterator(const char_t* log_prefix, const char_t* log_suffix,const char_t* name, const std::map<Type1, Type2>& type, const ParameterList& container_args, const Parameters& type_args)
{
    auto format_simple = [&]()
    {
        string_t type_str = GET_VARIABLE_TYPE(type);            
        std::stringstream ss;                               
        ss << log_prefix;
        ss << MAGENTA(type_str) + SPACE_STR + GREEN(name) + SPACE_STR+EQUAL_STR+SPACE_STR;
        ss << LEFT_BRACKET_STR;
        size_t length = container_args[0];                   
        auto itr = type.begin();
        size_t count = 0;
        for (auto itr = type.begin(); itr != type.end(); ++ itr)
        {
            ss << delog::message(NULL_STR, NULL_STR, (string_t("var")+LEFT_BRACKET_STR+std::to_string(count)+RIGHT_BRACKET_STR).c_str(), *itr, type_args); 
            count ++;
            if (count == length) 
                break;
            else ss << SPACE_STR;
        }
        ss << RIGHT_BRACKET_STR << log_suffix;
        return ss.str();                                    
    };

    auto format_verbose = [&]()
    {
        string_t type_str = GET_VARIABLE_TYPE(type);            
        std::stringstream ss;                               
        ss << log_prefix << string_t("Name: ") << GREEN(name) << log_suffix;        
        ss << log_prefix << string_t("Type: ") << MAGENTA(type_str) << log_suffix;         
        ss << log_prefix << string_t("Length: ") << type.size() << log_suffix;              
        size_t length = container_args[0];                   

        auto itr = type.begin();
        size_t count = 0;
        for (auto itr = type.begin(); itr != type.end(); ++ itr)
        {
            if (count == length) break;
            ss << log_prefix << "--------[" << count << "]--------" << log_suffix;                 
            ss << log_prefix << LEFT_BRACE_STR << log_suffix;
            ss << delog::message(log_prefix, log_suffix, (string_t("var")+LEFT_BRACKET_STR+std::to_string(count)+RIGHT_BRACKET_STR).c_str(), *itr, type_args);
            ss << log_prefix << RIGHT_BRACE_STR << log_suffix;
            count ++;
        }
        return ss.str();                                    
    };

    if (default_basic_types.find(typeid(Type1).name()) != default_basic_types.end() && 
        default_basic_types.find(typeid(Type2).name()) != default_basic_types.end())
        return format_simple();
    else
        return format_verbose();
}

template <typename Type1, typename Type2>
string_t format_iterator(const char_t* log_prefix, const char_t* log_suffix,const char_t* name, const std::unordered_map<Type1, Type2>& type, const ParameterList& container_args, const Parameters& type_args)
{
    auto format_simple = [&]()
    {
        string_t type_str = GET_VARIABLE_TYPE(type);            
        std::stringstream ss;                               
        ss << log_prefix;
        ss << MAGENTA(type_str) + SPACE_STR + GREEN(name) + SPACE_STR+EQUAL_STR+SPACE_STR;
        ss << LEFT_BRACKET_STR;
        size_t length = container_args[0];                   
        auto itr = type.begin();
        size_t count = 0;
        for (auto itr = type.begin(); itr != type.end(); ++ itr)
        {
            ss << delog::message(NULL_STR, NULL_STR, (string_t("var")+LEFT_BRACKET_STR+std::to_string(count)+RIGHT_BRACKET_STR).c_str(), *itr, type_args); 
            count ++;
            if (count == length) 
                break;
            else ss << SPACE_STR;
        }
        ss << RIGHT_BRACKET_STR << log_suffix;
        return ss.str();                                    
    };

    auto format_verbose = [&]()
    {
        string_t type_str = GET_VARIABLE_TYPE(type);            
        std::stringstream ss;                               
        ss << log_prefix << string_t("Name: ") << GREEN(name) << log_suffix;        
        ss << log_prefix << string_t("Type: ") << MAGENTA(type_str) << log_suffix;         
        ss << log_prefix << string_t("Length: ") << type.size() << log_suffix;              
        size_t length = container_args[0];                   

        auto itr = type.begin();
        size_t count = 0;
        for (auto itr = type.begin(); itr != type.end(); ++ itr)
        {
            if (count == length) break;
            ss << log_prefix << "--------[" << count << "]--------" << log_suffix;                 
            ss << log_prefix << LEFT_BRACE_STR << log_suffix;
            ss << delog::message(log_prefix, log_suffix, (string_t("var")+LEFT_BRACKET_STR+std::to_string(count)+RIGHT_BRACKET_STR).c_str(), *itr, type_args);
            ss << log_prefix << RIGHT_BRACE_STR << log_suffix;
            count ++;
        }
        return ss.str();                                    
    };

    if (default_basic_types.find(typeid(Type1).name()) != default_basic_types.end() && 
        default_basic_types.find(typeid(Type2).name()) != default_basic_types.end())
        return format_simple();
    else
        return format_verbose();
}

template <typename Type>
string_t format_iterator(const char_t* log_prefix, const char_t* log_suffix,const char_t* name, const Type& type, const ParameterList& container_args, const Parameters& type_args)
{
    auto format_simple = [&]()
    {
        string_t type_str = GET_VARIABLE_TYPE(type);            
        std::stringstream ss;                               
        ss << log_prefix;
        ss << MAGENTA(type_str) + SPACE_STR + GREEN(name) + SPACE_STR+EQUAL_STR+SPACE_STR;
        ss << LEFT_BRACKET_STR;
        size_t length = container_args[0];                   
        auto itr = type.begin();
        size_t count = 0;
        for (auto itr = type.begin(); itr != type.end(); ++ itr)
        {
            ss << delog::message(NULL_STR, NULL_STR, (string_t("var")+LEFT_BRACKET_STR+std::to_string(count)+RIGHT_BRACKET_STR).c_str(), *itr, type_args); 
            count ++;
            if (count == length) 
                break;
            else ss << SPACE_STR;
        }
        ss << RIGHT_BRACKET_STR << log_suffix;
        return ss.str();                                    
    };

    auto format_verbose = [&]()
    {
        string_t type_str = GET_VARIABLE_TYPE(type);            
        std::stringstream ss;                               
        ss << log_prefix << string_t("Name: ") << GREEN(name) << log_suffix;        
        ss << log_prefix << string_t("Type: ") << MAGENTA(type_str) << log_suffix;         
        ss << log_prefix << string_t("Length: ") << type.size() << log_suffix;              
        size_t length = container_args[0];                   

        auto itr = type.begin();
        size_t count = 0;
        for (auto itr = type.begin(); itr != type.end(); ++ itr)
        {
            if (count == length) break;
            ss << log_prefix << "--------[" << count << "]--------" << log_suffix;                 
            ss << log_prefix << LEFT_BRACE_STR << log_suffix;
            ss << delog::message(log_prefix, log_suffix, (string_t("var")+LEFT_BRACKET_STR+std::to_string(count)+RIGHT_BRACKET_STR).c_str(), *itr, type_args);
            ss << log_prefix << RIGHT_BRACE_STR << log_suffix;
            count ++;
        }
        return ss.str();                                    
    };

    if (default_basic_types.find(typeid(*(type.begin())).name()) != default_basic_types.end())
        return format_simple();
    else
        return format_verbose();
}


template <typename Type>
string_t format_stack(const char_t* log_prefix, const char_t* log_suffix,const char_t* name, const Type& type, const ParameterList& container_args, const Parameters& type_args)
{
    auto format_simple = [&]()
    {
        string_t type_str = GET_VARIABLE_TYPE(type);            
        std::stringstream ss;                               
        ss << log_prefix;
        ss << MAGENTA(type_str) + SPACE_STR + GREEN(name) + SPACE_STR+EQUAL_STR+SPACE_STR;
        ss << LEFT_BRACKET_STR;
        size_t length = container_args[0];                   

        Type copied = type;
        size_t count = 0;
        while (!copied.empty())
        {
            ss << delog::message(NULL_STR, NULL_STR, (string_t("var")+LEFT_BRACKET_STR+std::to_string(count)+RIGHT_BRACKET_STR).c_str(), copied.top(), type_args); 
            copied.pop();
            count ++;
            if (count == length) break;
            else ss << SPACE_STR;
        }
        ss << RIGHT_BRACKET_STR << log_suffix;
        return ss.str();                                    
    };

    auto format_verbose = [&]()
    {
        string_t type_str = GET_VARIABLE_TYPE(type);            
        std::stringstream ss;                               
        ss << log_prefix << string_t("Name: ") << GREEN(name) << log_suffix;        
        ss << log_prefix << string_t("Type: ") << MAGENTA(type_str) << log_suffix;         
        ss << log_prefix << string_t("Length: ") << type.size() << log_suffix;              
        size_t length = container_args[0];                   

        Type copied = type;
        size_t count = 0;
        while (!copied.empty())
        {
            if (count == length) break;
            ss << log_prefix << "--------[" << count << "]--------" << log_suffix;                 
            ss << log_prefix << LEFT_BRACE_STR << log_suffix;
            ss << delog::message(log_prefix, log_suffix, (string_t("var")+LEFT_BRACKET_STR+std::to_string(count)+RIGHT_BRACKET_STR).c_str(), copied.top(), type_args); 
            ss << log_prefix << RIGHT_BRACE_STR << log_suffix;
            copied.pop();
            count ++;
        }
        return ss.str();                                    
    };

    if (default_basic_types.find(typeid(type.top()).name()) != default_basic_types.end())
        return format_simple();
    else
        return format_verbose();
}

template <typename Type>
string_t format_queue(const char_t* log_prefix, const char_t* log_suffix,const char_t* name, const Type& type, const ParameterList& container_args, const Parameters& type_args)
{
    auto format_simple = [&]()
    {
        string_t type_str = GET_VARIABLE_TYPE(type);            
        std::stringstream ss;                               
        ss << log_prefix;
        ss << MAGENTA(type_str) + SPACE_STR + GREEN(name) + SPACE_STR+EQUAL_STR+SPACE_STR;
        ss << LEFT_BRACKET_STR;
        size_t length = container_args[0];                   

        Type copied = type;
        size_t count = 0;
        while (!copied.empty())
        {
            ss << delog::message(NULL_STR, NULL_STR, (string_t("var")+LEFT_BRACKET_STR+std::to_string(count)+RIGHT_BRACKET_STR).c_str(), copied.front(), type_args); 
            copied.pop();
            count ++;
            if (count == length) break;
            else ss << SPACE_STR;
        }
        ss << RIGHT_BRACKET_STR << log_suffix;
        return ss.str();                                    
    };

    auto format_verbose = [&]()
    {
        string_t type_str = GET_VARIABLE_TYPE(type);            
        std::stringstream ss;                               
        ss << log_prefix << string_t("Name: ") << GREEN(name) << log_suffix;        
        ss << log_prefix << string_t("Type: ") << MAGENTA(type_str) << log_suffix;         
        ss << log_prefix << string_t("Length: ") << type.size() << log_suffix;              
        size_t length = container_args[0];                   

        Type copied = type;
        size_t count = 0;
        while (!copied.empty())
        {
            if (count == length) break;
            ss << log_prefix << "--------[" << count << "]--------" << log_suffix;                 
            ss << log_prefix << LEFT_BRACE_STR << log_suffix;
            ss << delog::message(log_prefix, log_suffix, (string_t("var")+LEFT_BRACKET_STR+std::to_string(count)+RIGHT_BRACKET_STR).c_str(), copied.front(), type_args); 
            ss << log_prefix << RIGHT_BRACE_STR << log_suffix;
            copied.pop();
            count ++;
        }
        return ss.str();                                    
    };

    if (default_basic_types.find(typeid(type.front()).name()) != default_basic_types.end())
        return format_simple();
    else
        return format_verbose();
}


template <typename Type>
ParameterList parameters_to_range(const Type& type, const Parameters& container_args)
{
    ParameterList cargs = ParameterList(container_args);
    ParameterList cargs_default({0, (int)type.size()-1});

    for (size_t i = 0; i < cargs.size(); ++ i) 
    {
        cargs_default.set(i, cargs[i]);
    }

    return cargs_default;
}

template <typename Type>
ParameterList parameters_to_length(const Type& type, const Parameters& container_args)
{
    ParameterList cargs = ParameterList(container_args);
    ParameterList cargs_default({(int)type.size()});

    for (size_t i = 0; i < cargs.size(); ++ i) 
    {
        cargs_default.set(i, cargs[i]);
    }

    return cargs_default;
}

} // formats


class Primitive
{
public:
    Primitive(const string_t& log_prefix, const string_t& log_suffix): log_prefix_(log_prefix), log_suffix_(log_suffix){}

    // vector, list, deque, stack, queue
    template <template<typename, typename> class Container, typename Type1, typename Type2>
    string_t generate(const char_t* name, const Container<Type1, Type2>& value, const Parameters& container_args={}, const Parameters& type_args={})
    {
        return build(log_prefix_.c_str(), log_suffix_.c_str(), name, value, container_args, type_args);
    }

    // set
    template <template<typename, typename, typename> class Container, typename Type1, typename Type2, typename Type3>
    string_t generate(const char_t* name, const Container<Type1, Type2, Type3>& value, const Parameters& container_args={}, const Parameters& type_args={})
    {
        return build(log_prefix_.c_str(), log_suffix_.c_str(), name, value, container_args, type_args);
    }

    // unordered_set, map
    template <template<typename, typename, typename, typename> class Container, typename Type1, typename Type2, typename Type3, typename Type4>
    string_t generate(const char_t* name, const Container<Type1, Type2, Type3, Type4>& value, const Parameters& container_args={}, const Parameters& type_args={})
    {
        return build(log_prefix_.c_str(), log_suffix_.c_str(), name, value, container_args, type_args);
    }

    // unordered_map
    template <template<typename, typename, typename, typename, typename> class Container, typename Type1, typename Type2, typename Type3, typename Type4, typename Type5>
    string_t generate(const char_t* name, const Container<Type1, Type2, Type3, Type4, Type5>& value, const Parameters& container_args={}, const Parameters& type_args={})
    {
        return build(log_prefix_.c_str(), log_suffix_.c_str(), name, value, container_args, type_args);
    }

    // array 
    template <typename Type, size_t N>
    string_t generate(const char_t* name, const std::array<Type, N>& value, const Parameters& container_args={}, const Parameters& type_args={})
    {
        return build(log_prefix_.c_str(), log_suffix_.c_str(), name, value, container_args, type_args);
    }

private:
    template <typename Type>
    string_t build(const char_t* log_prefix, const char_t* log_suffix,const char_t* name, const std::vector<Type>& type, const Parameters& container_args, const Parameters& type_args)
    {
        auto cargs = formats::parameters_to_range(type, container_args);
        return formats::format_range(log_prefix, log_suffix, name, type, cargs, type_args);
    }

    template <typename Type>
    string_t build(const char_t* log_prefix, const char_t* log_suffix,const char_t* name, const std::deque<Type>& type, const Parameters& container_args, const Parameters& type_args)
    {
        auto cargs = formats::parameters_to_range(type, container_args);
        return formats::format_range(log_prefix, log_suffix,name, type, cargs, type_args);
    }

    template <typename Type, size_t N>
    string_t build(const char_t* log_prefix, const char_t* log_suffix,const char_t* name, const std::array<Type, N>& type, const Parameters& container_args, const Parameters& type_args)
    {
        auto cargs = formats::parameters_to_range(type, container_args);
        return formats::format_range(log_prefix, log_suffix,name, type, cargs, type_args);
    }

    template <typename Type>
    string_t build(const char_t* log_prefix, const char_t* log_suffix,const char_t* name, const std::list<Type>& type, const Parameters& container_args, const Parameters& type_args)
    {
        auto cargs = formats::parameters_to_length(type, container_args);
        return formats::format_iterator(log_prefix, log_suffix,name, type, cargs, type_args);
    }

    template <typename Type>
    string_t build(const char_t* log_prefix, const char_t* log_suffix,const char_t* name, const std::set<Type>& type, const Parameters& container_args, const Parameters& type_args)
    {
        auto cargs = formats::parameters_to_length(type, container_args);
        return formats::format_iterator(log_prefix, log_suffix,name, type, cargs, type_args);
    }

    template <typename Type>
    string_t build(const char_t* log_prefix, const char_t* log_suffix,const char_t* name, const std::unordered_set<Type>& type, const Parameters& container_args, const Parameters& type_args)
    {
        auto cargs = formats::parameters_to_length(type, container_args);
        return formats::format_iterator(log_prefix, log_suffix,name, type, cargs, type_args);
    }

    template <typename Type1, typename Type2>
    string_t build(const char_t* log_prefix, const char_t* log_suffix,const char_t* name, const std::map<Type1, Type2>& type, const Parameters& container_args, const Parameters& type_args)
    {
        auto cargs = formats::parameters_to_length(type, container_args);
        return formats::format_iterator(log_prefix, log_suffix,name, type, cargs, type_args);
    }

    template <typename Type1, typename Type2>
    string_t build(const char_t* log_prefix, const char_t* log_suffix,const char_t* name, const std::unordered_map<Type1, Type2>& type, const Parameters& container_args, const Parameters& type_args)
    {
        auto cargs = formats::parameters_to_length(type, container_args);
        return formats::format_iterator(log_prefix, log_suffix,name, type, cargs, type_args);
    }

    template <typename Type>
    string_t build(const char_t* log_prefix, const char_t* log_suffix,const char_t* name, const std::stack<Type>& type, const Parameters& container_args, const Parameters& type_args)
    {
        auto cargs = formats::parameters_to_length(type, container_args);
        return formats::format_stack(log_prefix, log_suffix,name, type, cargs, type_args);
    }

    template <typename Type>
    string_t build(const char_t* log_prefix, const char_t* log_suffix, const char_t* name, const std::queue<Type>& type, const Parameters& container_args, const Parameters& type_args)
    {
        auto cargs = formats::parameters_to_length(type, container_args);
        return formats::format_queue(log_prefix, log_suffix,name, type, cargs, type_args);
    }


private:
    string_t log_prefix_;
    string_t log_suffix_;
};
} // container
} // stl



#define REGISTER_STL_CONTAINER_ONE_PARAMETER(ContainerType)                                                                                     \
template <typename Type, typename... Args>                                                                                                      \
string_t message(const string_t& prefix, const string_t& suffix, const char_t* name, const ContainerType<Type>& type, const std::initializer_list<Args>&... args)                               \
{                                                                                                                                               \
    return delog::stl::container::Primitive(prefix, suffix).generate(name, type, args...);                                                                    \
}                                                                                                                                               \
template <typename Type>                                                                                                                        \
string_t message(const string_t& prefix, const string_t& suffix,const char_t* name, const ContainerType<Type>& type, const Parameters& container_args, const Parameters& type_args)            \
{                                                                                                                                               \
    return delog::stl::container::Primitive(prefix, suffix).generate(name, type, container_args, type_args);                                                  \
}

#define REGISTER_STL_CONTAINER_TWO_PARAMETER(ContainerType)                                                                                     \
template <typename T1, typename T2, typename... Args>                                                                                           \
string_t message(const string_t& prefix, const string_t& suffix,const char_t* name, const ContainerType<T1,T2>& type, const std::initializer_list<Args>&... args)                              \
{                                                                                                                                               \
    return delog::stl::container::Primitive(prefix, suffix).generate(name, type, args...);                                                                    \
}                                                                                                                                               \
template <typename T1, typename T2>                                                                                                             \
string_t message(const string_t& prefix, const string_t& suffix,const char_t* name, const ContainerType<T1,T2>& type, const Parameters& container_args, const Parameters& type_args)           \
{                                                                                                                                               \
    return delog::stl::container::Primitive(prefix, suffix).generate(name, type, container_args, type_args);                                                  \
}                                                                                  

#define REGISTER_STL_CONTAINER_TWO_PARAMETER_WITH_N(ContainerType)                                                                              \
template <typename T1, size_t N, typename... Args>                                                                                              \
string_t message(const string_t& prefix, const string_t& suffix,const char_t* name, const ContainerType<T1,N>& type, const std::initializer_list<Args>&... args)                               \
{                                                                                                                                               \
    return delog::stl::container::Primitive(prefix, suffix).generate(name, type, args...);                                                                    \
}                                                                                                                                               \
template <typename T1, size_t N>                                                                                                                \
string_t message(const string_t& prefix, const string_t& suffix,const char_t* name, const ContainerType<T1,N>& type, const Parameters& container_args, const Parameters& type_args)            \
{                                                                                                                                               \
    return delog::stl::container::Primitive(prefix, suffix).generate(name, type, container_args, type_args);                                                  \
}                                                                                  


REGISTER_STL_CONTAINER_ONE_PARAMETER(std::vector)
REGISTER_STL_CONTAINER_ONE_PARAMETER(std::list)
REGISTER_STL_CONTAINER_ONE_PARAMETER(std::deque)
REGISTER_STL_CONTAINER_ONE_PARAMETER(std::set)
REGISTER_STL_CONTAINER_ONE_PARAMETER(std::unordered_set)
REGISTER_STL_CONTAINER_ONE_PARAMETER(std::stack)
REGISTER_STL_CONTAINER_ONE_PARAMETER(std::queue)

REGISTER_STL_CONTAINER_TWO_PARAMETER_WITH_N(std::array)
REGISTER_STL_CONTAINER_TWO_PARAMETER(std::map)
REGISTER_STL_CONTAINER_TWO_PARAMETER(std::unordered_map)
}

#endif