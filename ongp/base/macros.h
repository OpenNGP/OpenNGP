#pragma one

#define SET_MEMBER_FUNC(Type, Variable)             \
inline void set_##Variable(Type Variable)           \
{                                                   \
    Variable##_ = Variable;                         \
}                                                   \

#define SET_MEMBER_FUNC_CONST(Type, Variable)             \
inline void set_##Variable(const Type Variable)           \
{                                                         \
    Variable##_ = Variable;                               \
}                                                         \

#define GET_MEMBER_FUNC(Type, Variable)     \
inline Type Variable()                      \
{                                           \
    return Variable##_;                     \
}                                           \

#define GET_MEMBER_FUNC_CONST(Type, Variable)   \
inline const Type Variable() const              \
{                                               \
    return Variable##_;                         \
}                                               \

#define SET_GET_MEMBER_FUNC(Type, Variable)  \
SET_MEMBER_FUNC(Type, Variable)              \
GET_MEMBER_FUNC(Type, Variable)              \

#define SET_GET_MEMBER_FUNC_CONST(Type, Variable)  \
SET_MEMBER_FUNC_CONST(Type, Variable)              \
GET_MEMBER_FUNC_CONST(Type, Variable)              \

//#define SET_MEMBER_FUNC_REFERENCE SET_MEMBER_FUNC
//#define GET_MEMBER_FUNC_REFERENCE GET_MEMBER_FUNC
//#define SET_GET_MEMBER_FUNC_REFERENCE SET_GET_MEMBER_FUNC
//
//
//#define SET_MEMBER_FUNC_POINTER(Type, Variable)             \
//inline void set_##Variable(Type* Variable)                  \
//{                                                           \
//    Variable##_ = Variable;                                 \
//}                                                           \
//
//#define GET_MEMBER_FUNC_POINTER(Type, Variable)     \
//inline const Type* Variable() const                 \
//{                                                   \
//    return Variable##_;                             \
//}                                                   \
//
//#define SET_GET_MEMBER_FUNC_POINTER(Type, Variable)  \
//SET_MEMBER_FUNC_POINTER(Type, Variable)              \
//GET_MEMBER_FUNC_POINTER(Type, Variable)              \
//