#ifndef ONGP_BASE_MACROS_H_
#define ONGP_BASE_MACROS_H_

#define SET_MEMBER_FUNC(Type, Variable)             \
inline void set_##Variable(const Type& Variable)    \
{                                                   \
    Variable##_ = Variable;                         \
}                                                   \

#define GET_MEMBER_FUNC(Type, Variable)     \
inline const Type& Variable() const         \
{                                           \
    return Variable##_;                     \
}                                           \

#define SET_GET_MEMBER_FUNC(Type, Variable)  \
SET_MEMBER_FUNC(Type, Variable)              \
GET_MEMBER_FUNC(Type, Variable)              \


#endif // ONGP_BASE_MACROS_H_