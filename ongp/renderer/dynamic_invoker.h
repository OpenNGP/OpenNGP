#include <boost/any.hpp>
#include <functional>
#include <iostream>
#include <list>

namespace ongp {
namespace renderer {
template <typename T>
auto fetch_back(T& t) ->
    typename std::remove_reference<decltype(t.back())>::type {
  typename std::remove_reference<decltype(t.back())>::type ret = t.back();
  t.pop_back();
  return ret;
}

template <typename X>
struct any_ref_cast {
  X do_cast(boost::any y) { return boost::any_cast<X>(y); }
};

template <typename X>
struct any_ref_cast<X&> {
  X& do_cast(boost::any y) {
    std::reference_wrapper<X> ref =
        boost::any_cast<std::reference_wrapper<X>>(y);
    return ref.get();
  }
};

template <typename X>
struct any_ref_cast<const X&> {
  const X& do_cast(boost::any y) {
    std::reference_wrapper<const X> ref =
        boost::any_cast<std::reference_wrapper<const X>>(y);
    return ref.get();
  }
};

template <typename Ret, typename... Arg>
Ret dynamic_call(Ret (*func)(Arg...), std::list<boost::any> args) {
  if (sizeof...(Arg) != args.size()) throw "Argument number mismatch!";

  return func(any_ref_cast<Arg>().do_cast(fetch_back(args))...);
}

}  // namespace renderer
}  // namespace ongp
