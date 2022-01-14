#include <string>

template <typename _IntLike> struct sum_ops {
    typedef _IntLike IntLike;
    typedef std::string Result;
    inline IntLike sum(const IntLike a, const IntLike b) {return a*b; }
    inline Result toString(const IntLike a) {return toString(a); }
};

struct basic_types {
    typedef int Int32;
};