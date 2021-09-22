#include <utility>

template <typename _A, typename _B>
struct pair {
    typedef _A A;
    typedef _B B;
    typedef std::pair<pair::A, pair::B> Pair;

    inline pair::Pair make_pair(const pair::A &a, const pair::B &b) {
        return std::make_pair(a, b);
    }
    inline pair::A first(const pair::Pair &pair) {
        return pair.first;
    }
    inline pair::B second(const pair::Pair &pair) {
        return pair.second;
    }
};

struct types {
    typedef short Int16;
    typedef int Int32;
};
