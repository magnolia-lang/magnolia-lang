#include <string>

template <typename _IntLike>
struct fizzbuzz_ops {
    typedef _IntLike IntLike;
    typedef std::string FizzBuzz;
    inline IntLike modulo(const IntLike a, const IntLike modulus) {
        return a % modulus;
    }
    inline IntLike five() { return 5; }
    inline IntLike three() { return 3; }
    inline IntLike zero() { return 0; }
    inline FizzBuzz fizz() { return "fizz"; }
    inline FizzBuzz buzz() { return "buzz"; }
    inline FizzBuzz fizzbuzz() { return "fizzbuzz"; }
    inline FizzBuzz nope() { return "nope"; }
};

struct basic_types {
    typedef int Int32;
};
