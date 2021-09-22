#include "base.hpp"
#include <cassert>


namespace examples {
namespace fizzbuzz {
namespace mg_src {
namespace fizzbuzz {
struct MyFizzBuzzProgram {
    struct _buzz;
    struct _doFizzBuzz;
    struct _five;
    struct _fizz;
    struct _fizzbuzz;
    struct _modulo;
    struct _nope;
    struct _three;
    struct _zero;
    typedef basic_types::Int32 Int32;
    typedef fizzbuzz_ops<Int32>::FizzBuzz FizzBuzz;
    struct _buzz {
        MyFizzBuzzProgram::FizzBuzz operator()();
    };

    struct _doFizzBuzz {
        MyFizzBuzzProgram::FizzBuzz operator()(const MyFizzBuzzProgram::Int32& i);
    };

    struct _five {
        MyFizzBuzzProgram::Int32 operator()();
    };

    struct _fizz {
        MyFizzBuzzProgram::FizzBuzz operator()();
    };

    struct _fizzbuzz {
        MyFizzBuzzProgram::FizzBuzz operator()();
    };

    struct _modulo {
        MyFizzBuzzProgram::Int32 operator()(const MyFizzBuzzProgram::Int32& a, const MyFizzBuzzProgram::Int32& modulus);
    };

    struct _nope {
        MyFizzBuzzProgram::FizzBuzz operator()();
    };

    struct _three {
        MyFizzBuzzProgram::Int32 operator()();
    };

    struct _zero {
        MyFizzBuzzProgram::Int32 operator()();
    };

private:
    static basic_types __basic_types;
public:
    static MyFizzBuzzProgram::_buzz buzz;
    static MyFizzBuzzProgram::_doFizzBuzz doFizzBuzz;
    static MyFizzBuzzProgram::_five five;
    static MyFizzBuzzProgram::_fizz fizz;
    static MyFizzBuzzProgram::_fizzbuzz fizzbuzz;
    static MyFizzBuzzProgram::_modulo modulo;
    static MyFizzBuzzProgram::_nope nope;
    static MyFizzBuzzProgram::_three three;
    static MyFizzBuzzProgram::_zero zero;
private:
    static fizzbuzz_ops<MyFizzBuzzProgram::Int32> __fizzbuzz_ops;
};
} // examples
} // fizzbuzz
} // mg_src
} // fizzbuzz