#include "base.hpp"
#include <cassert>


namespace examples {
namespace fizzbuzz {
namespace mg_src {
namespace fizzbuzz {
struct MyFizzBuzzProgram {
private:
    static basic_types __basic_types;
public:
    typedef basic_types::Int32 Int32;
private:
    static fizzbuzz_ops<MyFizzBuzzProgram::Int32> __fizzbuzz_ops;
public:
    struct _five {
        MyFizzBuzzProgram::Int32 operator()();
    };

    static MyFizzBuzzProgram::_five five;
    struct _modulo {
        MyFizzBuzzProgram::Int32 operator()(const MyFizzBuzzProgram::Int32& a, const MyFizzBuzzProgram::Int32& modulus);
    };

    static MyFizzBuzzProgram::_modulo modulo;
    struct _three {
        MyFizzBuzzProgram::Int32 operator()();
    };

    static MyFizzBuzzProgram::_three three;
    struct _zero {
        MyFizzBuzzProgram::Int32 operator()();
    };

    static MyFizzBuzzProgram::_zero zero;
    typedef fizzbuzz_ops<MyFizzBuzzProgram::Int32>::FizzBuzz FizzBuzz;
    struct _buzz {
        MyFizzBuzzProgram::FizzBuzz operator()();
    };

    static MyFizzBuzzProgram::_buzz buzz;
    struct _doFizzBuzz {
        MyFizzBuzzProgram::FizzBuzz operator()(const MyFizzBuzzProgram::Int32& i);
    };

    static MyFizzBuzzProgram::_doFizzBuzz doFizzBuzz;
    struct _fizz {
        MyFizzBuzzProgram::FizzBuzz operator()();
    };

    static MyFizzBuzzProgram::_fizz fizz;
    struct _fizzbuzz {
        MyFizzBuzzProgram::FizzBuzz operator()();
    };

    static MyFizzBuzzProgram::_fizzbuzz fizzbuzz;
    struct _nope {
        MyFizzBuzzProgram::FizzBuzz operator()();
    };

    static MyFizzBuzzProgram::_nope nope;
};
} // examples
} // fizzbuzz
} // mg_src
} // fizzbuzz