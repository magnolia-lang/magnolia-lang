#pragma once

#include "base.hpp"
#include <cassert>


namespace examples {
namespace fizzbuzz {
namespace mg_src {
namespace fizzbuzz_cpp {
struct MyFizzBuzzProgram {
private:
    static basic_types __basic_types;
public:
    typedef basic_types::Int32 Int32;
private:
    static fizzbuzz_ops<MyFizzBuzzProgram::Int32> __fizzbuzz_ops;
public:
    struct _five {
        inline MyFizzBuzzProgram::Int32 operator()() {
            return __fizzbuzz_ops.five();
        };
    };

    static MyFizzBuzzProgram::_five five;
    struct _modulo {
        inline MyFizzBuzzProgram::Int32 operator()(const MyFizzBuzzProgram::Int32& a, const MyFizzBuzzProgram::Int32& modulus) {
            return __fizzbuzz_ops.modulo(a, modulus);
        };
    };

    static MyFizzBuzzProgram::_modulo modulo;
    struct _three {
        inline MyFizzBuzzProgram::Int32 operator()() {
            return __fizzbuzz_ops.three();
        };
    };

    static MyFizzBuzzProgram::_three three;
    struct _zero {
        inline MyFizzBuzzProgram::Int32 operator()() {
            return __fizzbuzz_ops.zero();
        };
    };

    static MyFizzBuzzProgram::_zero zero;
    typedef fizzbuzz_ops<MyFizzBuzzProgram::Int32>::FizzBuzz FizzBuzz;
    struct _buzz {
        inline MyFizzBuzzProgram::FizzBuzz operator()() {
            return __fizzbuzz_ops.buzz();
        };
    };

    static MyFizzBuzzProgram::_buzz buzz;
    struct _doFizzBuzz {
        inline MyFizzBuzzProgram::FizzBuzz operator()(const MyFizzBuzzProgram::Int32& i) {
            if (((MyFizzBuzzProgram::modulo(i, MyFizzBuzzProgram::three())) == (MyFizzBuzzProgram::zero())) && ((MyFizzBuzzProgram::modulo(i, MyFizzBuzzProgram::five())) == (MyFizzBuzzProgram::zero())))
                return MyFizzBuzzProgram::fizzbuzz();
            else
                if ((MyFizzBuzzProgram::modulo(i, MyFizzBuzzProgram::three())) == (MyFizzBuzzProgram::zero()))
                    return MyFizzBuzzProgram::fizz();
                else
                    if ((MyFizzBuzzProgram::modulo(i, MyFizzBuzzProgram::five())) == (MyFizzBuzzProgram::zero()))
                        return MyFizzBuzzProgram::buzz();
                    else
                        return MyFizzBuzzProgram::nope();
        };
    };

    static MyFizzBuzzProgram::_doFizzBuzz doFizzBuzz;
    struct _fizz {
        inline MyFizzBuzzProgram::FizzBuzz operator()() {
            return __fizzbuzz_ops.fizz();
        };
    };

    static MyFizzBuzzProgram::_fizz fizz;
    struct _fizzbuzz {
        inline MyFizzBuzzProgram::FizzBuzz operator()() {
            return __fizzbuzz_ops.fizzbuzz();
        };
    };

    static MyFizzBuzzProgram::_fizzbuzz fizzbuzz;
    struct _nope {
        inline MyFizzBuzzProgram::FizzBuzz operator()() {
            return __fizzbuzz_ops.nope();
        };
    };

    static MyFizzBuzzProgram::_nope nope;
};
} // examples
} // fizzbuzz
} // mg_src
} // fizzbuzz_cpp