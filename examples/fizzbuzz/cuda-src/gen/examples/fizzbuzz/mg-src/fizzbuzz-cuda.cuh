#pragma once

#include "base.cuh"
#include <cassert>


namespace examples {
namespace fizzbuzz {
namespace mg_src {
namespace fizzbuzz_cuda {
struct MyFizzBuzzProgram {
private:
    basic_types __basic_types;
public:
    typedef basic_types::Int32 Int32;
private:
    fizzbuzz_ops<MyFizzBuzzProgram::Int32> __fizzbuzz_ops;
public:
    struct _five {
    private:
        fizzbuzz_ops<MyFizzBuzzProgram::Int32> __fizzbuzz_ops;
    public:
        __device__ __host__ inline MyFizzBuzzProgram::Int32 operator()() {
            return __fizzbuzz_ops.five();
        };
    };

    MyFizzBuzzProgram::_five five;
    struct _modulo {
    private:
        fizzbuzz_ops<MyFizzBuzzProgram::Int32> __fizzbuzz_ops;
    public:
        __device__ __host__ inline MyFizzBuzzProgram::Int32 operator()(const MyFizzBuzzProgram::Int32& a, const MyFizzBuzzProgram::Int32& modulus) {
            return __fizzbuzz_ops.modulo(a, modulus);
        };
    };

    MyFizzBuzzProgram::_modulo modulo;
    struct _three {
    private:
        fizzbuzz_ops<MyFizzBuzzProgram::Int32> __fizzbuzz_ops;
    public:
        __device__ __host__ inline MyFizzBuzzProgram::Int32 operator()() {
            return __fizzbuzz_ops.three();
        };
    };

    MyFizzBuzzProgram::_three three;
    struct _zero {
    private:
        fizzbuzz_ops<MyFizzBuzzProgram::Int32> __fizzbuzz_ops;
    public:
        __device__ __host__ inline MyFizzBuzzProgram::Int32 operator()() {
            return __fizzbuzz_ops.zero();
        };
    };

    MyFizzBuzzProgram::_zero zero;
    typedef fizzbuzz_ops<MyFizzBuzzProgram::Int32>::FizzBuzz FizzBuzz;
    struct _buzz {
    private:
        fizzbuzz_ops<MyFizzBuzzProgram::Int32> __fizzbuzz_ops;
    public:
        __device__ __host__ inline MyFizzBuzzProgram::FizzBuzz operator()() {
            return __fizzbuzz_ops.buzz();
        };
    };

    MyFizzBuzzProgram::_buzz buzz;
    struct _fizz {
    private:
        fizzbuzz_ops<MyFizzBuzzProgram::Int32> __fizzbuzz_ops;
    public:
        __device__ __host__ inline MyFizzBuzzProgram::FizzBuzz operator()() {
            return __fizzbuzz_ops.fizz();
        };
    };

    MyFizzBuzzProgram::_fizz fizz;
    struct _fizzbuzz {
    private:
        fizzbuzz_ops<MyFizzBuzzProgram::Int32> __fizzbuzz_ops;
    public:
        __device__ __host__ inline MyFizzBuzzProgram::FizzBuzz operator()() {
            return __fizzbuzz_ops.fizzbuzz();
        };
    };

    MyFizzBuzzProgram::_fizzbuzz fizzbuzz;
    struct _nope {
    private:
        fizzbuzz_ops<MyFizzBuzzProgram::Int32> __fizzbuzz_ops;
    public:
        __device__ __host__ inline MyFizzBuzzProgram::FizzBuzz operator()() {
            return __fizzbuzz_ops.nope();
        };
    };

    struct _doFizzBuzz {
    private:
        MyFizzBuzzProgram::_buzz buzz;
        MyFizzBuzzProgram::_five five;
        MyFizzBuzzProgram::_fizz fizz;
        MyFizzBuzzProgram::_fizzbuzz fizzbuzz;
        MyFizzBuzzProgram::_modulo modulo;
        MyFizzBuzzProgram::_nope nope;
        MyFizzBuzzProgram::_three three;
        MyFizzBuzzProgram::_zero zero;
    public:
        __device__ __host__ inline MyFizzBuzzProgram::FizzBuzz operator()(const MyFizzBuzzProgram::Int32& i) {
            if (((modulo(i, three())) == (zero())) && ((modulo(i, five())) == (zero())))
                return fizzbuzz();
            else
                if ((modulo(i, three())) == (zero()))
                    return fizz();
                else
                    if ((modulo(i, five())) == (zero()))
                        return buzz();
                    else
                        return nope();
        };
    };

    MyFizzBuzzProgram::_doFizzBuzz doFizzBuzz;
    MyFizzBuzzProgram::_nope nope;
};
} // examples
} // fizzbuzz
} // mg_src
} // fizzbuzz_cuda