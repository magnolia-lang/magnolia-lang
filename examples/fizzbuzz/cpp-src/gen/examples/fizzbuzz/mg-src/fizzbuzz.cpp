#include "gen/examples/fizzbuzz/mg-src/fizzbuzz.hpp"


namespace examples {
namespace fizzbuzz {
namespace mg_src {
namespace fizzbuzz {
    MyFizzBuzzProgram::FizzBuzz MyFizzBuzzProgram::buzz() {
        return __fizzbuzz_ops.buzz();
    };
    MyFizzBuzzProgram::FizzBuzz MyFizzBuzzProgram::doFizzBuzz(const MyFizzBuzzProgram::Int32& i) {
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
    MyFizzBuzzProgram::Int32 MyFizzBuzzProgram::five() {
        return __fizzbuzz_ops.five();
    };
    MyFizzBuzzProgram::FizzBuzz MyFizzBuzzProgram::fizz() {
        return __fizzbuzz_ops.fizz();
    };
    MyFizzBuzzProgram::FizzBuzz MyFizzBuzzProgram::fizzbuzz() {
        return __fizzbuzz_ops.fizzbuzz();
    };
    MyFizzBuzzProgram::Int32 MyFizzBuzzProgram::modulo(const MyFizzBuzzProgram::Int32& a, const MyFizzBuzzProgram::Int32& modulus) {
        return __fizzbuzz_ops.modulo(a, modulus);
    };
    MyFizzBuzzProgram::FizzBuzz MyFizzBuzzProgram::nope() {
        return __fizzbuzz_ops.nope();
    };
    MyFizzBuzzProgram::Int32 MyFizzBuzzProgram::three() {
        return __fizzbuzz_ops.three();
    };
    MyFizzBuzzProgram::Int32 MyFizzBuzzProgram::zero() {
        return __fizzbuzz_ops.zero();
    };
} // examples
} // fizzbuzz
} // mg_src
} // fizzbuzz