#include "gen/examples/fizzbuzz/mg-src/fizzbuzz.hpp"


namespace examples {
namespace fizzbuzz {
namespace mg_src {
namespace fizzbuzz {
    MyFizzBuzzProgram::FizzBuzz MyFizzBuzzProgram::buzz() {
        return __fizzbuzz_ops.buzz();
    };
    MyFizzBuzzProgram::FizzBuzz MyFizzBuzzProgram::doFizzBuzz(const MyFizzBuzzProgram::IntLike& i) {
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
    MyFizzBuzzProgram::IntLike MyFizzBuzzProgram::five() {
        return __fizzbuzz_ops.five();
    };
    MyFizzBuzzProgram::FizzBuzz MyFizzBuzzProgram::fizz() {
        return __fizzbuzz_ops.fizz();
    };
    MyFizzBuzzProgram::FizzBuzz MyFizzBuzzProgram::fizzbuzz() {
        return __fizzbuzz_ops.fizzbuzz();
    };
    MyFizzBuzzProgram::IntLike MyFizzBuzzProgram::modulo(const MyFizzBuzzProgram::IntLike& a, const MyFizzBuzzProgram::IntLike& modulus) {
        return __fizzbuzz_ops.modulo(a, modulus);
    };
    MyFizzBuzzProgram::FizzBuzz MyFizzBuzzProgram::nope() {
        return __fizzbuzz_ops.nope();
    };
    MyFizzBuzzProgram::IntLike MyFizzBuzzProgram::three() {
        return __fizzbuzz_ops.three();
    };
    MyFizzBuzzProgram::IntLike MyFizzBuzzProgram::zero() {
        return __fizzbuzz_ops.zero();
    };
} // examples
} // fizzbuzz
} // mg_src
} // fizzbuzz