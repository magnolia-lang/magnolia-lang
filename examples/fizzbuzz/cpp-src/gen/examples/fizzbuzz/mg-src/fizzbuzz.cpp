#include "gen/examples/fizzbuzz/mg-src/fizzbuzz.hpp"


namespace examples {
namespace fizzbuzz {
namespace mg_src {
namespace fizzbuzz {
    basic_types MyFizzBuzzProgram::__basic_types;
    MyFizzBuzzProgram::_buzz MyFizzBuzzProgram::buzz;
    MyFizzBuzzProgram::_doFizzBuzz MyFizzBuzzProgram::doFizzBuzz;
    MyFizzBuzzProgram::_five MyFizzBuzzProgram::five;
    MyFizzBuzzProgram::_fizz MyFizzBuzzProgram::fizz;
    MyFizzBuzzProgram::_fizzbuzz MyFizzBuzzProgram::fizzbuzz;
    MyFizzBuzzProgram::_modulo MyFizzBuzzProgram::modulo;
    MyFizzBuzzProgram::_nope MyFizzBuzzProgram::nope;
    MyFizzBuzzProgram::_three MyFizzBuzzProgram::three;
    MyFizzBuzzProgram::_zero MyFizzBuzzProgram::zero;
    fizzbuzz_ops<MyFizzBuzzProgram::Int32> MyFizzBuzzProgram::__fizzbuzz_ops;
    MyFizzBuzzProgram::FizzBuzz MyFizzBuzzProgram::_buzz::operator()() {
        return __fizzbuzz_ops.buzz();
    };
    MyFizzBuzzProgram::FizzBuzz MyFizzBuzzProgram::_doFizzBuzz::operator()(const MyFizzBuzzProgram::Int32& i) {
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
    MyFizzBuzzProgram::Int32 MyFizzBuzzProgram::_five::operator()() {
        return __fizzbuzz_ops.five();
    };
    MyFizzBuzzProgram::FizzBuzz MyFizzBuzzProgram::_fizz::operator()() {
        return __fizzbuzz_ops.fizz();
    };
    MyFizzBuzzProgram::FizzBuzz MyFizzBuzzProgram::_fizzbuzz::operator()() {
        return __fizzbuzz_ops.fizzbuzz();
    };
    MyFizzBuzzProgram::Int32 MyFizzBuzzProgram::_modulo::operator()(const MyFizzBuzzProgram::Int32& a, const MyFizzBuzzProgram::Int32& modulus) {
        return __fizzbuzz_ops.modulo(a, modulus);
    };
    MyFizzBuzzProgram::FizzBuzz MyFizzBuzzProgram::_nope::operator()() {
        return __fizzbuzz_ops.nope();
    };
    MyFizzBuzzProgram::Int32 MyFizzBuzzProgram::_three::operator()() {
        return __fizzbuzz_ops.three();
    };
    MyFizzBuzzProgram::Int32 MyFizzBuzzProgram::_zero::operator()() {
        return __fizzbuzz_ops.zero();
    };

} // examples
} // fizzbuzz
} // mg_src
} // fizzbuzz