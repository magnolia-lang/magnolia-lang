#include "base.hpp"
#include <cassert>


namespace examples {
namespace fizzbuzz {
namespace mg_src {
namespace fizzbuzz {
struct MyFizzBuzzProgram {
private:
    struct dummy_struct {};
public:
    typedef basic_types::Int32 Int32;
    typedef fizzbuzz_ops<dummy_struct>::FizzBuzz FizzBuzz;
    MyFizzBuzzProgram::FizzBuzz buzz();
    MyFizzBuzzProgram::FizzBuzz doFizzBuzz(const MyFizzBuzzProgram::Int32& i);
    MyFizzBuzzProgram::Int32 five();
    MyFizzBuzzProgram::FizzBuzz fizz();
    MyFizzBuzzProgram::FizzBuzz fizzbuzz();
    MyFizzBuzzProgram::Int32 modulo(const MyFizzBuzzProgram::Int32& a, const MyFizzBuzzProgram::Int32& modulus);
    MyFizzBuzzProgram::FizzBuzz nope();
    MyFizzBuzzProgram::Int32 three();
    MyFizzBuzzProgram::Int32 zero();
private:
    basic_types __basic_types;
    fizzbuzz_ops<Int32> __fizzbuzz_ops;

};
} // examples
} // fizzbuzz
} // mg_src
} // fizzbuzz