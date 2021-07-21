#include "base.hpp"
#include <cassert>


namespace fizzbuzz {
struct MyFizzBuzzProgram {
private:
    fizzbuzz_ops __fizzbuzz_ops;
protected:
    
public:
    typedef fizzbuzz_ops::FizzBuzz FizzBuzz;
    typedef fizzbuzz_ops::IntLike IntLike;
    MyFizzBuzzProgram::FizzBuzz buzz();
    MyFizzBuzzProgram::FizzBuzz doFizzBuzz(const MyFizzBuzzProgram::IntLike& i);
    MyFizzBuzzProgram::IntLike five();
    MyFizzBuzzProgram::FizzBuzz fizz();
    MyFizzBuzzProgram::FizzBuzz fizzbuzz();
    MyFizzBuzzProgram::IntLike modulo(const MyFizzBuzzProgram::IntLike& a, const MyFizzBuzzProgram::IntLike& modulus);
    MyFizzBuzzProgram::FizzBuzz nope();
    MyFizzBuzzProgram::IntLike three();
    MyFizzBuzzProgram::IntLike zero();
};
} // fizzbuzz