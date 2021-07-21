#include <iostream>
#include "gen/fizzbuzz.hpp"

int main(int argc, char **argv) {
    --argc;
    if (argc < 1) {
        std::cerr << "Usage: " << argv[0] << " <number-to-fizzbuzz-test>"
                  << std::endl;
        return 1;
    }

    fizzbuzz::MyFizzBuzzProgram P;

    std::cout << "Fizz my buzz: " << P.doFizzBuzz(atoi(argv[1])) << std::endl;
}
