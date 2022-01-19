#include <iostream>
#include "gen/examples/moa/mg-src/moa-cpp.hpp"

int main(int argc, char **argv) {
    --argc;
    if (argc < 1) {
        std::cerr << "Usage: " << argv[0] << " <row a> <col a> <row b> <col b>"
                  << std::endl;
        return 1;
    }

    examples::moa::mg_src::moa_cpp::ArrayProgram P;

    std::cout << "Matrix creation:" << std::endl;
    auto mm = P.test_matrix();
    P.print_matrix(mm);

    std::cout << "repeat test" << std::endl;

    int32_utils::Int32 innerb, middleb, outerb, innerc, middlec, outerc;
    auto m1 = P.test_matrix();
    auto m2 = P.test_matrix();

    auto mres = P.zeros(5,5);

    innerb = 5;
    middleb = 5;
    outerb  = 5;
    innerc = 0;
    middlec = 0;
    outerc = 0;

    P.repeat(m1, m2, outerb, middleb, innerb, middlec, innerc, outerc, mres);

    P.print_matrix(mres);
    // TODO: add a base build path to compiler args to avoid having so many
    // nested useless folders when compiling from somewhere else.

}