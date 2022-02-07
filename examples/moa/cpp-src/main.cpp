#include <iostream>
#include "gen/examples/moa/mg-src/moa-cpp.hpp"

using examples::moa::mg_src::moa_cpp::ArrayProgram;

int main(int argc, char **argv) {

    ArrayProgram P;

    auto sh = P.create_shape_3(2,3,3);

    auto a = P.create_array(sh);

    P.print_shape(sh);

    P.print_array(a);

    std::cout << P.dim(a) << std::endl;
    std::cout << P.total(a) << std::endl;
}