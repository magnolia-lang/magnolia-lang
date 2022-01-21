#include <iostream>
#include "gen/examples/moa/mg-src/moa-cpp.hpp"

using examples::moa::mg_src::moa_cpp::ArrayProgram;

int main(int argc, char **argv) {

    ArrayProgram P;

    auto m = P.test_matrix();
    std::cout << "Test matrix:" << std::endl;
    P.print_matrix(m);

    std::cout << "Test total index, access <1 2>, expect 5:" << std::endl;
    auto total = P.test_total_index();
    P.print_matrix(P.get(m,total));

    std::cout << "Test partial index, access <0>: expect <1 4 2 3 5>" << std::endl;
    auto partial = P.test_partial_index();
    auto s1 = P.get(m,partial);
    P.print_matrix(s1);

    std::cout << "Transposed M:" << std::endl;
    P.print_matrix(P.transpose(m));
    std::cout << "Test partial index on transposed M, access <0>: expect <1 3 2 4 5>" << std::endl;
    auto s2 = P.get(P.transpose(m), partial);
    P.print_matrix(s2);

    std::cout << "Shape of slice: " << std::endl;
    P.print_shape(s2);

    int32_utils::Int32 counter1;
    counter1 = 0;

    auto res1 = P.zeros(1,5);
    std::cout << "Multiply slices:" << std::endl;
    P.vecmult(s1,s2, res1, counter1);

    int32_utils::Int32 sumres1, sumcounter1;
    sumres1 = 0;
    sumcounter1 = 0;

    P.mapsum(res1, sumres1, sumcounter1);
    P.print_number(sumres1);

    std::cout << "Slice access at <2>:" << std::endl;
    auto getslice = P.get(s2, P.create_singleton_index(2));
    P.print_matrix(getslice);

    std::cout << "TEST MATMULT" << std::endl;

    auto m1 = P.test_matrix();
    auto m2 = P.transpose(P.test_matrix());

    auto res = P.zeros(5,5);

    int32_utils::Int32 i, k;
    i = 0;
    k = 0;

    P.matmult(m1, m2, res, i, k);
    P.print_matrix(res);
    // TODO: add a base build path to compiler args to avoid having so many
    // nested useless folders when compiling from somewhere else.

}