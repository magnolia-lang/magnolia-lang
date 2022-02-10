#include <iostream>
#include "gen/examples/moa/mg-src/moa-cpp.hpp"

using examples::moa::mg_src::moa_cpp::ArrayProgram;

int main(int argc, char **argv) {

    ArrayProgram P;

    std::cout << "Linear representation of array a:" << std::endl;
    auto a = P.test_array3_2_2();

    P.print_array(a);

    std::cout <<"Shape of a:" << std::endl;
    P.print_shape(P.shape(a));

    std::cout <<"Dim of a:" << std::endl;
    std::cout << P.dim(a) << std::endl;

    std::cout <<"Total elements of a:" << std::endl;
    std::cout << P.total(a) << std::endl;

    std::cout << "Total index, access <1 0 0>: expect 6" << std::endl;
    auto test = P.get(a, P.test_index());

    std::cout << P.unwrap_scalar(test) << std::endl;

    std::cout << "Total index, access <1 1 1>: expect 3" << std::endl;
    auto test2 = P.get(a, P.create_index3(1,1,1));
    std::cout << P.unwrap_scalar(test2) << std::endl;

    std::cout << "TEST PARTIAL INDEX, ACCESS <0>: expect 2 5 7 8" << std::endl;

    auto subarray = P.get(a, P.create_index1(0));

    P.print_array(subarray);

    std::cout << "TEST PARTIAL INDEX, ACCESS <0 1>: expect 7 8" << std::endl;

    auto subarray2 = P.get(a, P.create_index2(0,1));

    P.print_array(subarray2);

    std::cout << "Cat testing on vectors:" << std::endl;

    auto vec1 = P.test_vector2();
    std::cout << "Vector vec1: " << std::endl;
    P.print_array(vec1);
    std::cout << "Shape: ";
    P.print_shape(P.shape(vec1));

    auto vec2 = P.test_vector3();
    std::cout << "Vector vec2: " << std::endl;
    P.print_array(vec2);
    std::cout << "Shape: ";
    P.print_shape(P.shape(vec2));

    std::cout << "cat(vec1, vec2):" << std::endl;

    auto cat_vec = P.cat_vec(vec1, vec2);
    P.print_array(cat_vec);
    std::cout << "Shape: ";
    P.print_shape(P.shape(cat_vec));


    std::cout << "CAT ON ARRAYS:" << std::endl;

    auto cat1 = P.test_array3_2_2();
    P.print_array(cat1);

    auto cat_res = P.cat(cat1,cat1);

    std::cout << "Shape of cat(a1,a1): ";
    P.print_shape(P.shape(cat_res));

    std::cout << "Linear res:" << std::endl;
    P.print_array(cat_res);


    std::cout << "PADDING TESTS" << std::endl;

    auto up = P.create_shape2(2,2);
    auto p = P.create_shape3(3,2,2);
    auto unp = P.test_array3_3();
    auto pad = P.test_array3_2_2();

    auto padded = P.padr(unp, pad);

    auto ix = P.create_index1(1);

    P.print_array(P.get(padded, ix));


    P.print_shape(P.shape(padded));
    P.print_shape(P.padded_shape(padded));
}