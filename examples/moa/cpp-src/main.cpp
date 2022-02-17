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

    std::cout << "Shape of subarray: " << std::endl;

    P.print_shape(P.shape(subarray));

    std::cout << std::endl;
    std::cout << "TEST PARTIAL INDEX, ACCESS <0 1>: expect 7 8" << std::endl;

    auto subarray2 = P.get(a, P.create_index2(0,1));

    P.print_array(subarray2);

    std::cout << "Shape of subarray: " << std::endl;
    P.print_shape(P.shape(subarray2));

    std::cout << std::endl;
    std::cout << "CAT ON VECTORS:" << std::endl;
    std::cout << std::endl;
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

    std::cout << std::endl;

    std::cout << "CAT ON ARRAYS:" << std::endl;
    std::cout << std::endl;
    std::cout << "Array a1 = ";
    auto cat1 = P.test_array3_2_2();
    P.print_array(cat1);
    std::cout << "shape(a1) = ";
    P.print_shape(P.shape(cat1));

    auto cat_res = P.cat(cat1,cat1);

    std::cout << "Shape of cat(a1,a1): ";
    P.print_shape(P.shape(cat_res));

    std::cout << std::endl;
    std::cout << "cat(a1,a1) = " << std::endl;
    P.print_array(cat_res);

    std::cout << std::endl;

    std::cout << "PADDING TESTS" << std::endl;

    auto pad_test = P.test_array3_3();

    std::cout << "Original array:" << std::endl;
    P.print_array(pad_test);
    std::cout << "Shape: ";
    P.print_shape(P.shape(pad_test));

    std::cout << "circular_padr(A, 0): " << std::endl;
    auto padr = P.circular_padr(pad_test, 0);
    P.print_parray(padr);

    std::cout << "Padded shape:";
    P.print_shape(P.padded_shape(padr));

    std::cout << "Unpadded shape: ";

    P.print_shape(P.shape(padr));

    std::cout << std::endl;
    std::cout << "circular_padl(A, 0): " << std::endl;

    auto padl = P.circular_padl(pad_test, 0);
    P.print_parray(padl);

    std::cout << "Padded shape:";
    P.print_shape(P.padded_shape(padl));

    std::cout << "Unpadded shape: ";

    P.print_shape(P.shape(padl));

    std::cout << std::endl;
    std::cout << "Composition: padl(padr(a,0),0): TODO FIX" << std::endl;
    auto comp = P.circular_padl(P.circular_padr(padr,0),0);
    P.print_parray(comp);
    std::cout << "Shape of composed: ";
    P.print_shape(P.padded_shape(comp));

    std::cout << std::endl;
    std::cout << "TRANSFORMATIONS" << std::endl;
    std::cout << "Original array:" << std::endl;
    auto trans_array = P.test_array3_3();
    P.print_array(trans_array);

    std::cout << "transpose(a):" << std::endl;
    P.print_array(P.transpose(trans_array));

    std::cout << "reverse(a):" << std::endl;
    P.print_array(P.reverse(trans_array));


    std::cout << std::endl;
    std::cout << "INNER PRODUCT" << std::endl;
    std::cout << "Original arrays:" << std::endl;
    auto ip_array_input1 = P.test_array3_2_2();

    P.print_array(ip_array_input1);
    P.print_array(ip_array_input1);

    auto ip_res = P.inner_product(ip_array_input1,ip_array_input1);

    std::cout << "Shape of the inner product:" << std::endl;
    P.print_shape(P.shape(ip_res));
}