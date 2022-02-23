#pragma once

#include "base.hpp"
#include <cassert>


namespace examples {
namespace moa {
namespace mg_src {
namespace moa_cpp {
struct ArrayProgram {
private:
    static int32_utils __int32_utils;
public:
    typedef int32_utils::Int32 Int32;
    typedef array<ArrayProgram::Int32>::PaddedArray PaddedArray;
    struct _print_parray {
        inline void operator()(const ArrayProgram::PaddedArray& a) {
            return __array.print_parray(a);
        };
    };

    static ArrayProgram::_print_parray print_parray;
    typedef array<ArrayProgram::Int32>::Shape Shape;
    struct _cat_shape {
        inline ArrayProgram::Shape operator()(const ArrayProgram::Shape& a, const ArrayProgram::Shape& b) {
            return __array.cat_shape(a, b);
        };
    };

    static ArrayProgram::_cat_shape cat_shape;
    struct _padded_shape {
        inline ArrayProgram::Shape operator()(const ArrayProgram::PaddedArray& a) {
            return __array.padded_shape(a);
        };
    };

    static ArrayProgram::_padded_shape padded_shape;
    struct _print_shape {
        inline void operator()(const ArrayProgram::Shape& sh) {
            return __array.print_shape(sh);
        };
    };

    static ArrayProgram::_print_shape print_shape;
    struct _reverse_shape {
        inline ArrayProgram::Shape operator()(const ArrayProgram::Shape& s) {
            return __array.reverse_shape(s);
        };
    };

    static ArrayProgram::_reverse_shape reverse_shape;
    typedef array<ArrayProgram::Int32>::UInt32 UInt32;
    struct _create_shape1 {
        inline ArrayProgram::Shape operator()(const ArrayProgram::UInt32& a) {
            return __array.create_shape1(a);
        };
    };

    static ArrayProgram::_create_shape1 create_shape1;
    struct _create_shape2 {
        inline ArrayProgram::Shape operator()(const ArrayProgram::UInt32& a, const ArrayProgram::UInt32& b) {
            return __array.create_shape2(a, b);
        };
    };

    static ArrayProgram::_create_shape2 create_shape2;
    struct _create_shape3 {
        inline ArrayProgram::Shape operator()(const ArrayProgram::UInt32& a, const ArrayProgram::UInt32& b, const ArrayProgram::UInt32& c) {
            return __array.create_shape3(a, b, c);
        };
    };

    static ArrayProgram::_create_shape3 create_shape3;
    struct _padded_dim {
        inline ArrayProgram::UInt32 operator()(const ArrayProgram::PaddedArray& a) {
            return __array.padded_dim(a);
        };
    };

    static ArrayProgram::_padded_dim padded_dim;
    struct _padded_drop_shape_elem {
        inline ArrayProgram::Shape operator()(const ArrayProgram::PaddedArray& a, const ArrayProgram::UInt32& i) {
            return __array.padded_drop_shape_elem(a, i);
        };
    };

    static ArrayProgram::_padded_drop_shape_elem padded_drop_shape_elem;
    struct _padded_get_shape_elem {
        inline ArrayProgram::UInt32 operator()(const ArrayProgram::PaddedArray& a, const ArrayProgram::UInt32& i) {
            return __array.padded_get_shape_elem(a, i);
        };
    };

    static ArrayProgram::_padded_get_shape_elem padded_get_shape_elem;
    struct _padded_total {
        inline ArrayProgram::UInt32 operator()(const ArrayProgram::PaddedArray& a) {
            return __array.padded_total(a);
        };
    };

    static ArrayProgram::_padded_total padded_total;
    struct _print_uint {
        inline void operator()(const ArrayProgram::UInt32& u) {
            return __array.print_uint(u);
        };
    };

    static ArrayProgram::_print_uint print_uint;
private:
    static array<ArrayProgram::Int32> __array;
public:
    struct _elem_uint {
        inline ArrayProgram::UInt32 operator()(const ArrayProgram::Int32& a) {
            return __array.elem_uint(a);
        };
    };

    static ArrayProgram::_elem_uint elem_uint;
    struct _lt {
        inline bool operator()(const ArrayProgram::Int32& a, const ArrayProgram::Int32& b) {
            return __int32_utils.lt(a, b);
        };
    };

    static ArrayProgram::_lt lt;
    struct _one {
        inline ArrayProgram::Int32 operator()() {
            return __int32_utils.one();
        };
    };

    static ArrayProgram::_one one;
    struct _print_element {
        inline void operator()(const ArrayProgram::Int32& e) {
            return __array.print_element(e);
        };
    };

    static ArrayProgram::_print_element print_element;
    struct _uint_elem {
        inline ArrayProgram::Int32 operator()(const ArrayProgram::UInt32& a) {
            return __array.uint_elem(a);
        };
    };

    static ArrayProgram::_uint_elem uint_elem;
    struct _zero {
        inline ArrayProgram::Int32 operator()() {
            return __int32_utils.zero();
        };
    };

    static ArrayProgram::_zero zero;
    typedef array<ArrayProgram::Int32>::IndexContainer IndexContainer;
    struct _padded_transpose_body {
        inline void operator()(const ArrayProgram::PaddedArray& a, const ArrayProgram::IndexContainer& ixc, ArrayProgram::PaddedArray& res, ArrayProgram::UInt32& c) {
            ArrayProgram::Index current_ix = ArrayProgram::get_index_ixc(ixc, c);
            ArrayProgram::Int32 current_element = ArrayProgram::unwrap_scalar(ArrayProgram::get(a, ArrayProgram::reverse_index(current_ix)));
            ArrayProgram::set(res, current_ix, current_element);
            c = ArrayProgram::elem_uint(ArrayProgram::binary_add(ArrayProgram::uint_elem(c), ArrayProgram::one()));
        };
    };

    static ArrayProgram::_padded_transpose_body padded_transpose_body;
    struct _padded_transpose_repeat {
        inline void operator()(const ArrayProgram::PaddedArray& context1, const ArrayProgram::IndexContainer& context2, ArrayProgram::PaddedArray& state1, ArrayProgram::UInt32& state2) {
            return __while_loop2_27.repeat(context1, context2, state1, state2);
        };
    };

    static ArrayProgram::_padded_transpose_repeat padded_transpose_repeat;
    struct _padded_upper_bound {
        inline bool operator()(const ArrayProgram::PaddedArray& a, const ArrayProgram::IndexContainer& i, const ArrayProgram::PaddedArray& res, const ArrayProgram::UInt32& c) {
            return ArrayProgram::lt(ArrayProgram::uint_elem(c), ArrayProgram::uint_elem(ArrayProgram::total(i)));
        };
    };

private:
    static while_loop2_2<ArrayProgram::PaddedArray, ArrayProgram::IndexContainer, ArrayProgram::PaddedArray, ArrayProgram::UInt32, ArrayProgram::_padded_transpose_body, ArrayProgram::_padded_upper_bound> __while_loop2_27;
public:
    static ArrayProgram::_padded_upper_bound padded_upper_bound;
    struct _print_index_container {
        inline void operator()(const ArrayProgram::IndexContainer& i) {
            return __array.print_index_container(i);
        };
    };

    static ArrayProgram::_print_index_container print_index_container;
    typedef array<ArrayProgram::Int32>::Index Index;
    struct _cat_index {
        inline ArrayProgram::Index operator()(const ArrayProgram::Index& i, const ArrayProgram::Index& j) {
            return __array.cat_index(i, j);
        };
    };

    static ArrayProgram::_cat_index cat_index;
    struct _create_index1 {
        inline ArrayProgram::Index operator()(const ArrayProgram::UInt32& a) {
            return __array.create_index1(a);
        };
    };

    static ArrayProgram::_create_index1 create_index1;
    struct _create_index2 {
        inline ArrayProgram::Index operator()(const ArrayProgram::UInt32& a, const ArrayProgram::UInt32& b) {
            return __array.create_index2(a, b);
        };
    };

    static ArrayProgram::_create_index2 create_index2;
    struct _create_index3 {
        inline ArrayProgram::Index operator()(const ArrayProgram::UInt32& a, const ArrayProgram::UInt32& b, const ArrayProgram::UInt32& c) {
            return __array.create_index3(a, b, c);
        };
    };

    static ArrayProgram::_create_index3 create_index3;
    struct _drop_index_elem {
        inline ArrayProgram::Index operator()(const ArrayProgram::Index& ix, const ArrayProgram::UInt32& i) {
            return __array.drop_index_elem(ix, i);
        };
    };

    static ArrayProgram::_drop_index_elem drop_index_elem;
    struct _get_index_elem {
        inline ArrayProgram::UInt32 operator()(const ArrayProgram::Index& ix, const ArrayProgram::UInt32& i) {
            return __array.get_index_elem(ix, i);
        };
    };

    static ArrayProgram::_get_index_elem get_index_elem;
    struct _get_index_ixc {
        inline ArrayProgram::Index operator()(const ArrayProgram::IndexContainer& ixc, const ArrayProgram::UInt32& ix) {
            return __array.get_index_ixc(ixc, ix);
        };
    };

    static ArrayProgram::_get_index_ixc get_index_ixc;
    struct _print_index {
        inline void operator()(const ArrayProgram::Index& i) {
            return __array.print_index(i);
        };
    };

    static ArrayProgram::_print_index print_index;
    struct _reverse_index {
        inline ArrayProgram::Index operator()(const ArrayProgram::Index& ix) {
            return __array.reverse_index(ix);
        };
    };

    static ArrayProgram::_reverse_index reverse_index;
    struct _test_index {
        inline ArrayProgram::Index operator()() {
            return __array.test_index();
        };
    };

    static ArrayProgram::_test_index test_index;
    typedef array<ArrayProgram::Int32>::Array Array;
    struct _binaryMap {
        inline void operator()(const ArrayProgram::Array& a, const ArrayProgram::Array& b, const ArrayProgram::Index& ix) {
            assert((ArrayProgram::unwrap_scalar(ArrayProgram::get(ArrayProgram::binary_add(a, b), ix))) == (ArrayProgram::binary_add(ArrayProgram::unwrap_scalar(ArrayProgram::get(a, ix)), ArrayProgram::unwrap_scalar(ArrayProgram::get(b, ix)))));
            assert((ArrayProgram::unwrap_scalar(ArrayProgram::get(ArrayProgram::binary_sub(a, b), ix))) == (ArrayProgram::binary_sub(ArrayProgram::unwrap_scalar(ArrayProgram::get(a, ix)), ArrayProgram::unwrap_scalar(ArrayProgram::get(b, ix)))));
            assert((ArrayProgram::unwrap_scalar(ArrayProgram::get(ArrayProgram::mul(a, b), ix))) == (ArrayProgram::mul(ArrayProgram::unwrap_scalar(ArrayProgram::get(a, ix)), ArrayProgram::unwrap_scalar(ArrayProgram::get(b, ix)))));
            assert((ArrayProgram::unwrap_scalar(ArrayProgram::get(ArrayProgram::div(a, b), ix))) == (ArrayProgram::div(ArrayProgram::unwrap_scalar(ArrayProgram::get(a, ix)), ArrayProgram::unwrap_scalar(ArrayProgram::get(b, ix)))));
        };
    };

    static ArrayProgram::_binaryMap binaryMap;
    struct _binary_add {
        inline ArrayProgram::Int32 operator()(const ArrayProgram::Int32& a, const ArrayProgram::Int32& b) {
            return __int32_utils.binary_add(a, b);
        };
        inline ArrayProgram::Array operator()(const ArrayProgram::Array& a, const ArrayProgram::Array& b) {
            ArrayProgram::IndexContainer ix_space = ArrayProgram::create_valid_indices(a);
            ArrayProgram::Array b_upd = b;
            ArrayProgram::UInt32 counter = ArrayProgram::elem_uint(ArrayProgram::zero());
            ArrayProgram::bmb_plus_rep(a, ix_space, b_upd, counter);
            return b;
        };
    };

    static ArrayProgram::_binary_add binary_add;
    struct _binary_sub {
        inline ArrayProgram::Int32 operator()(const ArrayProgram::Int32& a, const ArrayProgram::Int32& b) {
            return __int32_utils.binary_sub(a, b);
        };
        inline ArrayProgram::Array operator()(const ArrayProgram::Array& a, const ArrayProgram::Array& b) {
            ArrayProgram::IndexContainer ix_space = ArrayProgram::create_valid_indices(a);
            ArrayProgram::Array b_upd = b;
            ArrayProgram::UInt32 counter = ArrayProgram::elem_uint(ArrayProgram::zero());
            ArrayProgram::bmb_sub_rep(a, ix_space, b_upd, counter);
            return b;
        };
    };

    static ArrayProgram::_binary_sub binary_sub;
    struct _bmb_div {
        inline void operator()(const ArrayProgram::Array& a, const ArrayProgram::IndexContainer& ix_space, ArrayProgram::Array& b, ArrayProgram::UInt32& c) {
            ArrayProgram::Index ix = ArrayProgram::get_index_ixc(ix_space, c);
            ArrayProgram::Int32 new_value = ArrayProgram::div(ArrayProgram::unwrap_scalar(ArrayProgram::get(a, ix)), ArrayProgram::unwrap_scalar(ArrayProgram::get(b, ix)));
            ArrayProgram::set(b, ix, new_value);
            c = ArrayProgram::elem_uint(ArrayProgram::binary_add(ArrayProgram::uint_elem(c), ArrayProgram::one()));
        };
    };

    static ArrayProgram::_bmb_div bmb_div;
    struct _bmb_div_rep {
        inline void operator()(const ArrayProgram::Array& context1, const ArrayProgram::IndexContainer& context2, ArrayProgram::Array& state1, ArrayProgram::UInt32& state2) {
            return __while_loop2_21.repeat(context1, context2, state1, state2);
        };
    };

    static ArrayProgram::_bmb_div_rep bmb_div_rep;
    struct _bmb_mul {
        inline void operator()(const ArrayProgram::Array& a, const ArrayProgram::IndexContainer& ix_space, ArrayProgram::Array& b, ArrayProgram::UInt32& c) {
            ArrayProgram::Index ix = ArrayProgram::get_index_ixc(ix_space, c);
            ArrayProgram::Int32 new_value = ArrayProgram::mul(ArrayProgram::unwrap_scalar(ArrayProgram::get(a, ix)), ArrayProgram::unwrap_scalar(ArrayProgram::get(b, ix)));
            ArrayProgram::set(b, ix, new_value);
            c = ArrayProgram::elem_uint(ArrayProgram::binary_add(ArrayProgram::uint_elem(c), ArrayProgram::one()));
        };
    };

    static ArrayProgram::_bmb_mul bmb_mul;
    struct _bmb_mul_rep {
        inline void operator()(const ArrayProgram::Array& context1, const ArrayProgram::IndexContainer& context2, ArrayProgram::Array& state1, ArrayProgram::UInt32& state2) {
            return __while_loop2_22.repeat(context1, context2, state1, state2);
        };
    };

    static ArrayProgram::_bmb_mul_rep bmb_mul_rep;
    struct _bmb_plus {
        inline void operator()(const ArrayProgram::Array& a, const ArrayProgram::IndexContainer& ix_space, ArrayProgram::Array& b, ArrayProgram::UInt32& c) {
            ArrayProgram::Index ix = ArrayProgram::get_index_ixc(ix_space, c);
            ArrayProgram::Int32 new_value = ArrayProgram::binary_add(ArrayProgram::unwrap_scalar(ArrayProgram::get(a, ix)), ArrayProgram::unwrap_scalar(ArrayProgram::get(b, ix)));
            ArrayProgram::set(b, ix, new_value);
            c = ArrayProgram::elem_uint(ArrayProgram::binary_add(ArrayProgram::uint_elem(c), ArrayProgram::one()));
        };
    };

    static ArrayProgram::_bmb_plus bmb_plus;
    struct _bmb_plus_rep {
        inline void operator()(const ArrayProgram::Array& context1, const ArrayProgram::IndexContainer& context2, ArrayProgram::Array& state1, ArrayProgram::UInt32& state2) {
            return __while_loop2_23.repeat(context1, context2, state1, state2);
        };
    };

    static ArrayProgram::_bmb_plus_rep bmb_plus_rep;
    struct _bmb_sub {
        inline void operator()(const ArrayProgram::Array& a, const ArrayProgram::IndexContainer& ix_space, ArrayProgram::Array& b, ArrayProgram::UInt32& c) {
            ArrayProgram::Index ix = ArrayProgram::get_index_ixc(ix_space, c);
            ArrayProgram::Int32 new_value = ArrayProgram::binary_sub(ArrayProgram::unwrap_scalar(ArrayProgram::get(a, ix)), ArrayProgram::unwrap_scalar(ArrayProgram::get(b, ix)));
            ArrayProgram::set(b, ix, new_value);
            c = ArrayProgram::elem_uint(ArrayProgram::binary_add(ArrayProgram::uint_elem(c), ArrayProgram::one()));
        };
    };

    static ArrayProgram::_bmb_sub bmb_sub;
    struct _bmb_sub_rep {
        inline void operator()(const ArrayProgram::Array& context1, const ArrayProgram::IndexContainer& context2, ArrayProgram::Array& state1, ArrayProgram::UInt32& state2) {
            return __while_loop2_24.repeat(context1, context2, state1, state2);
        };
    };

    static ArrayProgram::_bmb_sub_rep bmb_sub_rep;
    struct _cat {
        inline ArrayProgram::Array operator()(const ArrayProgram::Array& array1, const ArrayProgram::Array& array2) {
            ArrayProgram::Int32 take_a1 = ArrayProgram::uint_elem(ArrayProgram::get_shape_elem(array1, ArrayProgram::elem_uint(ArrayProgram::zero())));
            ArrayProgram::Int32 take_a2 = ArrayProgram::uint_elem(ArrayProgram::get_shape_elem(array2, ArrayProgram::elem_uint(ArrayProgram::zero())));
            ArrayProgram::Shape drop_a1 = ArrayProgram::drop_shape_elem(array1, ArrayProgram::elem_uint(ArrayProgram::zero()));
            ArrayProgram::Shape result_shape = ArrayProgram::cat_shape(ArrayProgram::create_shape1(ArrayProgram::elem_uint(ArrayProgram::binary_add(take_a1, take_a2))), drop_a1);
            ArrayProgram::Array res = ArrayProgram::create_array(result_shape);
            ArrayProgram::UInt32 counter = ArrayProgram::elem_uint(ArrayProgram::zero());
            ArrayProgram::cat_repeat(array1, array2, counter, res);
            return res;
        };
    };

    static ArrayProgram::_cat cat;
    struct _cat_body {
        inline void operator()(const ArrayProgram::Array& array1, const ArrayProgram::Array& array2, ArrayProgram::UInt32& counter, ArrayProgram::Array& res) {
            ArrayProgram::Int32 s_0 = ArrayProgram::uint_elem(ArrayProgram::total(array1));
            if (ArrayProgram::lt(ArrayProgram::uint_elem(counter), s_0))
            {
                ArrayProgram::set(res, counter, ArrayProgram::unwrap_scalar(ArrayProgram::get(array1, counter)));
                counter = ArrayProgram::elem_uint(ArrayProgram::binary_add(ArrayProgram::uint_elem(counter), ArrayProgram::one()));
            }
            else
            {
                ArrayProgram::UInt32 ix = ArrayProgram::elem_uint(ArrayProgram::binary_sub(ArrayProgram::uint_elem(counter), s_0));
                ArrayProgram::set(res, counter, ArrayProgram::unwrap_scalar(ArrayProgram::get(array2, ix)));
                counter = ArrayProgram::elem_uint(ArrayProgram::binary_add(ArrayProgram::uint_elem(counter), ArrayProgram::one()));
            }
        };
    };

    static ArrayProgram::_cat_body cat_body;
    struct _cat_cond {
        inline bool operator()(const ArrayProgram::Array& array1, const ArrayProgram::Array& array2, const ArrayProgram::UInt32& counter, const ArrayProgram::Array& res) {
            ArrayProgram::Int32 upper_bound = ArrayProgram::uint_elem(ArrayProgram::total(res));
            return ArrayProgram::lt(ArrayProgram::uint_elem(counter), upper_bound);
        };
    };

private:
    static while_loop2_2<ArrayProgram::Array, ArrayProgram::Array, ArrayProgram::UInt32, ArrayProgram::Array, ArrayProgram::_cat_body, ArrayProgram::_cat_cond> __while_loop2_20;
public:
    static ArrayProgram::_cat_cond cat_cond;
    struct _cat_repeat {
        inline void operator()(const ArrayProgram::Array& context1, const ArrayProgram::Array& context2, ArrayProgram::UInt32& state1, ArrayProgram::Array& state2) {
            return __while_loop2_20.repeat(context1, context2, state1, state2);
        };
    };

    static ArrayProgram::_cat_repeat cat_repeat;
    struct _cat_vec {
        inline ArrayProgram::Array operator()(const ArrayProgram::Array& vector1, const ArrayProgram::Array& vector2) {
            ArrayProgram::Shape res_shape = ArrayProgram::create_shape1(ArrayProgram::elem_uint(ArrayProgram::binary_add(ArrayProgram::uint_elem(ArrayProgram::total(vector1)), ArrayProgram::uint_elem(ArrayProgram::total(vector2)))));
            ArrayProgram::Array res = ArrayProgram::create_array(res_shape);
            ArrayProgram::UInt32 counter = ArrayProgram::elem_uint(ArrayProgram::zero());
            ArrayProgram::cat_vec_repeat(vector1, vector2, res, counter);
            return res;
        };
    };

    static ArrayProgram::_cat_vec cat_vec;
    struct _cat_vec_body {
        inline void operator()(const ArrayProgram::Array& v1, const ArrayProgram::Array& v2, ArrayProgram::Array& res, ArrayProgram::UInt32& counter) {
            ArrayProgram::Int32 v1_bound = ArrayProgram::uint_elem(ArrayProgram::total(v1));
            ArrayProgram::Index ix;
            if (ArrayProgram::lt(ArrayProgram::uint_elem(counter), ArrayProgram::uint_elem(ArrayProgram::total(v1))))
            {
                ix = ArrayProgram::create_index1(counter);
                ArrayProgram::set(res, ix, ArrayProgram::unwrap_scalar(ArrayProgram::get(v1, ix)));
            }
            else
            {
                ix = ArrayProgram::create_index1(ArrayProgram::elem_uint(ArrayProgram::binary_sub(ArrayProgram::uint_elem(counter), v1_bound)));
                ArrayProgram::Index res_ix = ArrayProgram::create_index1(counter);
                ArrayProgram::set(res, res_ix, ArrayProgram::unwrap_scalar(ArrayProgram::get(v2, ix)));
            }
            counter = ArrayProgram::elem_uint(ArrayProgram::binary_add(ArrayProgram::uint_elem(counter), ArrayProgram::one()));
        };
    };

    static ArrayProgram::_cat_vec_body cat_vec_body;
    struct _cat_vec_cond {
        inline bool operator()(const ArrayProgram::Array& v1, const ArrayProgram::Array& v2, const ArrayProgram::Array& res, const ArrayProgram::UInt32& counter) {
            return ArrayProgram::lt(ArrayProgram::uint_elem(counter), ArrayProgram::uint_elem(ArrayProgram::total(res)));
        };
    };

private:
    static while_loop2_2<ArrayProgram::Array, ArrayProgram::Array, ArrayProgram::Array, ArrayProgram::UInt32, ArrayProgram::_cat_vec_body, ArrayProgram::_cat_vec_cond> __while_loop2_2;
public:
    static ArrayProgram::_cat_vec_cond cat_vec_cond;
    struct _cat_vec_repeat {
        inline void operator()(const ArrayProgram::Array& context1, const ArrayProgram::Array& context2, ArrayProgram::Array& state1, ArrayProgram::UInt32& state2) {
            return __while_loop2_2.repeat(context1, context2, state1, state2);
        };
    };

    static ArrayProgram::_cat_vec_repeat cat_vec_repeat;
    struct _circular_padl {
        inline ArrayProgram::PaddedArray operator()(const ArrayProgram::PaddedArray& a, const ArrayProgram::UInt32& ix) {
            ArrayProgram::Array padding = ArrayProgram::get(a, ArrayProgram::create_index1(ix));
            ArrayProgram::Shape reshape_shape = ArrayProgram::cat_shape(ArrayProgram::create_shape1(ArrayProgram::elem_uint(ArrayProgram::one())), ArrayProgram::shape(padding));
            ArrayProgram::Array reshaped_padding = ArrayProgram::reshape(padding, reshape_shape);
            ArrayProgram::Array catenated_array = ArrayProgram::cat(reshaped_padding, ArrayProgram::padded_to_unpadded(a));
            ArrayProgram::Shape unpadded_shape = ArrayProgram::shape(a);
            ArrayProgram::Shape padded_shape = ArrayProgram::shape(catenated_array);
            ArrayProgram::PaddedArray res = ArrayProgram::create_padded_array(unpadded_shape, padded_shape, catenated_array);
            return res;
        };
        inline ArrayProgram::PaddedArray operator()(const ArrayProgram::Array& a, const ArrayProgram::UInt32& ix) {
            ArrayProgram::Array padding = ArrayProgram::get(a, ArrayProgram::create_index1(ix));
            ArrayProgram::Shape reshape_shape = ArrayProgram::cat_shape(ArrayProgram::create_shape1(ArrayProgram::elem_uint(ArrayProgram::one())), ArrayProgram::shape(padding));
            ArrayProgram::Array reshaped_padding = ArrayProgram::reshape(padding, reshape_shape);
            ArrayProgram::Array catenated_array = ArrayProgram::cat(reshaped_padding, a);
            ArrayProgram::Shape unpadded_shape = ArrayProgram::shape(a);
            ArrayProgram::Shape padded_shape = ArrayProgram::shape(catenated_array);
            ArrayProgram::PaddedArray res = ArrayProgram::create_padded_array(unpadded_shape, padded_shape, catenated_array);
            return res;
        };
    };

    static ArrayProgram::_circular_padl circular_padl;
    struct _circular_padr {
        inline ArrayProgram::PaddedArray operator()(const ArrayProgram::PaddedArray& a, const ArrayProgram::UInt32& ix) {
            ArrayProgram::Shape unpadded_shape = ArrayProgram::shape(a);
            ArrayProgram::Array padding = ArrayProgram::get(a, ArrayProgram::create_index1(ix));
            ArrayProgram::Shape reshape_shape = ArrayProgram::cat_shape(ArrayProgram::create_shape1(ArrayProgram::elem_uint(ArrayProgram::one())), ArrayProgram::shape(padding));
            ArrayProgram::Array reshaped_padding = ArrayProgram::reshape(padding, reshape_shape);
            ArrayProgram::Array catenated_array = ArrayProgram::cat(ArrayProgram::padded_to_unpadded(a), reshaped_padding);
            ArrayProgram::Shape padded_shape = ArrayProgram::shape(catenated_array);
            ArrayProgram::PaddedArray res = ArrayProgram::create_padded_array(unpadded_shape, padded_shape, catenated_array);
            return res;
        };
        inline ArrayProgram::PaddedArray operator()(const ArrayProgram::Array& a, const ArrayProgram::UInt32& ix) {
            ArrayProgram::Array padding = ArrayProgram::get(a, ArrayProgram::create_index1(ix));
            ArrayProgram::Shape reshape_shape = ArrayProgram::cat_shape(ArrayProgram::create_shape1(ArrayProgram::elem_uint(ArrayProgram::one())), ArrayProgram::shape(padding));
            ArrayProgram::Array reshaped_padding = ArrayProgram::reshape(padding, reshape_shape);
            ArrayProgram::Array catenated_array = ArrayProgram::cat(a, reshaped_padding);
            ArrayProgram::Shape unpadded_shape = ArrayProgram::shape(a);
            ArrayProgram::Shape padded_shape = ArrayProgram::shape(catenated_array);
            ArrayProgram::PaddedArray res = ArrayProgram::create_padded_array(unpadded_shape, padded_shape, catenated_array);
            return res;
        };
    };

    static ArrayProgram::_circular_padr circular_padr;
    struct _create_array {
        inline ArrayProgram::Array operator()(const ArrayProgram::Shape& sh) {
            return __array.create_array(sh);
        };
    };

    static ArrayProgram::_create_array create_array;
    struct _create_padded_array {
        inline ArrayProgram::PaddedArray operator()(const ArrayProgram::Shape& unpadded_shape, const ArrayProgram::Shape& padded_shape, const ArrayProgram::Array& padded_array) {
            return __array.create_padded_array(unpadded_shape, padded_shape, padded_array);
        };
    };

    static ArrayProgram::_create_padded_array create_padded_array;
    struct _create_valid_indices {
        inline ArrayProgram::IndexContainer operator()(const ArrayProgram::PaddedArray& a) {
            return __array.create_valid_indices(a);
        };
        inline ArrayProgram::IndexContainer operator()(const ArrayProgram::Array& a) {
            return __array.create_valid_indices(a);
        };
    };

    static ArrayProgram::_create_valid_indices create_valid_indices;
    struct _dim {
        inline ArrayProgram::UInt32 operator()(const ArrayProgram::Array& a) {
            return __array.dim(a);
        };
    };

    static ArrayProgram::_dim dim;
    struct _div {
        inline ArrayProgram::Int32 operator()(const ArrayProgram::Int32& a, const ArrayProgram::Int32& b) {
            return __int32_utils.div(a, b);
        };
        inline ArrayProgram::Array operator()(const ArrayProgram::Array& a, const ArrayProgram::Array& b) {
            ArrayProgram::IndexContainer ix_space = ArrayProgram::create_valid_indices(a);
            ArrayProgram::Array b_upd = b;
            ArrayProgram::UInt32 counter = ArrayProgram::elem_uint(ArrayProgram::zero());
            ArrayProgram::bmb_div_rep(a, ix_space, b_upd, counter);
            return b;
        };
    };

    static ArrayProgram::_div div;
    struct _drop_shape_elem {
        inline ArrayProgram::Shape operator()(const ArrayProgram::Array& a, const ArrayProgram::UInt32& i) {
            return __array.drop_shape_elem(a, i);
        };
    };

    static ArrayProgram::_drop_shape_elem drop_shape_elem;
    struct _get {
        inline ArrayProgram::Array operator()(const ArrayProgram::PaddedArray& a, const ArrayProgram::Index& ix) {
            return __array.get(a, ix);
        };
        inline ArrayProgram::Array operator()(const ArrayProgram::Array& a, const ArrayProgram::Index& ix) {
            return __array.get(a, ix);
        };
        inline ArrayProgram::Array operator()(const ArrayProgram::Array& a, const ArrayProgram::UInt32& ix) {
            return __array.get(a, ix);
        };
    };

    static ArrayProgram::_get get;
    struct _get_shape_elem {
        inline ArrayProgram::UInt32 operator()(const ArrayProgram::Array& a, const ArrayProgram::UInt32& i) {
            return __array.get_shape_elem(a, i);
        };
    };

    static ArrayProgram::_get_shape_elem get_shape_elem;
    struct _mapped_ops_cond {
        inline bool operator()(const ArrayProgram::Array& a, const ArrayProgram::IndexContainer& ix_space, const ArrayProgram::Array& b, const ArrayProgram::UInt32& c) {
            return ArrayProgram::lt(ArrayProgram::uint_elem(c), ArrayProgram::uint_elem(ArrayProgram::total(ix_space)));
        };
    };

private:
    static while_loop2_2<ArrayProgram::Array, ArrayProgram::IndexContainer, ArrayProgram::Array, ArrayProgram::UInt32, ArrayProgram::_bmb_div, ArrayProgram::_mapped_ops_cond> __while_loop2_21;
    static while_loop2_2<ArrayProgram::Array, ArrayProgram::IndexContainer, ArrayProgram::Array, ArrayProgram::UInt32, ArrayProgram::_bmb_mul, ArrayProgram::_mapped_ops_cond> __while_loop2_22;
    static while_loop2_2<ArrayProgram::Array, ArrayProgram::IndexContainer, ArrayProgram::Array, ArrayProgram::UInt32, ArrayProgram::_bmb_plus, ArrayProgram::_mapped_ops_cond> __while_loop2_23;
    static while_loop2_2<ArrayProgram::Array, ArrayProgram::IndexContainer, ArrayProgram::Array, ArrayProgram::UInt32, ArrayProgram::_bmb_sub, ArrayProgram::_mapped_ops_cond> __while_loop2_24;
public:
    static ArrayProgram::_mapped_ops_cond mapped_ops_cond;
    struct _mul {
        inline ArrayProgram::Int32 operator()(const ArrayProgram::Int32& a, const ArrayProgram::Int32& b) {
            return __int32_utils.mul(a, b);
        };
        inline ArrayProgram::Array operator()(const ArrayProgram::Array& a, const ArrayProgram::Array& b) {
            ArrayProgram::IndexContainer ix_space = ArrayProgram::create_valid_indices(a);
            ArrayProgram::Array b_upd = b;
            ArrayProgram::UInt32 counter = ArrayProgram::elem_uint(ArrayProgram::zero());
            ArrayProgram::bmb_mul_rep(a, ix_space, b_upd, counter);
            return b;
        };
    };

    static ArrayProgram::_mul mul;
    struct _padded_to_unpadded {
        inline ArrayProgram::Array operator()(const ArrayProgram::PaddedArray& a) {
            return __array.padded_to_unpadded(a);
        };
    };

    static ArrayProgram::_padded_to_unpadded padded_to_unpadded;
    struct _print_array {
        inline void operator()(const ArrayProgram::Array& a) {
            return __array.print_array(a);
        };
    };

    static ArrayProgram::_print_array print_array;
    struct _reshape {
        inline ArrayProgram::Array operator()(const ArrayProgram::Array& input_array, const ArrayProgram::Shape& s) {
            ArrayProgram::Array new_array = ArrayProgram::create_array(s);
            ArrayProgram::UInt32 counter = ArrayProgram::elem_uint(ArrayProgram::zero());
            ArrayProgram::reshape_repeat(input_array, new_array, counter);
            return new_array;
        };
    };

    static ArrayProgram::_reshape reshape;
    struct _reshape_body {
        inline void operator()(const ArrayProgram::Array& old_array, ArrayProgram::Array& new_array, ArrayProgram::UInt32& counter) {
            ArrayProgram::set(new_array, counter, ArrayProgram::unwrap_scalar(ArrayProgram::get(old_array, counter)));
            counter = ArrayProgram::elem_uint(ArrayProgram::binary_add(ArrayProgram::uint_elem(counter), ArrayProgram::one()));
        };
    };

    static ArrayProgram::_reshape_body reshape_body;
    struct _reshape_cond {
        inline bool operator()(const ArrayProgram::Array& old_array, const ArrayProgram::Array& new_array, const ArrayProgram::UInt32& counter) {
            return ArrayProgram::lt(ArrayProgram::uint_elem(counter), ArrayProgram::uint_elem(ArrayProgram::total(new_array)));
        };
    };

private:
    static while_loop1_2<ArrayProgram::Array, ArrayProgram::Array, ArrayProgram::UInt32, ArrayProgram::_reshape_body, ArrayProgram::_reshape_cond> __while_loop1_2;
public:
    static ArrayProgram::_reshape_cond reshape_cond;
    struct _reshape_repeat {
        inline void operator()(const ArrayProgram::Array& context1, ArrayProgram::Array& state1, ArrayProgram::UInt32& state2) {
            return __while_loop1_2.repeat(context1, state1, state2);
        };
    };

    static ArrayProgram::_reshape_repeat reshape_repeat;
    struct _reverse {
        inline ArrayProgram::Array operator()(const ArrayProgram::Array& a) {
            ArrayProgram::Array res_array = ArrayProgram::create_array(ArrayProgram::shape(a));
            ArrayProgram::IndexContainer valid_indices = ArrayProgram::create_valid_indices(res_array);
            ArrayProgram::UInt32 counter = ArrayProgram::elem_uint(ArrayProgram::zero());
            ArrayProgram::reverse_repeat(a, valid_indices, res_array, counter);
            return res_array;
        };
    };

    static ArrayProgram::_reverse reverse;
    struct _reverse_body {
        inline void operator()(const ArrayProgram::Array& input, const ArrayProgram::IndexContainer& indices, ArrayProgram::Array& res, ArrayProgram::UInt32& c) {
            ArrayProgram::Index ix = ArrayProgram::get_index_ixc(indices, c);
            ArrayProgram::Int32 elem = ArrayProgram::unwrap_scalar(ArrayProgram::get(input, ix));
            ArrayProgram::UInt32 sh_0 = ArrayProgram::get_shape_elem(input, ArrayProgram::elem_uint(ArrayProgram::zero()));
            ArrayProgram::UInt32 ix_0 = ArrayProgram::get_index_elem(ix, ArrayProgram::elem_uint(ArrayProgram::zero()));
            ArrayProgram::Int32 new_ix_0 = ArrayProgram::binary_sub(ArrayProgram::uint_elem(sh_0), ArrayProgram::binary_add(ArrayProgram::uint_elem(ix_0), ArrayProgram::one()));
            ArrayProgram::Index new_ix = ArrayProgram::cat_index(ArrayProgram::create_index1(ArrayProgram::elem_uint(new_ix_0)), ArrayProgram::drop_index_elem(ix, ArrayProgram::elem_uint(ArrayProgram::zero())));
            ArrayProgram::set(res, new_ix, elem);
            c = ArrayProgram::elem_uint(ArrayProgram::binary_add(ArrayProgram::uint_elem(c), ArrayProgram::one()));
        };
    };

    static ArrayProgram::_reverse_body reverse_body;
    struct _reverse_cond {
        inline bool operator()(const ArrayProgram::Array& input, const ArrayProgram::IndexContainer& indices, const ArrayProgram::Array& res, const ArrayProgram::UInt32& c) {
            return ArrayProgram::lt(ArrayProgram::uint_elem(c), ArrayProgram::uint_elem(ArrayProgram::total(indices)));
        };
    };

private:
    static while_loop2_2<ArrayProgram::Array, ArrayProgram::IndexContainer, ArrayProgram::Array, ArrayProgram::UInt32, ArrayProgram::_reverse_body, ArrayProgram::_reverse_cond> __while_loop2_25;
public:
    static ArrayProgram::_reverse_cond reverse_cond;
    struct _reverse_repeat {
        inline void operator()(const ArrayProgram::Array& context1, const ArrayProgram::IndexContainer& context2, ArrayProgram::Array& state1, ArrayProgram::UInt32& state2) {
            return __while_loop2_25.repeat(context1, context2, state1, state2);
        };
    };

    static ArrayProgram::_reverse_repeat reverse_repeat;
    struct _set {
        inline void operator()(ArrayProgram::PaddedArray& a, const ArrayProgram::Index& ix, const ArrayProgram::Int32& e) {
            return __array.set(a, ix, e);
        };
        inline void operator()(ArrayProgram::Array& a, const ArrayProgram::Index& ix, const ArrayProgram::Int32& e) {
            return __array.set(a, ix, e);
        };
        inline void operator()(ArrayProgram::Array& a, const ArrayProgram::UInt32& ix, const ArrayProgram::Int32& e) {
            return __array.set(a, ix, e);
        };
    };

    static ArrayProgram::_set set;
    struct _shape {
        inline ArrayProgram::Shape operator()(const ArrayProgram::PaddedArray& a) {
            return __array.shape(a);
        };
        inline ArrayProgram::Shape operator()(const ArrayProgram::Array& a) {
            return __array.shape(a);
        };
    };

    static ArrayProgram::_shape shape;
    struct _test_array3_2_2 {
        inline ArrayProgram::Array operator()() {
            return __array.test_array3_2_2();
        };
    };

    static ArrayProgram::_test_array3_2_2 test_array3_2_2;
    struct _test_array3_3 {
        inline ArrayProgram::Array operator()() {
            return __array.test_array3_3();
        };
    };

    static ArrayProgram::_test_array3_3 test_array3_3;
    struct _test_vector2 {
        inline ArrayProgram::Array operator()() {
            return __array.test_vector2();
        };
    };

    static ArrayProgram::_test_vector2 test_vector2;
    struct _test_vector3 {
        inline ArrayProgram::Array operator()() {
            return __array.test_vector3();
        };
    };

    static ArrayProgram::_test_vector3 test_vector3;
    struct _test_vector5 {
        inline ArrayProgram::Array operator()() {
            return __array.test_vector5();
        };
    };

    static ArrayProgram::_test_vector5 test_vector5;
    struct _total {
        inline ArrayProgram::UInt32 operator()(const ArrayProgram::IndexContainer& ixc) {
            return __array.total(ixc);
        };
        inline ArrayProgram::UInt32 operator()(const ArrayProgram::Shape& s) {
            return __array.total(s);
        };
        inline ArrayProgram::UInt32 operator()(const ArrayProgram::Array& a) {
            return __array.total(a);
        };
    };

    static ArrayProgram::_total total;
    struct _transpose {
        inline ArrayProgram::Array operator()(const ArrayProgram::Array& a) {
            ArrayProgram::Array transposed_array = ArrayProgram::create_array(ArrayProgram::reverse_shape(ArrayProgram::shape(a)));
            ArrayProgram::IndexContainer ix_space = ArrayProgram::create_valid_indices(transposed_array);
            ArrayProgram::UInt32 counter = ArrayProgram::elem_uint(ArrayProgram::zero());
            ArrayProgram::transpose_repeat(a, ix_space, transposed_array, counter);
            return transposed_array;
        };
        inline ArrayProgram::PaddedArray operator()(const ArrayProgram::PaddedArray& a) {
            ArrayProgram::Array reshaped_array = ArrayProgram::create_array(ArrayProgram::padded_shape(a));
            ArrayProgram::PaddedArray transposed_array = ArrayProgram::create_padded_array(ArrayProgram::reverse_shape(ArrayProgram::shape(a)), ArrayProgram::reverse_shape(ArrayProgram::padded_shape(a)), reshaped_array);
            ArrayProgram::IndexContainer ix_space = ArrayProgram::create_valid_indices(transposed_array);
            ArrayProgram::UInt32 counter = ArrayProgram::elem_uint(ArrayProgram::zero());
            ArrayProgram::padded_transpose_repeat(a, ix_space, transposed_array, counter);
            return transposed_array;
        };
    };

    static ArrayProgram::_transpose transpose;
    struct _transpose_body {
        inline void operator()(const ArrayProgram::Array& a, const ArrayProgram::IndexContainer& ixc, ArrayProgram::Array& res, ArrayProgram::UInt32& c) {
            ArrayProgram::Index current_ix = ArrayProgram::get_index_ixc(ixc, c);
            ArrayProgram::Int32 current_element = ArrayProgram::unwrap_scalar(ArrayProgram::get(a, ArrayProgram::reverse_index(current_ix)));
            ArrayProgram::set(res, current_ix, current_element);
            c = ArrayProgram::elem_uint(ArrayProgram::binary_add(ArrayProgram::uint_elem(c), ArrayProgram::one()));
        };
    };

    static ArrayProgram::_transpose_body transpose_body;
    struct _transpose_repeat {
        inline void operator()(const ArrayProgram::Array& context1, const ArrayProgram::IndexContainer& context2, ArrayProgram::Array& state1, ArrayProgram::UInt32& state2) {
            return __while_loop2_26.repeat(context1, context2, state1, state2);
        };
    };

    static ArrayProgram::_transpose_repeat transpose_repeat;
    struct _unaryMap {
        inline void operator()(const ArrayProgram::Array& a, const ArrayProgram::Index& ix) {
            assert((ArrayProgram::unwrap_scalar(ArrayProgram::get(ArrayProgram::unary_sub(a), ix))) == (ArrayProgram::unary_sub(ArrayProgram::unwrap_scalar(ArrayProgram::get(a, ix)))));
        };
    };

    static ArrayProgram::_unaryMap unaryMap;
    struct _unary_sub {
        inline ArrayProgram::Int32 operator()(const ArrayProgram::Int32& a) {
            return __int32_utils.unary_sub(a);
        };
        inline ArrayProgram::Array operator()(const ArrayProgram::Array& a) {
            ArrayProgram::IndexContainer ix_space = ArrayProgram::create_valid_indices(a);
            ArrayProgram::Array a_upd = a;
            ArrayProgram::UInt32 counter = ArrayProgram::elem_uint(ArrayProgram::zero());
            ArrayProgram::unary_sub_repeat(ix_space, a_upd, counter);
            return a_upd;
        };
    };

    static ArrayProgram::_unary_sub unary_sub;
    struct _unary_sub_body {
        inline void operator()(const ArrayProgram::IndexContainer& ix_space, ArrayProgram::Array& a, ArrayProgram::UInt32& c) {
            ArrayProgram::Index ix = ArrayProgram::get_index_ixc(ix_space, c);
            ArrayProgram::Int32 new_value = ArrayProgram::unary_sub(ArrayProgram::unwrap_scalar(ArrayProgram::get(a, ix)));
            ArrayProgram::set(a, ix, new_value);
            c = ArrayProgram::elem_uint(ArrayProgram::binary_add(ArrayProgram::uint_elem(c), ArrayProgram::one()));
        };
    };

    static ArrayProgram::_unary_sub_body unary_sub_body;
    struct _unary_sub_cond {
        inline bool operator()(const ArrayProgram::IndexContainer& ix_space, const ArrayProgram::Array& a, const ArrayProgram::UInt32& c) {
            return ArrayProgram::lt(ArrayProgram::uint_elem(c), ArrayProgram::uint_elem(ArrayProgram::total(ix_space)));
        };
    };

private:
    static while_loop1_2<ArrayProgram::IndexContainer, ArrayProgram::Array, ArrayProgram::UInt32, ArrayProgram::_unary_sub_body, ArrayProgram::_unary_sub_cond> __while_loop1_20;
public:
    static ArrayProgram::_unary_sub_cond unary_sub_cond;
    struct _unary_sub_repeat {
        inline void operator()(const ArrayProgram::IndexContainer& context1, ArrayProgram::Array& state1, ArrayProgram::UInt32& state2) {
            return __while_loop1_20.repeat(context1, state1, state2);
        };
    };

    static ArrayProgram::_unary_sub_repeat unary_sub_repeat;
    struct _unwrap_scalar {
        inline ArrayProgram::Int32 operator()(const ArrayProgram::Array& a) {
            return __array.unwrap_scalar(a);
        };
    };

    static ArrayProgram::_unwrap_scalar unwrap_scalar;
    struct _upper_bound {
        inline bool operator()(const ArrayProgram::Array& a, const ArrayProgram::IndexContainer& i, const ArrayProgram::Array& res, const ArrayProgram::UInt32& c) {
            return ArrayProgram::lt(ArrayProgram::uint_elem(c), ArrayProgram::uint_elem(ArrayProgram::total(i)));
        };
    };

private:
    static while_loop2_2<ArrayProgram::Array, ArrayProgram::IndexContainer, ArrayProgram::Array, ArrayProgram::UInt32, ArrayProgram::_transpose_body, ArrayProgram::_upper_bound> __while_loop2_26;
public:
    static ArrayProgram::_upper_bound upper_bound;
};
} // examples
} // moa
} // mg_src
} // moa_cpp

namespace examples {
namespace moa {
namespace mg_src {
namespace moa_cpp {
struct BurgerProgram {
private:
    static int32_utils __int32_utils;
public:
    typedef int32_utils::Int32 Int32;
    typedef array<BurgerProgram::Int32>::PaddedArray PaddedArray;
    struct _print_parray {
        inline void operator()(const BurgerProgram::PaddedArray& a) {
            return __array.print_parray(a);
        };
    };

    static BurgerProgram::_print_parray print_parray;
    typedef array<BurgerProgram::Int32>::Shape Shape;
    struct _cat_shape {
        inline BurgerProgram::Shape operator()(const BurgerProgram::Shape& a, const BurgerProgram::Shape& b) {
            return __array.cat_shape(a, b);
        };
    };

    static BurgerProgram::_cat_shape cat_shape;
    struct _padded_shape {
        inline BurgerProgram::Shape operator()(const BurgerProgram::PaddedArray& a) {
            return __array.padded_shape(a);
        };
    };

    static BurgerProgram::_padded_shape padded_shape;
    struct _print_shape {
        inline void operator()(const BurgerProgram::Shape& sh) {
            return __array.print_shape(sh);
        };
    };

    static BurgerProgram::_print_shape print_shape;
    struct _reverse_shape {
        inline BurgerProgram::Shape operator()(const BurgerProgram::Shape& s) {
            return __array.reverse_shape(s);
        };
    };

    static BurgerProgram::_reverse_shape reverse_shape;
    typedef array<BurgerProgram::Int32>::UInt32 UInt32;
    struct _create_shape1 {
        inline BurgerProgram::Shape operator()(const BurgerProgram::UInt32& a) {
            return __array.create_shape1(a);
        };
    };

    static BurgerProgram::_create_shape1 create_shape1;
    struct _create_shape2 {
        inline BurgerProgram::Shape operator()(const BurgerProgram::UInt32& a, const BurgerProgram::UInt32& b) {
            return __array.create_shape2(a, b);
        };
    };

    static BurgerProgram::_create_shape2 create_shape2;
    struct _create_shape3 {
        inline BurgerProgram::Shape operator()(const BurgerProgram::UInt32& a, const BurgerProgram::UInt32& b, const BurgerProgram::UInt32& c) {
            return __array.create_shape3(a, b, c);
        };
    };

    static BurgerProgram::_create_shape3 create_shape3;
    struct _padded_dim {
        inline BurgerProgram::UInt32 operator()(const BurgerProgram::PaddedArray& a) {
            return __array.padded_dim(a);
        };
    };

    static BurgerProgram::_padded_dim padded_dim;
    struct _padded_drop_shape_elem {
        inline BurgerProgram::Shape operator()(const BurgerProgram::PaddedArray& a, const BurgerProgram::UInt32& i) {
            return __array.padded_drop_shape_elem(a, i);
        };
    };

    static BurgerProgram::_padded_drop_shape_elem padded_drop_shape_elem;
    struct _padded_get_shape_elem {
        inline BurgerProgram::UInt32 operator()(const BurgerProgram::PaddedArray& a, const BurgerProgram::UInt32& i) {
            return __array.padded_get_shape_elem(a, i);
        };
    };

    static BurgerProgram::_padded_get_shape_elem padded_get_shape_elem;
    struct _padded_total {
        inline BurgerProgram::UInt32 operator()(const BurgerProgram::PaddedArray& a) {
            return __array.padded_total(a);
        };
    };

    static BurgerProgram::_padded_total padded_total;
    struct _print_uint {
        inline void operator()(const BurgerProgram::UInt32& u) {
            return __array.print_uint(u);
        };
    };

    static BurgerProgram::_print_uint print_uint;
private:
    static array<BurgerProgram::Int32> __array;
public:
    struct _elem_uint {
        inline BurgerProgram::UInt32 operator()(const BurgerProgram::Int32& a) {
            return __array.elem_uint(a);
        };
    };

    static BurgerProgram::_elem_uint elem_uint;
    struct _lt {
        inline bool operator()(const BurgerProgram::Int32& a, const BurgerProgram::Int32& b) {
            return __int32_utils.lt(a, b);
        };
    };

    static BurgerProgram::_lt lt;
    struct _one {
        inline BurgerProgram::Int32 operator()() {
            return __int32_utils.one();
        };
    };

    static BurgerProgram::_one one;
    struct _print_element {
        inline void operator()(const BurgerProgram::Int32& e) {
            return __array.print_element(e);
        };
    };

    static BurgerProgram::_print_element print_element;
    struct _uint_elem {
        inline BurgerProgram::Int32 operator()(const BurgerProgram::UInt32& a) {
            return __array.uint_elem(a);
        };
    };

    static BurgerProgram::_uint_elem uint_elem;
    struct _zero {
        inline BurgerProgram::Int32 operator()() {
            return __int32_utils.zero();
        };
    };

    static BurgerProgram::_zero zero;
    typedef array<BurgerProgram::Int32>::IndexContainer IndexContainer;
    struct _print_index_container {
        inline void operator()(const BurgerProgram::IndexContainer& i) {
            return __array.print_index_container(i);
        };
    };

    static BurgerProgram::_print_index_container print_index_container;
    typedef array<BurgerProgram::Int32>::Index Index;
    struct _cat_index {
        inline BurgerProgram::Index operator()(const BurgerProgram::Index& i, const BurgerProgram::Index& j) {
            return __array.cat_index(i, j);
        };
    };

    static BurgerProgram::_cat_index cat_index;
    struct _create_index1 {
        inline BurgerProgram::Index operator()(const BurgerProgram::UInt32& a) {
            return __array.create_index1(a);
        };
    };

    static BurgerProgram::_create_index1 create_index1;
    struct _create_index2 {
        inline BurgerProgram::Index operator()(const BurgerProgram::UInt32& a, const BurgerProgram::UInt32& b) {
            return __array.create_index2(a, b);
        };
    };

    static BurgerProgram::_create_index2 create_index2;
    struct _create_index3 {
        inline BurgerProgram::Index operator()(const BurgerProgram::UInt32& a, const BurgerProgram::UInt32& b, const BurgerProgram::UInt32& c) {
            return __array.create_index3(a, b, c);
        };
    };

    static BurgerProgram::_create_index3 create_index3;
    struct _drop_index_elem {
        inline BurgerProgram::Index operator()(const BurgerProgram::Index& ix, const BurgerProgram::UInt32& i) {
            return __array.drop_index_elem(ix, i);
        };
    };

    static BurgerProgram::_drop_index_elem drop_index_elem;
    struct _get_index_elem {
        inline BurgerProgram::UInt32 operator()(const BurgerProgram::Index& ix, const BurgerProgram::UInt32& i) {
            return __array.get_index_elem(ix, i);
        };
    };

    static BurgerProgram::_get_index_elem get_index_elem;
    struct _get_index_ixc {
        inline BurgerProgram::Index operator()(const BurgerProgram::IndexContainer& ixc, const BurgerProgram::UInt32& ix) {
            return __array.get_index_ixc(ixc, ix);
        };
    };

    static BurgerProgram::_get_index_ixc get_index_ixc;
    struct _print_index {
        inline void operator()(const BurgerProgram::Index& i) {
            return __array.print_index(i);
        };
    };

    static BurgerProgram::_print_index print_index;
    struct _reverse_index {
        inline BurgerProgram::Index operator()(const BurgerProgram::Index& ix) {
            return __array.reverse_index(ix);
        };
    };

    static BurgerProgram::_reverse_index reverse_index;
    struct _test_index {
        inline BurgerProgram::Index operator()() {
            return __array.test_index();
        };
    };

    static BurgerProgram::_test_index test_index;
    typedef array<BurgerProgram::Int32>::Array Array;
    struct _binaryMap {
        inline void operator()(const BurgerProgram::Array& a, const BurgerProgram::Array& b, const BurgerProgram::Index& ix) {
            assert((BurgerProgram::unwrap_scalar(BurgerProgram::get(BurgerProgram::binary_add(a, b), ix))) == (BurgerProgram::binary_add(BurgerProgram::unwrap_scalar(BurgerProgram::get(a, ix)), BurgerProgram::unwrap_scalar(BurgerProgram::get(b, ix)))));
            assert((BurgerProgram::unwrap_scalar(BurgerProgram::get(BurgerProgram::binary_sub(a, b), ix))) == (BurgerProgram::binary_sub(BurgerProgram::unwrap_scalar(BurgerProgram::get(a, ix)), BurgerProgram::unwrap_scalar(BurgerProgram::get(b, ix)))));
            assert((BurgerProgram::unwrap_scalar(BurgerProgram::get(BurgerProgram::mul(a, b), ix))) == (BurgerProgram::mul(BurgerProgram::unwrap_scalar(BurgerProgram::get(a, ix)), BurgerProgram::unwrap_scalar(BurgerProgram::get(b, ix)))));
            assert((BurgerProgram::unwrap_scalar(BurgerProgram::get(BurgerProgram::div(a, b), ix))) == (BurgerProgram::div(BurgerProgram::unwrap_scalar(BurgerProgram::get(a, ix)), BurgerProgram::unwrap_scalar(BurgerProgram::get(b, ix)))));
        };
    };

    static BurgerProgram::_binaryMap binaryMap;
    struct _binary_add {
        inline BurgerProgram::Int32 operator()(const BurgerProgram::Int32& a, const BurgerProgram::Int32& b) {
            return __int32_utils.binary_add(a, b);
        };
        inline BurgerProgram::Array operator()(const BurgerProgram::Array& a, const BurgerProgram::Array& b) {
            BurgerProgram::IndexContainer ix_space = BurgerProgram::create_valid_indices(a);
            BurgerProgram::Array b_upd = b;
            BurgerProgram::UInt32 counter = BurgerProgram::elem_uint(BurgerProgram::zero());
            BurgerProgram::bmb_plus_rep(a, ix_space, b_upd, counter);
            return b;
        };
    };

    static BurgerProgram::_binary_add binary_add;
    struct _binary_sub {
        inline BurgerProgram::Int32 operator()(const BurgerProgram::Int32& a, const BurgerProgram::Int32& b) {
            return __int32_utils.binary_sub(a, b);
        };
        inline BurgerProgram::Array operator()(const BurgerProgram::Array& a, const BurgerProgram::Array& b) {
            BurgerProgram::IndexContainer ix_space = BurgerProgram::create_valid_indices(a);
            BurgerProgram::Array b_upd = b;
            BurgerProgram::UInt32 counter = BurgerProgram::elem_uint(BurgerProgram::zero());
            BurgerProgram::bmb_sub_rep(a, ix_space, b_upd, counter);
            return b;
        };
    };

    static BurgerProgram::_binary_sub binary_sub;
    struct _bmb_div {
        inline void operator()(const BurgerProgram::Array& a, const BurgerProgram::IndexContainer& ix_space, BurgerProgram::Array& b, BurgerProgram::UInt32& c) {
            BurgerProgram::Index ix = BurgerProgram::get_index_ixc(ix_space, c);
            BurgerProgram::Int32 new_value = BurgerProgram::div(BurgerProgram::unwrap_scalar(BurgerProgram::get(a, ix)), BurgerProgram::unwrap_scalar(BurgerProgram::get(b, ix)));
            BurgerProgram::set(b, ix, new_value);
            c = BurgerProgram::elem_uint(BurgerProgram::binary_add(BurgerProgram::uint_elem(c), BurgerProgram::one()));
        };
    };

    static BurgerProgram::_bmb_div bmb_div;
    struct _bmb_div_rep {
        inline void operator()(const BurgerProgram::Array& context1, const BurgerProgram::IndexContainer& context2, BurgerProgram::Array& state1, BurgerProgram::UInt32& state2) {
            return __while_loop2_2.repeat(context1, context2, state1, state2);
        };
    };

    static BurgerProgram::_bmb_div_rep bmb_div_rep;
    struct _bmb_mul {
        inline void operator()(const BurgerProgram::Array& a, const BurgerProgram::IndexContainer& ix_space, BurgerProgram::Array& b, BurgerProgram::UInt32& c) {
            BurgerProgram::Index ix = BurgerProgram::get_index_ixc(ix_space, c);
            BurgerProgram::Int32 new_value = BurgerProgram::mul(BurgerProgram::unwrap_scalar(BurgerProgram::get(a, ix)), BurgerProgram::unwrap_scalar(BurgerProgram::get(b, ix)));
            BurgerProgram::set(b, ix, new_value);
            c = BurgerProgram::elem_uint(BurgerProgram::binary_add(BurgerProgram::uint_elem(c), BurgerProgram::one()));
        };
    };

    static BurgerProgram::_bmb_mul bmb_mul;
    struct _bmb_mul_rep {
        inline void operator()(const BurgerProgram::Array& context1, const BurgerProgram::IndexContainer& context2, BurgerProgram::Array& state1, BurgerProgram::UInt32& state2) {
            return __while_loop2_20.repeat(context1, context2, state1, state2);
        };
    };

    static BurgerProgram::_bmb_mul_rep bmb_mul_rep;
    struct _bmb_plus {
        inline void operator()(const BurgerProgram::Array& a, const BurgerProgram::IndexContainer& ix_space, BurgerProgram::Array& b, BurgerProgram::UInt32& c) {
            BurgerProgram::Index ix = BurgerProgram::get_index_ixc(ix_space, c);
            BurgerProgram::Int32 new_value = BurgerProgram::binary_add(BurgerProgram::unwrap_scalar(BurgerProgram::get(a, ix)), BurgerProgram::unwrap_scalar(BurgerProgram::get(b, ix)));
            BurgerProgram::set(b, ix, new_value);
            c = BurgerProgram::elem_uint(BurgerProgram::binary_add(BurgerProgram::uint_elem(c), BurgerProgram::one()));
        };
    };

    static BurgerProgram::_bmb_plus bmb_plus;
    struct _bmb_plus_rep {
        inline void operator()(const BurgerProgram::Array& context1, const BurgerProgram::IndexContainer& context2, BurgerProgram::Array& state1, BurgerProgram::UInt32& state2) {
            return __while_loop2_21.repeat(context1, context2, state1, state2);
        };
    };

    static BurgerProgram::_bmb_plus_rep bmb_plus_rep;
    struct _bmb_sub {
        inline void operator()(const BurgerProgram::Array& a, const BurgerProgram::IndexContainer& ix_space, BurgerProgram::Array& b, BurgerProgram::UInt32& c) {
            BurgerProgram::Index ix = BurgerProgram::get_index_ixc(ix_space, c);
            BurgerProgram::Int32 new_value = BurgerProgram::binary_sub(BurgerProgram::unwrap_scalar(BurgerProgram::get(a, ix)), BurgerProgram::unwrap_scalar(BurgerProgram::get(b, ix)));
            BurgerProgram::set(b, ix, new_value);
            c = BurgerProgram::elem_uint(BurgerProgram::binary_add(BurgerProgram::uint_elem(c), BurgerProgram::one()));
        };
    };

    static BurgerProgram::_bmb_sub bmb_sub;
    struct _bmb_sub_rep {
        inline void operator()(const BurgerProgram::Array& context1, const BurgerProgram::IndexContainer& context2, BurgerProgram::Array& state1, BurgerProgram::UInt32& state2) {
            return __while_loop2_22.repeat(context1, context2, state1, state2);
        };
    };

    static BurgerProgram::_bmb_sub_rep bmb_sub_rep;
    struct _create_array {
        inline BurgerProgram::Array operator()(const BurgerProgram::Shape& sh) {
            return __array.create_array(sh);
        };
    };

    static BurgerProgram::_create_array create_array;
    struct _create_padded_array {
        inline BurgerProgram::PaddedArray operator()(const BurgerProgram::Shape& unpadded_shape, const BurgerProgram::Shape& padded_shape, const BurgerProgram::Array& padded_array) {
            return __array.create_padded_array(unpadded_shape, padded_shape, padded_array);
        };
    };

    static BurgerProgram::_create_padded_array create_padded_array;
    struct _create_valid_indices {
        inline BurgerProgram::IndexContainer operator()(const BurgerProgram::PaddedArray& a) {
            return __array.create_valid_indices(a);
        };
        inline BurgerProgram::IndexContainer operator()(const BurgerProgram::Array& a) {
            return __array.create_valid_indices(a);
        };
    };

    static BurgerProgram::_create_valid_indices create_valid_indices;
    struct _dim {
        inline BurgerProgram::UInt32 operator()(const BurgerProgram::Array& a) {
            return __array.dim(a);
        };
    };

    static BurgerProgram::_dim dim;
    struct _div {
        inline BurgerProgram::Int32 operator()(const BurgerProgram::Int32& a, const BurgerProgram::Int32& b) {
            return __int32_utils.div(a, b);
        };
        inline BurgerProgram::Array operator()(const BurgerProgram::Array& a, const BurgerProgram::Array& b) {
            BurgerProgram::IndexContainer ix_space = BurgerProgram::create_valid_indices(a);
            BurgerProgram::Array b_upd = b;
            BurgerProgram::UInt32 counter = BurgerProgram::elem_uint(BurgerProgram::zero());
            BurgerProgram::bmb_div_rep(a, ix_space, b_upd, counter);
            return b;
        };
    };

    static BurgerProgram::_div div;
    struct _drop_shape_elem {
        inline BurgerProgram::Shape operator()(const BurgerProgram::Array& a, const BurgerProgram::UInt32& i) {
            return __array.drop_shape_elem(a, i);
        };
    };

    static BurgerProgram::_drop_shape_elem drop_shape_elem;
    struct _get {
        inline BurgerProgram::Array operator()(const BurgerProgram::PaddedArray& a, const BurgerProgram::Index& ix) {
            return __array.get(a, ix);
        };
        inline BurgerProgram::Array operator()(const BurgerProgram::Array& a, const BurgerProgram::Index& ix) {
            return __array.get(a, ix);
        };
        inline BurgerProgram::Array operator()(const BurgerProgram::Array& a, const BurgerProgram::UInt32& ix) {
            return __array.get(a, ix);
        };
    };

    static BurgerProgram::_get get;
    struct _get_shape_elem {
        inline BurgerProgram::UInt32 operator()(const BurgerProgram::Array& a, const BurgerProgram::UInt32& i) {
            return __array.get_shape_elem(a, i);
        };
    };

    static BurgerProgram::_get_shape_elem get_shape_elem;
    struct _mapped_ops_cond {
        inline bool operator()(const BurgerProgram::Array& a, const BurgerProgram::IndexContainer& ix_space, const BurgerProgram::Array& b, const BurgerProgram::UInt32& c) {
            return BurgerProgram::lt(BurgerProgram::uint_elem(c), BurgerProgram::uint_elem(BurgerProgram::total(ix_space)));
        };
    };

private:
    static while_loop2_2<BurgerProgram::Array, BurgerProgram::IndexContainer, BurgerProgram::Array, BurgerProgram::UInt32, BurgerProgram::_bmb_div, BurgerProgram::_mapped_ops_cond> __while_loop2_2;
    static while_loop2_2<BurgerProgram::Array, BurgerProgram::IndexContainer, BurgerProgram::Array, BurgerProgram::UInt32, BurgerProgram::_bmb_mul, BurgerProgram::_mapped_ops_cond> __while_loop2_20;
    static while_loop2_2<BurgerProgram::Array, BurgerProgram::IndexContainer, BurgerProgram::Array, BurgerProgram::UInt32, BurgerProgram::_bmb_plus, BurgerProgram::_mapped_ops_cond> __while_loop2_21;
    static while_loop2_2<BurgerProgram::Array, BurgerProgram::IndexContainer, BurgerProgram::Array, BurgerProgram::UInt32, BurgerProgram::_bmb_sub, BurgerProgram::_mapped_ops_cond> __while_loop2_22;
public:
    static BurgerProgram::_mapped_ops_cond mapped_ops_cond;
    struct _mul {
        inline BurgerProgram::Int32 operator()(const BurgerProgram::Int32& a, const BurgerProgram::Int32& b) {
            return __int32_utils.mul(a, b);
        };
        inline BurgerProgram::Array operator()(const BurgerProgram::Array& a, const BurgerProgram::Array& b) {
            BurgerProgram::IndexContainer ix_space = BurgerProgram::create_valid_indices(a);
            BurgerProgram::Array b_upd = b;
            BurgerProgram::UInt32 counter = BurgerProgram::elem_uint(BurgerProgram::zero());
            BurgerProgram::bmb_mul_rep(a, ix_space, b_upd, counter);
            return b;
        };
    };

    static BurgerProgram::_mul mul;
    struct _padded_to_unpadded {
        inline BurgerProgram::Array operator()(const BurgerProgram::PaddedArray& a) {
            return __array.padded_to_unpadded(a);
        };
    };

    static BurgerProgram::_padded_to_unpadded padded_to_unpadded;
    struct _print_array {
        inline void operator()(const BurgerProgram::Array& a) {
            return __array.print_array(a);
        };
    };

    static BurgerProgram::_print_array print_array;
    struct _set {
        inline void operator()(BurgerProgram::PaddedArray& a, const BurgerProgram::Index& ix, const BurgerProgram::Int32& e) {
            return __array.set(a, ix, e);
        };
        inline void operator()(BurgerProgram::Array& a, const BurgerProgram::Index& ix, const BurgerProgram::Int32& e) {
            return __array.set(a, ix, e);
        };
        inline void operator()(BurgerProgram::Array& a, const BurgerProgram::UInt32& ix, const BurgerProgram::Int32& e) {
            return __array.set(a, ix, e);
        };
    };

    static BurgerProgram::_set set;
    struct _shape {
        inline BurgerProgram::Shape operator()(const BurgerProgram::PaddedArray& a) {
            return __array.shape(a);
        };
        inline BurgerProgram::Shape operator()(const BurgerProgram::Array& a) {
            return __array.shape(a);
        };
    };

    static BurgerProgram::_shape shape;
    struct _test_array3_2_2 {
        inline BurgerProgram::Array operator()() {
            return __array.test_array3_2_2();
        };
    };

    static BurgerProgram::_test_array3_2_2 test_array3_2_2;
    struct _test_array3_3 {
        inline BurgerProgram::Array operator()() {
            return __array.test_array3_3();
        };
    };

    static BurgerProgram::_test_array3_3 test_array3_3;
    struct _test_vector2 {
        inline BurgerProgram::Array operator()() {
            return __array.test_vector2();
        };
    };

    static BurgerProgram::_test_vector2 test_vector2;
    struct _test_vector3 {
        inline BurgerProgram::Array operator()() {
            return __array.test_vector3();
        };
    };

    static BurgerProgram::_test_vector3 test_vector3;
    struct _test_vector5 {
        inline BurgerProgram::Array operator()() {
            return __array.test_vector5();
        };
    };

    static BurgerProgram::_test_vector5 test_vector5;
    struct _total {
        inline BurgerProgram::UInt32 operator()(const BurgerProgram::IndexContainer& ixc) {
            return __array.total(ixc);
        };
        inline BurgerProgram::UInt32 operator()(const BurgerProgram::Shape& s) {
            return __array.total(s);
        };
        inline BurgerProgram::UInt32 operator()(const BurgerProgram::Array& a) {
            return __array.total(a);
        };
    };

    static BurgerProgram::_total total;
    struct _unaryMap {
        inline void operator()(const BurgerProgram::Array& a, const BurgerProgram::Index& ix) {
            assert((BurgerProgram::unwrap_scalar(BurgerProgram::get(BurgerProgram::unary_sub(a), ix))) == (BurgerProgram::unary_sub(BurgerProgram::unwrap_scalar(BurgerProgram::get(a, ix)))));
        };
    };

    static BurgerProgram::_unaryMap unaryMap;
    struct _unary_sub {
        inline BurgerProgram::Int32 operator()(const BurgerProgram::Int32& a) {
            return __int32_utils.unary_sub(a);
        };
        inline BurgerProgram::Array operator()(const BurgerProgram::Array& a) {
            BurgerProgram::IndexContainer ix_space = BurgerProgram::create_valid_indices(a);
            BurgerProgram::Array a_upd = a;
            BurgerProgram::UInt32 counter = BurgerProgram::elem_uint(BurgerProgram::zero());
            BurgerProgram::unary_sub_repeat(ix_space, a_upd, counter);
            return a_upd;
        };
    };

    static BurgerProgram::_unary_sub unary_sub;
    struct _unary_sub_body {
        inline void operator()(const BurgerProgram::IndexContainer& ix_space, BurgerProgram::Array& a, BurgerProgram::UInt32& c) {
            BurgerProgram::Index ix = BurgerProgram::get_index_ixc(ix_space, c);
            BurgerProgram::Int32 new_value = BurgerProgram::unary_sub(BurgerProgram::unwrap_scalar(BurgerProgram::get(a, ix)));
            BurgerProgram::set(a, ix, new_value);
            c = BurgerProgram::elem_uint(BurgerProgram::binary_add(BurgerProgram::uint_elem(c), BurgerProgram::one()));
        };
    };

    static BurgerProgram::_unary_sub_body unary_sub_body;
    struct _unary_sub_cond {
        inline bool operator()(const BurgerProgram::IndexContainer& ix_space, const BurgerProgram::Array& a, const BurgerProgram::UInt32& c) {
            return BurgerProgram::lt(BurgerProgram::uint_elem(c), BurgerProgram::uint_elem(BurgerProgram::total(ix_space)));
        };
    };

private:
    static while_loop1_2<BurgerProgram::IndexContainer, BurgerProgram::Array, BurgerProgram::UInt32, BurgerProgram::_unary_sub_body, BurgerProgram::_unary_sub_cond> __while_loop1_2;
public:
    static BurgerProgram::_unary_sub_cond unary_sub_cond;
    struct _unary_sub_repeat {
        inline void operator()(const BurgerProgram::IndexContainer& context1, BurgerProgram::Array& state1, BurgerProgram::UInt32& state2) {
            return __while_loop1_2.repeat(context1, state1, state2);
        };
    };

    static BurgerProgram::_unary_sub_repeat unary_sub_repeat;
    struct _unwrap_scalar {
        inline BurgerProgram::Int32 operator()(const BurgerProgram::Array& a) {
            return __array.unwrap_scalar(a);
        };
    };

    static BurgerProgram::_unwrap_scalar unwrap_scalar;
};
} // examples
} // moa
} // mg_src
} // moa_cpp

namespace examples {
namespace moa {
namespace mg_src {
namespace moa_cpp {
struct TestingSuite {
private:
    static int32_utils __int32_utils;
public:
    typedef int32_utils::Int32 Int32;
    typedef array<TestingSuite::Int32>::PaddedArray PaddedArray;
    struct _print_parray {
        inline void operator()(const TestingSuite::PaddedArray& a) {
            return __array.print_parray(a);
        };
    };

    static TestingSuite::_print_parray print_parray;
    typedef array<TestingSuite::Int32>::Shape Shape;
    struct _cat_shape {
        inline TestingSuite::Shape operator()(const TestingSuite::Shape& a, const TestingSuite::Shape& b) {
            return __array.cat_shape(a, b);
        };
    };

    static TestingSuite::_cat_shape cat_shape;
    struct _padded_shape {
        inline TestingSuite::Shape operator()(const TestingSuite::PaddedArray& a) {
            return __array.padded_shape(a);
        };
    };

    static TestingSuite::_padded_shape padded_shape;
    struct _print_shape {
        inline void operator()(const TestingSuite::Shape& sh) {
            return __array.print_shape(sh);
        };
    };

    static TestingSuite::_print_shape print_shape;
    struct _reverse_shape {
        inline TestingSuite::Shape operator()(const TestingSuite::Shape& s) {
            return __array.reverse_shape(s);
        };
    };

    static TestingSuite::_reverse_shape reverse_shape;
    typedef array<TestingSuite::Int32>::UInt32 UInt32;
    struct _create_shape1 {
        inline TestingSuite::Shape operator()(const TestingSuite::UInt32& a) {
            return __array.create_shape1(a);
        };
    };

    static TestingSuite::_create_shape1 create_shape1;
    struct _create_shape2 {
        inline TestingSuite::Shape operator()(const TestingSuite::UInt32& a, const TestingSuite::UInt32& b) {
            return __array.create_shape2(a, b);
        };
    };

    static TestingSuite::_create_shape2 create_shape2;
    struct _create_shape3 {
        inline TestingSuite::Shape operator()(const TestingSuite::UInt32& a, const TestingSuite::UInt32& b, const TestingSuite::UInt32& c) {
            return __array.create_shape3(a, b, c);
        };
    };

    static TestingSuite::_create_shape3 create_shape3;
    struct _padded_dim {
        inline TestingSuite::UInt32 operator()(const TestingSuite::PaddedArray& a) {
            return __array.padded_dim(a);
        };
    };

    static TestingSuite::_padded_dim padded_dim;
    struct _padded_drop_shape_elem {
        inline TestingSuite::Shape operator()(const TestingSuite::PaddedArray& a, const TestingSuite::UInt32& i) {
            return __array.padded_drop_shape_elem(a, i);
        };
    };

    static TestingSuite::_padded_drop_shape_elem padded_drop_shape_elem;
    struct _padded_get_shape_elem {
        inline TestingSuite::UInt32 operator()(const TestingSuite::PaddedArray& a, const TestingSuite::UInt32& i) {
            return __array.padded_get_shape_elem(a, i);
        };
    };

    static TestingSuite::_padded_get_shape_elem padded_get_shape_elem;
    struct _padded_total {
        inline TestingSuite::UInt32 operator()(const TestingSuite::PaddedArray& a) {
            return __array.padded_total(a);
        };
    };

    static TestingSuite::_padded_total padded_total;
    struct _print_uint {
        inline void operator()(const TestingSuite::UInt32& u) {
            return __array.print_uint(u);
        };
    };

    static TestingSuite::_print_uint print_uint;
private:
    static array<TestingSuite::Int32> __array;
public:
    struct _elem_uint {
        inline TestingSuite::UInt32 operator()(const TestingSuite::Int32& a) {
            return __array.elem_uint(a);
        };
    };

    static TestingSuite::_elem_uint elem_uint;
    struct _lt {
        inline bool operator()(const TestingSuite::Int32& a, const TestingSuite::Int32& b) {
            return __int32_utils.lt(a, b);
        };
    };

    static TestingSuite::_lt lt;
    struct _one {
        inline TestingSuite::Int32 operator()() {
            return __int32_utils.one();
        };
    };

    static TestingSuite::_one one;
    struct _print_element {
        inline void operator()(const TestingSuite::Int32& e) {
            return __array.print_element(e);
        };
    };

    static TestingSuite::_print_element print_element;
    struct _uint_elem {
        inline TestingSuite::Int32 operator()(const TestingSuite::UInt32& a) {
            return __array.uint_elem(a);
        };
    };

    static TestingSuite::_uint_elem uint_elem;
    struct _zero {
        inline TestingSuite::Int32 operator()() {
            return __int32_utils.zero();
        };
    };

    static TestingSuite::_zero zero;
    typedef array<TestingSuite::Int32>::IndexContainer IndexContainer;
    struct _print_index_container {
        inline void operator()(const TestingSuite::IndexContainer& i) {
            return __array.print_index_container(i);
        };
    };

    static TestingSuite::_print_index_container print_index_container;
    typedef array<TestingSuite::Int32>::Index Index;
    struct _cat_index {
        inline TestingSuite::Index operator()(const TestingSuite::Index& i, const TestingSuite::Index& j) {
            return __array.cat_index(i, j);
        };
    };

    static TestingSuite::_cat_index cat_index;
    struct _create_index1 {
        inline TestingSuite::Index operator()(const TestingSuite::UInt32& a) {
            return __array.create_index1(a);
        };
    };

    static TestingSuite::_create_index1 create_index1;
    struct _create_index2 {
        inline TestingSuite::Index operator()(const TestingSuite::UInt32& a, const TestingSuite::UInt32& b) {
            return __array.create_index2(a, b);
        };
    };

    static TestingSuite::_create_index2 create_index2;
    struct _create_index3 {
        inline TestingSuite::Index operator()(const TestingSuite::UInt32& a, const TestingSuite::UInt32& b, const TestingSuite::UInt32& c) {
            return __array.create_index3(a, b, c);
        };
    };

    static TestingSuite::_create_index3 create_index3;
    struct _drop_index_elem {
        inline TestingSuite::Index operator()(const TestingSuite::Index& ix, const TestingSuite::UInt32& i) {
            return __array.drop_index_elem(ix, i);
        };
    };

    static TestingSuite::_drop_index_elem drop_index_elem;
    struct _get_index_elem {
        inline TestingSuite::UInt32 operator()(const TestingSuite::Index& ix, const TestingSuite::UInt32& i) {
            return __array.get_index_elem(ix, i);
        };
    };

    static TestingSuite::_get_index_elem get_index_elem;
    struct _get_index_ixc {
        inline TestingSuite::Index operator()(const TestingSuite::IndexContainer& ixc, const TestingSuite::UInt32& ix) {
            return __array.get_index_ixc(ixc, ix);
        };
    };

    static TestingSuite::_get_index_ixc get_index_ixc;
    struct _print_index {
        inline void operator()(const TestingSuite::Index& i) {
            return __array.print_index(i);
        };
    };

    static TestingSuite::_print_index print_index;
    struct _reverse_index {
        inline TestingSuite::Index operator()(const TestingSuite::Index& ix) {
            return __array.reverse_index(ix);
        };
    };

    static TestingSuite::_reverse_index reverse_index;
    struct _test_index {
        inline TestingSuite::Index operator()() {
            return __array.test_index();
        };
    };

    static TestingSuite::_test_index test_index;
    typedef array<TestingSuite::Int32>::Array Array;
    struct _binaryMap {
        inline void operator()(const TestingSuite::Array& a, const TestingSuite::Array& b, const TestingSuite::Index& ix) {
            assert((TestingSuite::unwrap_scalar(TestingSuite::get(TestingSuite::binary_add(a, b), ix))) == (TestingSuite::binary_add(TestingSuite::unwrap_scalar(TestingSuite::get(a, ix)), TestingSuite::unwrap_scalar(TestingSuite::get(b, ix)))));
            assert((TestingSuite::unwrap_scalar(TestingSuite::get(TestingSuite::binary_sub(a, b), ix))) == (TestingSuite::binary_sub(TestingSuite::unwrap_scalar(TestingSuite::get(a, ix)), TestingSuite::unwrap_scalar(TestingSuite::get(b, ix)))));
            assert((TestingSuite::unwrap_scalar(TestingSuite::get(TestingSuite::mul(a, b), ix))) == (TestingSuite::mul(TestingSuite::unwrap_scalar(TestingSuite::get(a, ix)), TestingSuite::unwrap_scalar(TestingSuite::get(b, ix)))));
            assert((TestingSuite::unwrap_scalar(TestingSuite::get(TestingSuite::div(a, b), ix))) == (TestingSuite::div(TestingSuite::unwrap_scalar(TestingSuite::get(a, ix)), TestingSuite::unwrap_scalar(TestingSuite::get(b, ix)))));
        };
    };

    static TestingSuite::_binaryMap binaryMap;
    struct _binary_add {
        inline TestingSuite::Array operator()(const TestingSuite::Array& a, const TestingSuite::Array& b) {
            TestingSuite::IndexContainer ix_space = TestingSuite::create_valid_indices(a);
            TestingSuite::Array b_upd = b;
            TestingSuite::UInt32 counter = TestingSuite::elem_uint(TestingSuite::zero());
            TestingSuite::bmb_plus_rep(a, ix_space, b_upd, counter);
            return b;
        };
        inline TestingSuite::Int32 operator()(const TestingSuite::Int32& a, const TestingSuite::Int32& b) {
            return __int32_utils.binary_add(a, b);
        };
    };

    static TestingSuite::_binary_add binary_add;
    struct _binary_sub {
        inline TestingSuite::Array operator()(const TestingSuite::Array& a, const TestingSuite::Array& b) {
            TestingSuite::IndexContainer ix_space = TestingSuite::create_valid_indices(a);
            TestingSuite::Array b_upd = b;
            TestingSuite::UInt32 counter = TestingSuite::elem_uint(TestingSuite::zero());
            TestingSuite::bmb_sub_rep(a, ix_space, b_upd, counter);
            return b;
        };
        inline TestingSuite::Int32 operator()(const TestingSuite::Int32& a, const TestingSuite::Int32& b) {
            return __int32_utils.binary_sub(a, b);
        };
    };

    static TestingSuite::_binary_sub binary_sub;
    struct _bmb_div {
        inline void operator()(const TestingSuite::Array& a, const TestingSuite::IndexContainer& ix_space, TestingSuite::Array& b, TestingSuite::UInt32& c) {
            TestingSuite::Index ix = TestingSuite::get_index_ixc(ix_space, c);
            TestingSuite::Int32 new_value = TestingSuite::div(TestingSuite::unwrap_scalar(TestingSuite::get(a, ix)), TestingSuite::unwrap_scalar(TestingSuite::get(b, ix)));
            TestingSuite::set(b, ix, new_value);
            c = TestingSuite::elem_uint(TestingSuite::binary_add(TestingSuite::uint_elem(c), TestingSuite::one()));
        };
    };

    static TestingSuite::_bmb_div bmb_div;
    struct _bmb_div_rep {
        inline void operator()(const TestingSuite::Array& context1, const TestingSuite::IndexContainer& context2, TestingSuite::Array& state1, TestingSuite::UInt32& state2) {
            return __while_loop2_2.repeat(context1, context2, state1, state2);
        };
    };

    static TestingSuite::_bmb_div_rep bmb_div_rep;
    struct _bmb_mul {
        inline void operator()(const TestingSuite::Array& a, const TestingSuite::IndexContainer& ix_space, TestingSuite::Array& b, TestingSuite::UInt32& c) {
            TestingSuite::Index ix = TestingSuite::get_index_ixc(ix_space, c);
            TestingSuite::Int32 new_value = TestingSuite::mul(TestingSuite::unwrap_scalar(TestingSuite::get(a, ix)), TestingSuite::unwrap_scalar(TestingSuite::get(b, ix)));
            TestingSuite::set(b, ix, new_value);
            c = TestingSuite::elem_uint(TestingSuite::binary_add(TestingSuite::uint_elem(c), TestingSuite::one()));
        };
    };

    static TestingSuite::_bmb_mul bmb_mul;
    struct _bmb_mul_rep {
        inline void operator()(const TestingSuite::Array& context1, const TestingSuite::IndexContainer& context2, TestingSuite::Array& state1, TestingSuite::UInt32& state2) {
            return __while_loop2_20.repeat(context1, context2, state1, state2);
        };
    };

    static TestingSuite::_bmb_mul_rep bmb_mul_rep;
    struct _bmb_plus {
        inline void operator()(const TestingSuite::Array& a, const TestingSuite::IndexContainer& ix_space, TestingSuite::Array& b, TestingSuite::UInt32& c) {
            TestingSuite::Index ix = TestingSuite::get_index_ixc(ix_space, c);
            TestingSuite::Int32 new_value = TestingSuite::binary_add(TestingSuite::unwrap_scalar(TestingSuite::get(a, ix)), TestingSuite::unwrap_scalar(TestingSuite::get(b, ix)));
            TestingSuite::set(b, ix, new_value);
            c = TestingSuite::elem_uint(TestingSuite::binary_add(TestingSuite::uint_elem(c), TestingSuite::one()));
        };
    };

    static TestingSuite::_bmb_plus bmb_plus;
    struct _bmb_plus_rep {
        inline void operator()(const TestingSuite::Array& context1, const TestingSuite::IndexContainer& context2, TestingSuite::Array& state1, TestingSuite::UInt32& state2) {
            return __while_loop2_21.repeat(context1, context2, state1, state2);
        };
    };

    static TestingSuite::_bmb_plus_rep bmb_plus_rep;
    struct _bmb_sub {
        inline void operator()(const TestingSuite::Array& a, const TestingSuite::IndexContainer& ix_space, TestingSuite::Array& b, TestingSuite::UInt32& c) {
            TestingSuite::Index ix = TestingSuite::get_index_ixc(ix_space, c);
            TestingSuite::Int32 new_value = TestingSuite::binary_sub(TestingSuite::unwrap_scalar(TestingSuite::get(a, ix)), TestingSuite::unwrap_scalar(TestingSuite::get(b, ix)));
            TestingSuite::set(b, ix, new_value);
            c = TestingSuite::elem_uint(TestingSuite::binary_add(TestingSuite::uint_elem(c), TestingSuite::one()));
        };
    };

    static TestingSuite::_bmb_sub bmb_sub;
    struct _bmb_sub_rep {
        inline void operator()(const TestingSuite::Array& context1, const TestingSuite::IndexContainer& context2, TestingSuite::Array& state1, TestingSuite::UInt32& state2) {
            return __while_loop2_22.repeat(context1, context2, state1, state2);
        };
    };

    static TestingSuite::_bmb_sub_rep bmb_sub_rep;
    struct _create_array {
        inline TestingSuite::Array operator()(const TestingSuite::Shape& sh) {
            return __array.create_array(sh);
        };
    };

    static TestingSuite::_create_array create_array;
    struct _create_padded_array {
        inline TestingSuite::PaddedArray operator()(const TestingSuite::Shape& unpadded_shape, const TestingSuite::Shape& padded_shape, const TestingSuite::Array& padded_array) {
            return __array.create_padded_array(unpadded_shape, padded_shape, padded_array);
        };
    };

    static TestingSuite::_create_padded_array create_padded_array;
    struct _create_valid_indices {
        inline TestingSuite::IndexContainer operator()(const TestingSuite::Array& a) {
            return __array.create_valid_indices(a);
        };
        inline TestingSuite::IndexContainer operator()(const TestingSuite::PaddedArray& a) {
            return __array.create_valid_indices(a);
        };
    };

    static TestingSuite::_create_valid_indices create_valid_indices;
    struct _dim {
        inline TestingSuite::UInt32 operator()(const TestingSuite::Array& a) {
            return __array.dim(a);
        };
    };

    static TestingSuite::_dim dim;
    struct _div {
        inline TestingSuite::Array operator()(const TestingSuite::Array& a, const TestingSuite::Array& b) {
            TestingSuite::IndexContainer ix_space = TestingSuite::create_valid_indices(a);
            TestingSuite::Array b_upd = b;
            TestingSuite::UInt32 counter = TestingSuite::elem_uint(TestingSuite::zero());
            TestingSuite::bmb_div_rep(a, ix_space, b_upd, counter);
            return b;
        };
        inline TestingSuite::Int32 operator()(const TestingSuite::Int32& a, const TestingSuite::Int32& b) {
            return __int32_utils.div(a, b);
        };
    };

    static TestingSuite::_div div;
    struct _drop_shape_elem {
        inline TestingSuite::Shape operator()(const TestingSuite::Array& a, const TestingSuite::UInt32& i) {
            return __array.drop_shape_elem(a, i);
        };
    };

    static TestingSuite::_drop_shape_elem drop_shape_elem;
    struct _get {
        inline TestingSuite::Array operator()(const TestingSuite::Array& a, const TestingSuite::UInt32& ix) {
            return __array.get(a, ix);
        };
        inline TestingSuite::Array operator()(const TestingSuite::Array& a, const TestingSuite::Index& ix) {
            return __array.get(a, ix);
        };
        inline TestingSuite::Array operator()(const TestingSuite::PaddedArray& a, const TestingSuite::Index& ix) {
            return __array.get(a, ix);
        };
    };

    static TestingSuite::_get get;
    struct _get_shape_elem {
        inline TestingSuite::UInt32 operator()(const TestingSuite::Array& a, const TestingSuite::UInt32& i) {
            return __array.get_shape_elem(a, i);
        };
    };

    static TestingSuite::_get_shape_elem get_shape_elem;
    struct _mapped_ops_cond {
        inline bool operator()(const TestingSuite::Array& a, const TestingSuite::IndexContainer& ix_space, const TestingSuite::Array& b, const TestingSuite::UInt32& c) {
            return TestingSuite::lt(TestingSuite::uint_elem(c), TestingSuite::uint_elem(TestingSuite::total(ix_space)));
        };
    };

private:
    static while_loop2_2<TestingSuite::Array, TestingSuite::IndexContainer, TestingSuite::Array, TestingSuite::UInt32, TestingSuite::_bmb_div, TestingSuite::_mapped_ops_cond> __while_loop2_2;
    static while_loop2_2<TestingSuite::Array, TestingSuite::IndexContainer, TestingSuite::Array, TestingSuite::UInt32, TestingSuite::_bmb_mul, TestingSuite::_mapped_ops_cond> __while_loop2_20;
    static while_loop2_2<TestingSuite::Array, TestingSuite::IndexContainer, TestingSuite::Array, TestingSuite::UInt32, TestingSuite::_bmb_plus, TestingSuite::_mapped_ops_cond> __while_loop2_21;
    static while_loop2_2<TestingSuite::Array, TestingSuite::IndexContainer, TestingSuite::Array, TestingSuite::UInt32, TestingSuite::_bmb_sub, TestingSuite::_mapped_ops_cond> __while_loop2_22;
public:
    static TestingSuite::_mapped_ops_cond mapped_ops_cond;
    struct _mul {
        inline TestingSuite::Array operator()(const TestingSuite::Array& a, const TestingSuite::Array& b) {
            TestingSuite::IndexContainer ix_space = TestingSuite::create_valid_indices(a);
            TestingSuite::Array b_upd = b;
            TestingSuite::UInt32 counter = TestingSuite::elem_uint(TestingSuite::zero());
            TestingSuite::bmb_mul_rep(a, ix_space, b_upd, counter);
            return b;
        };
        inline TestingSuite::Int32 operator()(const TestingSuite::Int32& a, const TestingSuite::Int32& b) {
            return __int32_utils.mul(a, b);
        };
    };

    static TestingSuite::_mul mul;
    struct _padded_to_unpadded {
        inline TestingSuite::Array operator()(const TestingSuite::PaddedArray& a) {
            return __array.padded_to_unpadded(a);
        };
    };

    static TestingSuite::_padded_to_unpadded padded_to_unpadded;
    struct _print_array {
        inline void operator()(const TestingSuite::Array& a) {
            return __array.print_array(a);
        };
    };

    static TestingSuite::_print_array print_array;
    struct _rav_repeat {
        inline void operator()(const TestingSuite::Array& context1, TestingSuite::Array& state1, TestingSuite::UInt32& state2) {
            return __while_loop1_2.repeat(context1, state1, state2);
        };
    };

    static TestingSuite::_rav_repeat rav_repeat;
    struct _ravel {
        inline TestingSuite::Array operator()(const TestingSuite::Array& a) {
            TestingSuite::UInt32 c = TestingSuite::elem_uint(TestingSuite::zero());
            TestingSuite::Array flat = TestingSuite::create_array(TestingSuite::create_shape1(TestingSuite::total(a)));
            TestingSuite::rav_repeat(a, flat, c);
            return flat;
        };
    };

    static TestingSuite::_ravel ravel;
    struct _ravel_body {
        inline void operator()(const TestingSuite::Array& input, TestingSuite::Array& flat, TestingSuite::UInt32& c) {
            TestingSuite::Index linear_ix = TestingSuite::create_index1(c);
            TestingSuite::set(flat, linear_ix, TestingSuite::unwrap_scalar(TestingSuite::get(input, c)));
            c = TestingSuite::elem_uint(TestingSuite::binary_add(TestingSuite::uint_elem(c), TestingSuite::one()));
        };
    };

    static TestingSuite::_ravel_body ravel_body;
    struct _ravel_cond {
        inline bool operator()(const TestingSuite::Array& input, const TestingSuite::Array& flat, const TestingSuite::UInt32& c) {
            return TestingSuite::lt(TestingSuite::uint_elem(c), TestingSuite::uint_elem(TestingSuite::total(input)));
        };
    };

private:
    static while_loop1_2<TestingSuite::Array, TestingSuite::Array, TestingSuite::UInt32, TestingSuite::_ravel_body, TestingSuite::_ravel_cond> __while_loop1_2;
public:
    static TestingSuite::_ravel_cond ravel_cond;
    struct _set {
        inline void operator()(TestingSuite::Array& a, const TestingSuite::UInt32& ix, const TestingSuite::Int32& e) {
            return __array.set(a, ix, e);
        };
        inline void operator()(TestingSuite::Array& a, const TestingSuite::Index& ix, const TestingSuite::Int32& e) {
            return __array.set(a, ix, e);
        };
        inline void operator()(TestingSuite::PaddedArray& a, const TestingSuite::Index& ix, const TestingSuite::Int32& e) {
            return __array.set(a, ix, e);
        };
    };

    static TestingSuite::_set set;
    struct _shape {
        inline TestingSuite::Shape operator()(const TestingSuite::Array& a) {
            return __array.shape(a);
        };
        inline TestingSuite::Shape operator()(const TestingSuite::PaddedArray& a) {
            return __array.shape(a);
        };
    };

    static TestingSuite::_shape shape;
    struct _test_array3_2_2 {
        inline TestingSuite::Array operator()() {
            return __array.test_array3_2_2();
        };
    };

    static TestingSuite::_test_array3_2_2 test_array3_2_2;
    struct _test_array3_3 {
        inline TestingSuite::Array operator()() {
            return __array.test_array3_3();
        };
    };

    static TestingSuite::_test_array3_3 test_array3_3;
    struct _test_vector2 {
        inline TestingSuite::Array operator()() {
            return __array.test_vector2();
        };
    };

    static TestingSuite::_test_vector2 test_vector2;
    struct _test_vector3 {
        inline TestingSuite::Array operator()() {
            return __array.test_vector3();
        };
    };

    static TestingSuite::_test_vector3 test_vector3;
    struct _test_vector5 {
        inline TestingSuite::Array operator()() {
            return __array.test_vector5();
        };
    };

    static TestingSuite::_test_vector5 test_vector5;
    struct _total {
        inline TestingSuite::UInt32 operator()(const TestingSuite::Array& a) {
            return __array.total(a);
        };
        inline TestingSuite::UInt32 operator()(const TestingSuite::Shape& s) {
            return __array.total(s);
        };
        inline TestingSuite::UInt32 operator()(const TestingSuite::IndexContainer& ixc) {
            return __array.total(ixc);
        };
    };

    static TestingSuite::_total total;
    struct _unaryMap {
        inline void operator()(const TestingSuite::Array& a, const TestingSuite::Index& ix) {
            assert((TestingSuite::unwrap_scalar(TestingSuite::get(TestingSuite::unary_sub(a), ix))) == (TestingSuite::unary_sub(TestingSuite::unwrap_scalar(TestingSuite::get(a, ix)))));
        };
    };

    static TestingSuite::_unaryMap unaryMap;
    struct _unary_sub {
        inline TestingSuite::Array operator()(const TestingSuite::Array& a) {
            TestingSuite::IndexContainer ix_space = TestingSuite::create_valid_indices(a);
            TestingSuite::Array a_upd = a;
            TestingSuite::UInt32 counter = TestingSuite::elem_uint(TestingSuite::zero());
            TestingSuite::unary_sub_repeat(ix_space, a_upd, counter);
            return a_upd;
        };
        inline TestingSuite::Int32 operator()(const TestingSuite::Int32& a) {
            return __int32_utils.unary_sub(a);
        };
    };

    static TestingSuite::_unary_sub unary_sub;
    struct _unary_sub_body {
        inline void operator()(const TestingSuite::IndexContainer& ix_space, TestingSuite::Array& a, TestingSuite::UInt32& c) {
            TestingSuite::Index ix = TestingSuite::get_index_ixc(ix_space, c);
            TestingSuite::Int32 new_value = TestingSuite::unary_sub(TestingSuite::unwrap_scalar(TestingSuite::get(a, ix)));
            TestingSuite::set(a, ix, new_value);
            c = TestingSuite::elem_uint(TestingSuite::binary_add(TestingSuite::uint_elem(c), TestingSuite::one()));
        };
    };

    static TestingSuite::_unary_sub_body unary_sub_body;
    struct _unary_sub_cond {
        inline bool operator()(const TestingSuite::IndexContainer& ix_space, const TestingSuite::Array& a, const TestingSuite::UInt32& c) {
            return TestingSuite::lt(TestingSuite::uint_elem(c), TestingSuite::uint_elem(TestingSuite::total(ix_space)));
        };
    };

private:
    static while_loop1_2<TestingSuite::IndexContainer, TestingSuite::Array, TestingSuite::UInt32, TestingSuite::_unary_sub_body, TestingSuite::_unary_sub_cond> __while_loop1_20;
public:
    static TestingSuite::_unary_sub_cond unary_sub_cond;
    struct _unary_sub_repeat {
        inline void operator()(const TestingSuite::IndexContainer& context1, TestingSuite::Array& state1, TestingSuite::UInt32& state2) {
            return __while_loop1_20.repeat(context1, state1, state2);
        };
    };

    static TestingSuite::_unary_sub_repeat unary_sub_repeat;
    struct _unwrap_scalar {
        inline TestingSuite::Int32 operator()(const TestingSuite::Array& a) {
            return __array.unwrap_scalar(a);
        };
    };

    static TestingSuite::_unwrap_scalar unwrap_scalar;
};
} // examples
} // moa
} // mg_src
} // moa_cpp