#pragma once

#include "base.hpp"
#include <cassert>


namespace examples {
namespace moa {
namespace mg_src {
namespace moa_cpp {
struct ArrayProgram {
    struct _zero {
        template <typename T>
        inline T operator()() {
            T o;
            ArrayProgram::zero0(o);
            return o;
        };
    };

    static ArrayProgram::_zero zero;
    struct _one {
        template <typename T>
        inline T operator()() {
            T o;
            ArrayProgram::one0(o);
            return o;
        };
    };

    static ArrayProgram::_one one;
private:
    static int32_utils __int32_utils;
    static float64_utils __float64_utils;
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
private:
    static inline void one0(ArrayProgram::Int32& o) {
        o = __int32_utils.one();
    };
    static inline void zero0(ArrayProgram::Int32& o) {
        o = __int32_utils.zero();
    };
public:
    typedef array<ArrayProgram::Int32>::IndexContainer IndexContainer;
    struct _padded_transpose_body {
        inline void operator()(const ArrayProgram::PaddedArray& a, const ArrayProgram::IndexContainer& ixc, ArrayProgram::PaddedArray& res, ArrayProgram::UInt32& c) {
            ArrayProgram::Index current_ix = ArrayProgram::get_index_ixc(ixc, c);
            ArrayProgram::Int32 current_element = ArrayProgram::unwrap_scalar(ArrayProgram::get(a, ArrayProgram::reverse_index(current_ix)));
            ArrayProgram::set(res, current_ix, current_element);
            c = ArrayProgram::elem_uint(ArrayProgram::binary_add(ArrayProgram::uint_elem(c), ArrayProgram::one.operator()<Int32>()));
        };
    };

    static ArrayProgram::_padded_transpose_body padded_transpose_body;
    struct _padded_transpose_repeat {
        inline void operator()(const ArrayProgram::PaddedArray& context1, const ArrayProgram::IndexContainer& context2, ArrayProgram::PaddedArray& state1, ArrayProgram::UInt32& state2) {
            return __while_loop2_23.repeat(context1, context2, state1, state2);
        };
    };

    static ArrayProgram::_padded_transpose_repeat padded_transpose_repeat;
    struct _padded_upper_bound {
        inline bool operator()(const ArrayProgram::PaddedArray& a, const ArrayProgram::IndexContainer& i, const ArrayProgram::PaddedArray& res, const ArrayProgram::UInt32& c) {
            return ArrayProgram::lt(ArrayProgram::uint_elem(c), ArrayProgram::uint_elem(ArrayProgram::total(i)));
        };
    };

private:
    static while_loop2_2<ArrayProgram::PaddedArray, ArrayProgram::IndexContainer, ArrayProgram::PaddedArray, ArrayProgram::UInt32, ArrayProgram::_padded_transpose_body, ArrayProgram::_padded_upper_bound> __while_loop2_23;
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
    typedef float64_utils::Float64 Float64;
    struct _binary_add {
        inline ArrayProgram::Int32 operator()(const ArrayProgram::Int32& a, const ArrayProgram::Int32& b) {
            return __int32_utils.binary_add(a, b);
        };
        inline ArrayProgram::Float64 operator()(const ArrayProgram::Float64& a, const ArrayProgram::Float64& b) {
            return __float64_utils.binary_add(a, b);
        };
    };

    static ArrayProgram::_binary_add binary_add;
    struct _binary_sub {
        inline ArrayProgram::Int32 operator()(const ArrayProgram::Int32& a, const ArrayProgram::Int32& b) {
            return __int32_utils.binary_sub(a, b);
        };
        inline ArrayProgram::Float64 operator()(const ArrayProgram::Float64& a, const ArrayProgram::Float64& b) {
            return __float64_utils.binary_sub(a, b);
        };
    };

    static ArrayProgram::_binary_sub binary_sub;
    struct _div {
        inline ArrayProgram::Int32 operator()(const ArrayProgram::Int32& a, const ArrayProgram::Int32& b) {
            return __int32_utils.div(a, b);
        };
        inline ArrayProgram::Float64 operator()(const ArrayProgram::Float64& a, const ArrayProgram::Float64& b) {
            return __float64_utils.div(a, b);
        };
    };

    static ArrayProgram::_div div;
    struct _eq {
        inline bool operator()(const ArrayProgram::Float64& a, const ArrayProgram::Float64& b) {
            return __float64_utils.eq(a, b);
        };
        inline bool operator()(const ArrayProgram::Int32& a, const ArrayProgram::Int32& b) {
            return __int32_utils.eq(a, b);
        };
    };

    static ArrayProgram::_eq eq;
    struct _lt {
        inline bool operator()(const ArrayProgram::Int32& a, const ArrayProgram::Int32& b) {
            return __int32_utils.lt(a, b);
        };
        inline bool operator()(const ArrayProgram::Float64& a, const ArrayProgram::Float64& b) {
            return __float64_utils.lt(a, b);
        };
    };

    static ArrayProgram::_lt lt;
    struct _mul {
        inline ArrayProgram::Int32 operator()(const ArrayProgram::Int32& a, const ArrayProgram::Int32& b) {
            return __int32_utils.mul(a, b);
        };
        inline ArrayProgram::Float64 operator()(const ArrayProgram::Float64& a, const ArrayProgram::Float64& b) {
            return __float64_utils.mul(a, b);
        };
    };

    static ArrayProgram::_mul mul;
    struct _unary_sub {
        inline ArrayProgram::Int32 operator()(const ArrayProgram::Int32& a) {
            return __int32_utils.unary_sub(a);
        };
        inline ArrayProgram::Float64 operator()(const ArrayProgram::Float64& a) {
            return __float64_utils.unary_sub(a);
        };
    };

    static ArrayProgram::_unary_sub unary_sub;
private:
    static inline void one0(ArrayProgram::Float64& o) {
        o = __float64_utils.one();
    };
    static inline void zero0(ArrayProgram::Float64& o) {
        o = __float64_utils.zero();
    };
public:
    typedef array<ArrayProgram::Int32>::Array Array;
    struct _bmapvector_body {
        inline void operator()(const ArrayProgram::Int32& e, ArrayProgram::Array& v, ArrayProgram::UInt32& c) {
            ArrayProgram::Int32 new_value = ArrayProgram::mul(e, ArrayProgram::unwrap_scalar(ArrayProgram::get(v, c)));
            ArrayProgram::set(v, c, new_value);
            c = ArrayProgram::elem_uint(ArrayProgram::binary_add(ArrayProgram::uint_elem(c), ArrayProgram::one.operator()<Int32>()));
        };
    };

    static ArrayProgram::_bmapvector_body bmapvector_body;
    struct _bmapvector_body_add {
        inline void operator()(const ArrayProgram::Int32& e, ArrayProgram::Array& v, ArrayProgram::UInt32& c) {
            ArrayProgram::Int32 new_value = ArrayProgram::binary_add(e, ArrayProgram::unwrap_scalar(ArrayProgram::get(v, c)));
            ArrayProgram::set(v, c, new_value);
            c = ArrayProgram::elem_uint(ArrayProgram::binary_add(ArrayProgram::uint_elem(c), ArrayProgram::one.operator()<Int32>()));
        };
    };

    static ArrayProgram::_bmapvector_body_add bmapvector_body_add;
    struct _bmapvector_cond {
        inline bool operator()(const ArrayProgram::Int32& e, const ArrayProgram::Array& v, const ArrayProgram::UInt32& c) {
            return ArrayProgram::lt(ArrayProgram::uint_elem(c), ArrayProgram::uint_elem(ArrayProgram::total(v)));
        };
    };

private:
    static while_loop1_2<ArrayProgram::Int32, ArrayProgram::Array, ArrayProgram::UInt32, ArrayProgram::_bmapvector_body, ArrayProgram::_bmapvector_cond> __while_loop1_23;
    static while_loop1_2<ArrayProgram::Int32, ArrayProgram::Array, ArrayProgram::UInt32, ArrayProgram::_bmapvector_body_add, ArrayProgram::_bmapvector_cond> __while_loop1_24;
public:
    static ArrayProgram::_bmapvector_cond bmapvector_cond;
    struct _bmapvector_repeat {
        inline void operator()(const ArrayProgram::Int32& context1, ArrayProgram::Array& state1, ArrayProgram::UInt32& state2) {
            return __while_loop1_23.repeat(context1, state1, state2);
        };
    };

    static ArrayProgram::_bmapvector_repeat bmapvector_repeat;
    struct _bmapvector_repeat_add {
        inline void operator()(const ArrayProgram::Int32& context1, ArrayProgram::Array& state1, ArrayProgram::UInt32& state2) {
            return __while_loop1_24.repeat(context1, state1, state2);
        };
    };

    static ArrayProgram::_bmapvector_repeat_add bmapvector_repeat_add;
    struct _bopmap_vec_add {
        inline void operator()(const ArrayProgram::Int32& e, ArrayProgram::Array& a) {
            ArrayProgram::UInt32 counter = ArrayProgram::elem_uint(ArrayProgram::zero.operator()<Int32>());
            ArrayProgram::bmapvector_repeat_add(e, a, counter);
        };
    };

    static ArrayProgram::_bopmap_vec_add bopmap_vec_add;
    struct _bopmap_vec_mult {
        inline void operator()(const ArrayProgram::Int32& e, ArrayProgram::Array& a) {
            ArrayProgram::UInt32 counter = ArrayProgram::elem_uint(ArrayProgram::zero.operator()<Int32>());
            ArrayProgram::bmapvector_repeat(e, a, counter);
        };
    };

    static ArrayProgram::_bopmap_vec_mult bopmap_vec_mult;
    struct _cat {
        inline ArrayProgram::Array operator()(const ArrayProgram::Array& array1, const ArrayProgram::Array& array2) {
            ArrayProgram::Int32 take_a1 = ArrayProgram::uint_elem(ArrayProgram::get_shape_elem(array1, ArrayProgram::elem_uint(ArrayProgram::zero.operator()<Int32>())));
            ArrayProgram::Int32 take_a2 = ArrayProgram::uint_elem(ArrayProgram::get_shape_elem(array2, ArrayProgram::elem_uint(ArrayProgram::zero.operator()<Int32>())));
            ArrayProgram::Shape drop_a1 = ArrayProgram::drop_shape_elem(array1, ArrayProgram::elem_uint(ArrayProgram::zero.operator()<Int32>()));
            ArrayProgram::Shape result_shape = ArrayProgram::cat_shape(ArrayProgram::create_shape1(ArrayProgram::elem_uint(ArrayProgram::binary_add(take_a1, take_a2))), drop_a1);
            ArrayProgram::Array res = ArrayProgram::create_array(result_shape);
            ArrayProgram::UInt32 counter = ArrayProgram::elem_uint(ArrayProgram::zero.operator()<Int32>());
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
                counter = ArrayProgram::elem_uint(ArrayProgram::binary_add(ArrayProgram::uint_elem(counter), ArrayProgram::one.operator()<Int32>()));
            }
            else
            {
                ArrayProgram::UInt32 ix = ArrayProgram::elem_uint(ArrayProgram::binary_sub(ArrayProgram::uint_elem(counter), s_0));
                ArrayProgram::set(res, counter, ArrayProgram::unwrap_scalar(ArrayProgram::get(array2, ix)));
                counter = ArrayProgram::elem_uint(ArrayProgram::binary_add(ArrayProgram::uint_elem(counter), ArrayProgram::one.operator()<Int32>()));
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
            ArrayProgram::UInt32 counter = ArrayProgram::elem_uint(ArrayProgram::zero.operator()<Int32>());
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
            counter = ArrayProgram::elem_uint(ArrayProgram::binary_add(ArrayProgram::uint_elem(counter), ArrayProgram::one.operator()<Int32>()));
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
        inline ArrayProgram::PaddedArray operator()(const ArrayProgram::Array& a, const ArrayProgram::UInt32& ix) {
            ArrayProgram::Array padding = ArrayProgram::get(a, ArrayProgram::create_index1(ix));
            ArrayProgram::Shape reshape_shape = ArrayProgram::cat_shape(ArrayProgram::create_shape1(ArrayProgram::elem_uint(ArrayProgram::one.operator()<Int32>())), ArrayProgram::shape(padding));
            ArrayProgram::Array reshaped_padding = ArrayProgram::reshape(padding, reshape_shape);
            ArrayProgram::Array catenated_array = ArrayProgram::cat(reshaped_padding, a);
            ArrayProgram::Shape unpadded_shape = ArrayProgram::shape(a);
            ArrayProgram::Shape padded_shape = ArrayProgram::shape(catenated_array);
            ArrayProgram::PaddedArray res = ArrayProgram::create_padded_array(unpadded_shape, padded_shape, catenated_array);
            return res;
        };
        inline ArrayProgram::PaddedArray operator()(const ArrayProgram::PaddedArray& a, const ArrayProgram::UInt32& ix) {
            ArrayProgram::Array padding = ArrayProgram::get(a, ArrayProgram::create_index1(ix));
            ArrayProgram::Shape reshape_shape = ArrayProgram::cat_shape(ArrayProgram::create_shape1(ArrayProgram::elem_uint(ArrayProgram::one.operator()<Int32>())), ArrayProgram::shape(padding));
            ArrayProgram::Array reshaped_padding = ArrayProgram::reshape(padding, reshape_shape);
            ArrayProgram::Array catenated_array = ArrayProgram::cat(reshaped_padding, ArrayProgram::padded_to_unpadded(a));
            ArrayProgram::Shape unpadded_shape = ArrayProgram::shape(a);
            ArrayProgram::Shape padded_shape = ArrayProgram::shape(catenated_array);
            ArrayProgram::PaddedArray res = ArrayProgram::create_padded_array(unpadded_shape, padded_shape, catenated_array);
            return res;
        };
    };

    static ArrayProgram::_circular_padl circular_padl;
    struct _circular_padr {
        inline ArrayProgram::PaddedArray operator()(const ArrayProgram::Array& a, const ArrayProgram::UInt32& ix) {
            ArrayProgram::Array padding = ArrayProgram::get(a, ArrayProgram::create_index1(ix));
            ArrayProgram::Shape reshape_shape = ArrayProgram::cat_shape(ArrayProgram::create_shape1(ArrayProgram::elem_uint(ArrayProgram::one.operator()<Int32>())), ArrayProgram::shape(padding));
            ArrayProgram::Array reshaped_padding = ArrayProgram::reshape(padding, reshape_shape);
            ArrayProgram::Array catenated_array = ArrayProgram::cat(a, reshaped_padding);
            ArrayProgram::Shape unpadded_shape = ArrayProgram::shape(a);
            ArrayProgram::Shape padded_shape = ArrayProgram::shape(catenated_array);
            ArrayProgram::PaddedArray res = ArrayProgram::create_padded_array(unpadded_shape, padded_shape, catenated_array);
            return res;
        };
        inline ArrayProgram::PaddedArray operator()(const ArrayProgram::PaddedArray& a, const ArrayProgram::UInt32& ix) {
            ArrayProgram::Shape unpadded_shape = ArrayProgram::shape(a);
            ArrayProgram::Array padding = ArrayProgram::get(a, ArrayProgram::create_index1(ix));
            ArrayProgram::Shape reshape_shape = ArrayProgram::cat_shape(ArrayProgram::create_shape1(ArrayProgram::elem_uint(ArrayProgram::one.operator()<Int32>())), ArrayProgram::shape(padding));
            ArrayProgram::Array reshaped_padding = ArrayProgram::reshape(padding, reshape_shape);
            ArrayProgram::Array catenated_array = ArrayProgram::cat(ArrayProgram::padded_to_unpadded(a), reshaped_padding);
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
    struct _inner_matmult_body {
        inline void operator()(const ArrayProgram::Array& a1, const ArrayProgram::Array& a2, const ArrayProgram::Index& ix_i, const ArrayProgram::Index& ix_j, const ArrayProgram::IndexContainer& ixc_k, ArrayProgram::Array& res, ArrayProgram::UInt32& counter) {
            ArrayProgram::Index ix_k = ArrayProgram::get_index_ixc(ixc_k, counter);
            ArrayProgram::Index i_cat_k = ArrayProgram::cat_index(ix_i, ix_k);
            ArrayProgram::Array ik_from_a1 = ArrayProgram::get(a1, i_cat_k);
            ArrayProgram::Array k_from_a2 = ArrayProgram::get(a2, ix_k);
            ArrayProgram::bopmap_vec_mult(ArrayProgram::unwrap_scalar(ik_from_a1), k_from_a2);
            ArrayProgram::print_array(k_from_a2);
            res = ArrayProgram::pointwise_add(res, k_from_a2);
            counter = ArrayProgram::elem_uint(ArrayProgram::binary_add(ArrayProgram::uint_elem(counter), ArrayProgram::one.operator()<Int32>()));
        };
    };

    static ArrayProgram::_inner_matmult_body inner_matmult_body;
    struct _inner_matmult_cond {
        inline bool operator()(const ArrayProgram::Array& a1, const ArrayProgram::Array& a2, const ArrayProgram::Index& ix_i, const ArrayProgram::Index& ix_j, const ArrayProgram::IndexContainer& ixc_k, const ArrayProgram::Array& res, const ArrayProgram::UInt32& c) {
            return ArrayProgram::lt(ArrayProgram::uint_elem(c), ArrayProgram::uint_elem(ArrayProgram::total(ixc_k)));
        };
    };

private:
    static while_loop5_2<ArrayProgram::Array, ArrayProgram::Array, ArrayProgram::Index, ArrayProgram::Index, ArrayProgram::IndexContainer, ArrayProgram::Array, ArrayProgram::UInt32, ArrayProgram::_inner_matmult_body, ArrayProgram::_inner_matmult_cond> __while_loop5_2;
public:
    static ArrayProgram::_inner_matmult_cond inner_matmult_cond;
    struct _inner_matmult_repeat {
        inline void operator()(const ArrayProgram::Array& context1, const ArrayProgram::Array& context2, const ArrayProgram::Index& context3, const ArrayProgram::Index& context4, const ArrayProgram::IndexContainer& context5, ArrayProgram::Array& state1, ArrayProgram::UInt32& state2) {
            return __while_loop5_2.repeat(context1, context2, context3, context4, context5, state1, state2);
        };
    };

    static ArrayProgram::_inner_matmult_repeat inner_matmult_repeat;
    struct _matmult2d {
        inline ArrayProgram::Array operator()(const ArrayProgram::Array& a1, const ArrayProgram::Array& a2) {
            ArrayProgram::UInt32 shape_a1_last_ix = ArrayProgram::elem_uint(ArrayProgram::binary_sub(ArrayProgram::uint_elem(ArrayProgram::total(ArrayProgram::shape(a1))), ArrayProgram::one.operator()<Int32>()));
            ArrayProgram::Shape sh_a1_drop_last = ArrayProgram::drop_shape_elem(a1, shape_a1_last_ix);
            ArrayProgram::Shape sh_a2_drop_first = ArrayProgram::drop_shape_elem(a2, ArrayProgram::elem_uint(ArrayProgram::zero.operator()<Int32>()));
            ArrayProgram::Shape ip_shape = ArrayProgram::cat_shape(sh_a1_drop_last, sh_a2_drop_first);
            ArrayProgram::Array ip_array = ArrayProgram::create_array(ip_shape);
            ArrayProgram::IndexContainer indices_a1 = ArrayProgram::create_valid_indices(ArrayProgram::create_array(sh_a1_drop_last));
            ArrayProgram::IndexContainer indices_a2 = ArrayProgram::create_valid_indices(ArrayProgram::create_array(sh_a2_drop_first));
            ArrayProgram::Shape sh_a1_take_last = ArrayProgram::create_shape1(ArrayProgram::get_shape_elem(a1, shape_a1_last_ix));
            ArrayProgram::IndexContainer indices_k = ArrayProgram::create_valid_indices(ArrayProgram::create_array(sh_a1_take_last));
            ArrayProgram::UInt32 counter = ArrayProgram::elem_uint(ArrayProgram::zero.operator()<Int32>());
            ArrayProgram::matmult2d_repeat(a1, a2, indices_a1, indices_a2, indices_k, ip_array, counter);
            return ip_array;
        };
    };

    static ArrayProgram::_matmult2d matmult2d;
    struct _matmult2d_body {
        inline void operator()(const ArrayProgram::Array& a1, const ArrayProgram::Array& a2, const ArrayProgram::IndexContainer& ixc_i, const ArrayProgram::IndexContainer& ixc_j, const ArrayProgram::IndexContainer& ixc_k, ArrayProgram::Array& res, ArrayProgram::UInt32& counter) {
            ArrayProgram::Index ix_i = ArrayProgram::get_index_ixc(ixc_i, counter);
            ArrayProgram::UInt32 middle_counter = ArrayProgram::elem_uint(ArrayProgram::zero.operator()<Int32>());
            ArrayProgram::middle_matmult_repeat(a1, a2, ix_i, ixc_j, ixc_k, res, middle_counter);
            counter = ArrayProgram::elem_uint(ArrayProgram::binary_add(ArrayProgram::uint_elem(counter), ArrayProgram::one.operator()<Int32>()));
        };
    };

    static ArrayProgram::_matmult2d_body matmult2d_body;
    struct _matmult2d_cond {
        inline bool operator()(const ArrayProgram::Array& a1, const ArrayProgram::Array& a2, const ArrayProgram::IndexContainer& ixc_i, const ArrayProgram::IndexContainer& ixc_j, const ArrayProgram::IndexContainer& ixc_k, const ArrayProgram::Array& res, const ArrayProgram::UInt32& c) {
            return ArrayProgram::lt(ArrayProgram::uint_elem(c), ArrayProgram::uint_elem(ArrayProgram::total(ixc_i)));
        };
    };

private:
    static while_loop5_2<ArrayProgram::Array, ArrayProgram::Array, ArrayProgram::IndexContainer, ArrayProgram::IndexContainer, ArrayProgram::IndexContainer, ArrayProgram::Array, ArrayProgram::UInt32, ArrayProgram::_matmult2d_body, ArrayProgram::_matmult2d_cond> __while_loop5_21;
public:
    static ArrayProgram::_matmult2d_cond matmult2d_cond;
    struct _matmult2d_repeat {
        inline void operator()(const ArrayProgram::Array& context1, const ArrayProgram::Array& context2, const ArrayProgram::IndexContainer& context3, const ArrayProgram::IndexContainer& context4, const ArrayProgram::IndexContainer& context5, ArrayProgram::Array& state1, ArrayProgram::UInt32& state2) {
            return __while_loop5_21.repeat(context1, context2, context3, context4, context5, state1, state2);
        };
    };

    static ArrayProgram::_matmult2d_repeat matmult2d_repeat;
    struct _middle_matmult_body {
        inline void operator()(const ArrayProgram::Array& a1, const ArrayProgram::Array& a2, const ArrayProgram::Index& ix_i, const ArrayProgram::IndexContainer& ixc_j, const ArrayProgram::IndexContainer& ixc_k, ArrayProgram::Array& res, ArrayProgram::UInt32& counter) {
            ArrayProgram::Index ix_j = ArrayProgram::get_index_ixc(ixc_j, counter);
            ArrayProgram::UInt32 inner_counter = ArrayProgram::elem_uint(ArrayProgram::zero.operator()<Int32>());
            ArrayProgram::Array res_vec = ArrayProgram::create_array(ArrayProgram::shape(ArrayProgram::get(a1, ix_j)));
            ArrayProgram::inner_matmult_repeat(a1, a2, ix_i, ix_j, ixc_k, res_vec, inner_counter);
            ArrayProgram::Int32 reduced = ArrayProgram::reduce_vec_add(res_vec);
            ArrayProgram::print_index(ArrayProgram::cat_index(ix_i, ix_j));
            ArrayProgram::set(res, ArrayProgram::cat_index(ix_i, ix_j), reduced);
            counter = ArrayProgram::elem_uint(ArrayProgram::binary_add(ArrayProgram::uint_elem(counter), ArrayProgram::one.operator()<Int32>()));
        };
    };

    static ArrayProgram::_middle_matmult_body middle_matmult_body;
    struct _middle_matmult_cond {
        inline bool operator()(const ArrayProgram::Array& a1, const ArrayProgram::Array& a2, const ArrayProgram::Index& ix_i, const ArrayProgram::IndexContainer& ixc_j, const ArrayProgram::IndexContainer& ixc_k, const ArrayProgram::Array& res, const ArrayProgram::UInt32& c) {
            return ArrayProgram::lt(ArrayProgram::uint_elem(c), ArrayProgram::uint_elem(ArrayProgram::total(ixc_j)));
        };
    };

private:
    static while_loop5_2<ArrayProgram::Array, ArrayProgram::Array, ArrayProgram::Index, ArrayProgram::IndexContainer, ArrayProgram::IndexContainer, ArrayProgram::Array, ArrayProgram::UInt32, ArrayProgram::_middle_matmult_body, ArrayProgram::_middle_matmult_cond> __while_loop5_20;
public:
    static ArrayProgram::_middle_matmult_cond middle_matmult_cond;
    struct _middle_matmult_repeat {
        inline void operator()(const ArrayProgram::Array& context1, const ArrayProgram::Array& context2, const ArrayProgram::Index& context3, const ArrayProgram::IndexContainer& context4, const ArrayProgram::IndexContainer& context5, ArrayProgram::Array& state1, ArrayProgram::UInt32& state2) {
            return __while_loop5_20.repeat(context1, context2, context3, context4, context5, state1, state2);
        };
    };

    static ArrayProgram::_middle_matmult_repeat middle_matmult_repeat;
    struct _padded_to_unpadded {
        inline ArrayProgram::Array operator()(const ArrayProgram::PaddedArray& a) {
            return __array.padded_to_unpadded(a);
        };
    };

    static ArrayProgram::_padded_to_unpadded padded_to_unpadded;
    struct _pointwise_add {
        inline ArrayProgram::Array operator()(const ArrayProgram::Array& a1, const ArrayProgram::Array& a2) {
            ArrayProgram::Array res = ArrayProgram::create_array(ArrayProgram::shape(a1));
            ArrayProgram::UInt32 c = ArrayProgram::elem_uint(ArrayProgram::zero.operator()<Int32>());
            ArrayProgram::pointwise_repeat(a1, res, c);
            c = ArrayProgram::elem_uint(ArrayProgram::zero.operator()<Int32>());
            ArrayProgram::pointwise_repeat(a2, res, c);
            return res;
        };
    };

    static ArrayProgram::_pointwise_add pointwise_add;
    struct _pointwise_add_vector_body {
        inline void operator()(const ArrayProgram::Array& v, ArrayProgram::Array& res, ArrayProgram::UInt32& c) {
            ArrayProgram::Int32 elem = ArrayProgram::unwrap_scalar(ArrayProgram::get(v, c));
            ArrayProgram::set(res, c, ArrayProgram::binary_add(ArrayProgram::unwrap_scalar(ArrayProgram::get(res, c)), elem));
            c = ArrayProgram::elem_uint(ArrayProgram::binary_add(ArrayProgram::uint_elem(c), ArrayProgram::one.operator()<Int32>()));
        };
    };

    static ArrayProgram::_pointwise_add_vector_body pointwise_add_vector_body;
    struct _pointwise_cond {
        inline bool operator()(const ArrayProgram::Array& v, const ArrayProgram::Array& res, const ArrayProgram::UInt32& c) {
            return ArrayProgram::lt(ArrayProgram::uint_elem(c), ArrayProgram::uint_elem(ArrayProgram::total(res)));
        };
    };

private:
    static while_loop1_2<ArrayProgram::Array, ArrayProgram::Array, ArrayProgram::UInt32, ArrayProgram::_pointwise_add_vector_body, ArrayProgram::_pointwise_cond> __while_loop1_2;
public:
    static ArrayProgram::_pointwise_cond pointwise_cond;
    struct _pointwise_repeat {
        inline void operator()(const ArrayProgram::Array& context1, ArrayProgram::Array& state1, ArrayProgram::UInt32& state2) {
            return __while_loop1_2.repeat(context1, state1, state2);
        };
    };

    static ArrayProgram::_pointwise_repeat pointwise_repeat;
    struct _print_array {
        inline void operator()(const ArrayProgram::Array& a) {
            return __array.print_array(a);
        };
    };

    static ArrayProgram::_print_array print_array;
    struct _reduce_body {
        inline void operator()(const ArrayProgram::Array& input, ArrayProgram::Int32& res, ArrayProgram::UInt32& c) {
            ArrayProgram::Int32 current_element = ArrayProgram::unwrap_scalar(ArrayProgram::get(input, c));
            res = ArrayProgram::mul(res, current_element);
            c = ArrayProgram::elem_uint(ArrayProgram::binary_add(ArrayProgram::uint_elem(c), ArrayProgram::one.operator()<Int32>()));
        };
    };

    static ArrayProgram::_reduce_body reduce_body;
    struct _reduce_body_add {
        inline void operator()(const ArrayProgram::Array& input, ArrayProgram::Int32& res, ArrayProgram::UInt32& c) {
            ArrayProgram::Int32 current_element = ArrayProgram::unwrap_scalar(ArrayProgram::get(input, c));
            res = ArrayProgram::binary_add(res, current_element);
            c = ArrayProgram::elem_uint(ArrayProgram::binary_add(ArrayProgram::uint_elem(c), ArrayProgram::one.operator()<Int32>()));
        };
    };

    static ArrayProgram::_reduce_body_add reduce_body_add;
    struct _reduce_cond {
        inline bool operator()(const ArrayProgram::Array& input, const ArrayProgram::Int32& res, const ArrayProgram::UInt32& c) {
            return ArrayProgram::lt(ArrayProgram::uint_elem(c), ArrayProgram::uint_elem(ArrayProgram::total(input)));
        };
    };

private:
    static while_loop1_2<ArrayProgram::Array, ArrayProgram::Int32, ArrayProgram::UInt32, ArrayProgram::_reduce_body, ArrayProgram::_reduce_cond> __while_loop1_21;
    static while_loop1_2<ArrayProgram::Array, ArrayProgram::Int32, ArrayProgram::UInt32, ArrayProgram::_reduce_body_add, ArrayProgram::_reduce_cond> __while_loop1_22;
public:
    static ArrayProgram::_reduce_cond reduce_cond;
    struct _reduce_vec_add {
        inline ArrayProgram::Int32 operator()(const ArrayProgram::Array& a) {
            ArrayProgram::Int32 result = ArrayProgram::zero.operator()<Int32>();
            ArrayProgram::UInt32 counter = ArrayProgram::elem_uint(ArrayProgram::zero.operator()<Int32>());
            ArrayProgram::repeat_reduce_vec_add(a, result, counter);
            return result;
        };
    };

    static ArrayProgram::_reduce_vec_add reduce_vec_add;
    struct _reduce_vec_mult {
        inline ArrayProgram::Int32 operator()(const ArrayProgram::Array& a) {
            ArrayProgram::Int32 result = ArrayProgram::one.operator()<Int32>();
            ArrayProgram::UInt32 counter = ArrayProgram::elem_uint(ArrayProgram::zero.operator()<Int32>());
            ArrayProgram::repeat_reduce_vec_mult(a, result, counter);
            return result;
        };
    };

    static ArrayProgram::_reduce_vec_mult reduce_vec_mult;
    struct _repeat_reduce_vec_add {
        inline void operator()(const ArrayProgram::Array& context1, ArrayProgram::Int32& state1, ArrayProgram::UInt32& state2) {
            return __while_loop1_22.repeat(context1, state1, state2);
        };
    };

    static ArrayProgram::_repeat_reduce_vec_add repeat_reduce_vec_add;
    struct _repeat_reduce_vec_mult {
        inline void operator()(const ArrayProgram::Array& context1, ArrayProgram::Int32& state1, ArrayProgram::UInt32& state2) {
            return __while_loop1_21.repeat(context1, state1, state2);
        };
    };

    static ArrayProgram::_repeat_reduce_vec_mult repeat_reduce_vec_mult;
    struct _reshape {
        inline ArrayProgram::Array operator()(const ArrayProgram::Array& input_array, const ArrayProgram::Shape& s) {
            ArrayProgram::Array new_array = ArrayProgram::create_array(s);
            ArrayProgram::UInt32 counter = ArrayProgram::elem_uint(ArrayProgram::zero.operator()<Int32>());
            ArrayProgram::reshape_repeat(input_array, new_array, counter);
            return new_array;
        };
    };

    static ArrayProgram::_reshape reshape;
    struct _reshape_body {
        inline void operator()(const ArrayProgram::Array& old_array, ArrayProgram::Array& new_array, ArrayProgram::UInt32& counter) {
            ArrayProgram::set(new_array, counter, ArrayProgram::unwrap_scalar(ArrayProgram::get(old_array, counter)));
            counter = ArrayProgram::elem_uint(ArrayProgram::binary_add(ArrayProgram::uint_elem(counter), ArrayProgram::one.operator()<Int32>()));
        };
    };

    static ArrayProgram::_reshape_body reshape_body;
    struct _reshape_cond {
        inline bool operator()(const ArrayProgram::Array& old_array, const ArrayProgram::Array& new_array, const ArrayProgram::UInt32& counter) {
            return ArrayProgram::lt(ArrayProgram::uint_elem(counter), ArrayProgram::uint_elem(ArrayProgram::total(new_array)));
        };
    };

private:
    static while_loop1_2<ArrayProgram::Array, ArrayProgram::Array, ArrayProgram::UInt32, ArrayProgram::_reshape_body, ArrayProgram::_reshape_cond> __while_loop1_20;
public:
    static ArrayProgram::_reshape_cond reshape_cond;
    struct _reshape_repeat {
        inline void operator()(const ArrayProgram::Array& context1, ArrayProgram::Array& state1, ArrayProgram::UInt32& state2) {
            return __while_loop1_20.repeat(context1, state1, state2);
        };
    };

    static ArrayProgram::_reshape_repeat reshape_repeat;
    struct _reverse {
        inline ArrayProgram::Array operator()(const ArrayProgram::Array& a) {
            ArrayProgram::Array res_array = ArrayProgram::create_array(ArrayProgram::shape(a));
            ArrayProgram::IndexContainer valid_indices = ArrayProgram::create_valid_indices(res_array);
            ArrayProgram::UInt32 counter = ArrayProgram::elem_uint(ArrayProgram::zero.operator()<Int32>());
            ArrayProgram::reverse_repeat(a, valid_indices, res_array, counter);
            return res_array;
        };
    };

    static ArrayProgram::_reverse reverse;
    struct _reverse_body {
        inline void operator()(const ArrayProgram::Array& input, const ArrayProgram::IndexContainer& indices, ArrayProgram::Array& res, ArrayProgram::UInt32& c) {
            ArrayProgram::Index ix = ArrayProgram::get_index_ixc(indices, c);
            ArrayProgram::Int32 elem = ArrayProgram::unwrap_scalar(ArrayProgram::get(input, ix));
            ArrayProgram::UInt32 sh_0 = ArrayProgram::get_shape_elem(input, ArrayProgram::elem_uint(ArrayProgram::zero.operator()<Int32>()));
            ArrayProgram::UInt32 ix_0 = ArrayProgram::get_index_elem(ix, ArrayProgram::elem_uint(ArrayProgram::zero.operator()<Int32>()));
            ArrayProgram::Int32 new_ix_0 = ArrayProgram::binary_sub(ArrayProgram::uint_elem(sh_0), ArrayProgram::binary_add(ArrayProgram::uint_elem(ix_0), ArrayProgram::one.operator()<Int32>()));
            ArrayProgram::Index new_ix = ArrayProgram::cat_index(ArrayProgram::create_index1(ArrayProgram::elem_uint(new_ix_0)), ArrayProgram::drop_index_elem(ix, ArrayProgram::elem_uint(ArrayProgram::zero.operator()<Int32>())));
            ArrayProgram::set(res, new_ix, elem);
            c = ArrayProgram::elem_uint(ArrayProgram::binary_add(ArrayProgram::uint_elem(c), ArrayProgram::one.operator()<Int32>()));
        };
    };

    static ArrayProgram::_reverse_body reverse_body;
    struct _reverse_cond {
        inline bool operator()(const ArrayProgram::Array& input, const ArrayProgram::IndexContainer& indices, const ArrayProgram::Array& res, const ArrayProgram::UInt32& c) {
            return ArrayProgram::lt(ArrayProgram::uint_elem(c), ArrayProgram::uint_elem(ArrayProgram::total(indices)));
        };
    };

private:
    static while_loop2_2<ArrayProgram::Array, ArrayProgram::IndexContainer, ArrayProgram::Array, ArrayProgram::UInt32, ArrayProgram::_reverse_body, ArrayProgram::_reverse_cond> __while_loop2_21;
public:
    static ArrayProgram::_reverse_cond reverse_cond;
    struct _reverse_repeat {
        inline void operator()(const ArrayProgram::Array& context1, const ArrayProgram::IndexContainer& context2, ArrayProgram::Array& state1, ArrayProgram::UInt32& state2) {
            return __while_loop2_21.repeat(context1, context2, state1, state2);
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
        inline ArrayProgram::PaddedArray operator()(const ArrayProgram::PaddedArray& a) {
            ArrayProgram::Array reshaped_array = ArrayProgram::create_array(ArrayProgram::padded_shape(a));
            ArrayProgram::PaddedArray transposed_array = ArrayProgram::create_padded_array(ArrayProgram::reverse_shape(ArrayProgram::shape(a)), ArrayProgram::reverse_shape(ArrayProgram::padded_shape(a)), reshaped_array);
            ArrayProgram::IndexContainer ix_space = ArrayProgram::create_valid_indices(transposed_array);
            ArrayProgram::UInt32 counter = ArrayProgram::elem_uint(ArrayProgram::zero.operator()<Int32>());
            ArrayProgram::padded_transpose_repeat(a, ix_space, transposed_array, counter);
            return transposed_array;
        };
        inline ArrayProgram::Array operator()(const ArrayProgram::Array& a) {
            ArrayProgram::Array transposed_array = ArrayProgram::create_array(ArrayProgram::reverse_shape(ArrayProgram::shape(a)));
            ArrayProgram::IndexContainer ix_space = ArrayProgram::create_valid_indices(transposed_array);
            ArrayProgram::UInt32 counter = ArrayProgram::elem_uint(ArrayProgram::zero.operator()<Int32>());
            ArrayProgram::transpose_repeat(a, ix_space, transposed_array, counter);
            return transposed_array;
        };
    };

    static ArrayProgram::_transpose transpose;
    struct _transpose_body {
        inline void operator()(const ArrayProgram::Array& a, const ArrayProgram::IndexContainer& ixc, ArrayProgram::Array& res, ArrayProgram::UInt32& c) {
            ArrayProgram::Index current_ix = ArrayProgram::get_index_ixc(ixc, c);
            ArrayProgram::Int32 current_element = ArrayProgram::unwrap_scalar(ArrayProgram::get(a, ArrayProgram::reverse_index(current_ix)));
            ArrayProgram::set(res, current_ix, current_element);
            c = ArrayProgram::elem_uint(ArrayProgram::binary_add(ArrayProgram::uint_elem(c), ArrayProgram::one.operator()<Int32>()));
        };
    };

    static ArrayProgram::_transpose_body transpose_body;
    struct _transpose_repeat {
        inline void operator()(const ArrayProgram::Array& context1, const ArrayProgram::IndexContainer& context2, ArrayProgram::Array& state1, ArrayProgram::UInt32& state2) {
            return __while_loop2_22.repeat(context1, context2, state1, state2);
        };
    };

    static ArrayProgram::_transpose_repeat transpose_repeat;
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
    static while_loop2_2<ArrayProgram::Array, ArrayProgram::IndexContainer, ArrayProgram::Array, ArrayProgram::UInt32, ArrayProgram::_transpose_body, ArrayProgram::_upper_bound> __while_loop2_22;
public:
    static ArrayProgram::_upper_bound upper_bound;
};
} // examples
} // moa
} // mg_src
} // moa_cpp