#pragma once

#include "base.hpp"
#include <cassert>


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
private:
    static array<BurgerProgram::Int32> __array;
public:
    struct _abs {
        inline BurgerProgram::Int32 operator()(const BurgerProgram::Int32& a) {
            return __int32_utils.abs(a);
        };
    };

    static BurgerProgram::_abs abs;
    struct _eq {
        inline bool operator()(const BurgerProgram::Int32& a, const BurgerProgram::Int32& b) {
            return __int32_utils.eq(a, b);
        };
    };

    static BurgerProgram::_eq eq;
    struct _le {
        inline bool operator()(const BurgerProgram::Int32& a, const BurgerProgram::Int32& b) {
            return __int32_utils.le(a, b);
        };
    };

    static BurgerProgram::_le le;
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
    struct _zero {
        inline BurgerProgram::Int32 operator()() {
            return __int32_utils.zero();
        };
    };

    static BurgerProgram::_zero zero;
    typedef array<BurgerProgram::Int32>::Int Int;
    struct _create_shape1 {
        inline BurgerProgram::Shape operator()(const BurgerProgram::Int& a) {
            return __array.create_shape1(a);
        };
    };

    static BurgerProgram::_create_shape1 create_shape1;
    struct _create_shape2 {
        inline BurgerProgram::Shape operator()(const BurgerProgram::Int& a, const BurgerProgram::Int& b) {
            return __array.create_shape2(a, b);
        };
    };

    static BurgerProgram::_create_shape2 create_shape2;
    struct _create_shape3 {
        inline BurgerProgram::Shape operator()(const BurgerProgram::Int& a, const BurgerProgram::Int& b, const BurgerProgram::Int& c) {
            return __array.create_shape3(a, b, c);
        };
    };

    static BurgerProgram::_create_shape3 create_shape3;
    struct _elem_int {
        inline BurgerProgram::Int operator()(const BurgerProgram::Int32& a) {
            return __array.elem_int(a);
        };
    };

    static BurgerProgram::_elem_int elem_int;
    struct _int_elem {
        inline BurgerProgram::Int32 operator()(const BurgerProgram::Int& a) {
            return __array.int_elem(a);
        };
    };

    static BurgerProgram::_int_elem int_elem;
    struct _padded_dim {
        inline BurgerProgram::Int operator()(const BurgerProgram::PaddedArray& a) {
            return __array.padded_dim(a);
        };
    };

    static BurgerProgram::_padded_dim padded_dim;
    struct _padded_drop_shape_elem {
        inline BurgerProgram::Shape operator()(const BurgerProgram::PaddedArray& a, const BurgerProgram::Int& i) {
            return __array.padded_drop_shape_elem(a, i);
        };
    };

    static BurgerProgram::_padded_drop_shape_elem padded_drop_shape_elem;
    struct _padded_get_shape_elem {
        inline BurgerProgram::Int operator()(const BurgerProgram::PaddedArray& a, const BurgerProgram::Int& i) {
            return __array.padded_get_shape_elem(a, i);
        };
    };

    static BurgerProgram::_padded_get_shape_elem padded_get_shape_elem;
    struct _padded_total {
        inline BurgerProgram::Int operator()(const BurgerProgram::PaddedArray& a) {
            return __array.padded_total(a);
        };
    };

    static BurgerProgram::_padded_total padded_total;
    struct _print_int {
        inline void operator()(const BurgerProgram::Int& u) {
            return __array.print_int(u);
        };
    };

    static BurgerProgram::_print_int print_int;
    typedef array<BurgerProgram::Int32>::IndexContainer IndexContainer;
    struct _padded_transpose_body {
        inline void operator()(const BurgerProgram::PaddedArray& a, const BurgerProgram::IndexContainer& ixc, BurgerProgram::PaddedArray& res, BurgerProgram::Int& c) {
            BurgerProgram::Index current_ix = BurgerProgram::get_index_ixc(ixc, c);
            BurgerProgram::Int32 current_element = BurgerProgram::unwrap_scalar(BurgerProgram::get(a, BurgerProgram::reverse_index(current_ix)));
            BurgerProgram::set(res, current_ix, current_element);
            c = BurgerProgram::elem_int(BurgerProgram::binary_add(BurgerProgram::int_elem(c), BurgerProgram::one()));
        };
    };

    static BurgerProgram::_padded_transpose_body padded_transpose_body;
    struct _padded_transpose_repeat {
        inline void operator()(const BurgerProgram::PaddedArray& context1, const BurgerProgram::IndexContainer& context2, BurgerProgram::PaddedArray& state1, BurgerProgram::Int& state2) {
            return __while_loop2_29.repeat(context1, context2, state1, state2);
        };
    };

    static BurgerProgram::_padded_transpose_repeat padded_transpose_repeat;
    struct _padded_upper_bound {
        inline bool operator()(const BurgerProgram::PaddedArray& a, const BurgerProgram::IndexContainer& i, const BurgerProgram::PaddedArray& res, const BurgerProgram::Int& c) {
            return BurgerProgram::lt(BurgerProgram::int_elem(c), BurgerProgram::int_elem(BurgerProgram::total(i)));
        };
    };

private:
    static while_loop2_2<BurgerProgram::PaddedArray, BurgerProgram::IndexContainer, BurgerProgram::PaddedArray, BurgerProgram::Int, BurgerProgram::_padded_transpose_body, BurgerProgram::_padded_upper_bound> __while_loop2_29;
public:
    static BurgerProgram::_padded_upper_bound padded_upper_bound;
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
        inline BurgerProgram::Index operator()(const BurgerProgram::Int& a) {
            return __array.create_index1(a);
        };
    };

    static BurgerProgram::_create_index1 create_index1;
    struct _create_index2 {
        inline BurgerProgram::Index operator()(const BurgerProgram::Int& a, const BurgerProgram::Int& b) {
            return __array.create_index2(a, b);
        };
    };

    static BurgerProgram::_create_index2 create_index2;
    struct _create_index3 {
        inline BurgerProgram::Index operator()(const BurgerProgram::Int& a, const BurgerProgram::Int& b, const BurgerProgram::Int& c) {
            return __array.create_index3(a, b, c);
        };
    };

    static BurgerProgram::_create_index3 create_index3;
    struct _drop_index_elem {
        inline BurgerProgram::Index operator()(const BurgerProgram::Index& ix, const BurgerProgram::Int& i) {
            return __array.drop_index_elem(ix, i);
        };
    };

    static BurgerProgram::_drop_index_elem drop_index_elem;
    struct _get_index_elem {
        inline BurgerProgram::Int operator()(const BurgerProgram::Index& ix, const BurgerProgram::Int& i) {
            return __array.get_index_elem(ix, i);
        };
    };

    static BurgerProgram::_get_index_elem get_index_elem;
    struct _get_index_ixc {
        inline BurgerProgram::Index operator()(const BurgerProgram::IndexContainer& ixc, const BurgerProgram::Int& ix) {
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
        inline BurgerProgram::Array operator()(const BurgerProgram::Array& a, const BurgerProgram::Array& b) {
            BurgerProgram::IndexContainer ix_space = BurgerProgram::create_total_indices(a);
            BurgerProgram::Array b_upd = b;
            BurgerProgram::Int counter = BurgerProgram::elem_int(BurgerProgram::zero());
            BurgerProgram::bmb_plus_rep(a, ix_space, b_upd, counter);
            return b;
        };
        inline BurgerProgram::Int32 operator()(const BurgerProgram::Int32& a, const BurgerProgram::Int32& b) {
            return __int32_utils.binary_add(a, b);
        };
    };

    static BurgerProgram::_binary_add binary_add;
    struct _binary_sub {
        inline BurgerProgram::Array operator()(const BurgerProgram::Array& a, const BurgerProgram::Array& b) {
            BurgerProgram::IndexContainer ix_space = BurgerProgram::create_total_indices(a);
            BurgerProgram::Array b_upd = b;
            BurgerProgram::Int counter = BurgerProgram::elem_int(BurgerProgram::zero());
            BurgerProgram::bmb_sub_rep(a, ix_space, b_upd, counter);
            return b;
        };
        inline BurgerProgram::Int32 operator()(const BurgerProgram::Int32& a, const BurgerProgram::Int32& b) {
            return __int32_utils.binary_sub(a, b);
        };
    };

    static BurgerProgram::_binary_sub binary_sub;
    struct _bmb_div {
        inline void operator()(const BurgerProgram::Array& a, const BurgerProgram::IndexContainer& ix_space, BurgerProgram::Array& b, BurgerProgram::Int& c) {
            BurgerProgram::Index ix = BurgerProgram::get_index_ixc(ix_space, c);
            BurgerProgram::Int32 new_value = BurgerProgram::div(BurgerProgram::unwrap_scalar(BurgerProgram::get(a, ix)), BurgerProgram::unwrap_scalar(BurgerProgram::get(b, ix)));
            BurgerProgram::set(b, ix, new_value);
            c = BurgerProgram::elem_int(BurgerProgram::binary_add(BurgerProgram::int_elem(c), BurgerProgram::one()));
        };
    };

    static BurgerProgram::_bmb_div bmb_div;
    struct _bmb_div_rep {
        inline void operator()(const BurgerProgram::Array& context1, const BurgerProgram::IndexContainer& context2, BurgerProgram::Array& state1, BurgerProgram::Int& state2) {
            return __while_loop2_21.repeat(context1, context2, state1, state2);
        };
    };

    static BurgerProgram::_bmb_div_rep bmb_div_rep;
    struct _bmb_mul {
        inline void operator()(const BurgerProgram::Array& a, const BurgerProgram::IndexContainer& ix_space, BurgerProgram::Array& b, BurgerProgram::Int& c) {
            BurgerProgram::Index ix = BurgerProgram::get_index_ixc(ix_space, c);
            BurgerProgram::Int32 new_value = BurgerProgram::mul(BurgerProgram::unwrap_scalar(BurgerProgram::get(a, ix)), BurgerProgram::unwrap_scalar(BurgerProgram::get(b, ix)));
            BurgerProgram::set(b, ix, new_value);
            c = BurgerProgram::elem_int(BurgerProgram::binary_add(BurgerProgram::int_elem(c), BurgerProgram::one()));
        };
    };

    static BurgerProgram::_bmb_mul bmb_mul;
    struct _bmb_mul_rep {
        inline void operator()(const BurgerProgram::Array& context1, const BurgerProgram::IndexContainer& context2, BurgerProgram::Array& state1, BurgerProgram::Int& state2) {
            return __while_loop2_22.repeat(context1, context2, state1, state2);
        };
    };

    static BurgerProgram::_bmb_mul_rep bmb_mul_rep;
    struct _bmb_plus {
        inline void operator()(const BurgerProgram::Array& a, const BurgerProgram::IndexContainer& ix_space, BurgerProgram::Array& b, BurgerProgram::Int& c) {
            BurgerProgram::Index ix = BurgerProgram::get_index_ixc(ix_space, c);
            BurgerProgram::Int32 new_value = BurgerProgram::binary_add(BurgerProgram::unwrap_scalar(BurgerProgram::get(a, ix)), BurgerProgram::unwrap_scalar(BurgerProgram::get(b, ix)));
            BurgerProgram::set(b, ix, new_value);
            c = BurgerProgram::elem_int(BurgerProgram::binary_add(BurgerProgram::int_elem(c), BurgerProgram::one()));
        };
    };

    static BurgerProgram::_bmb_plus bmb_plus;
    struct _bmb_plus_rep {
        inline void operator()(const BurgerProgram::Array& context1, const BurgerProgram::IndexContainer& context2, BurgerProgram::Array& state1, BurgerProgram::Int& state2) {
            return __while_loop2_23.repeat(context1, context2, state1, state2);
        };
    };

    static BurgerProgram::_bmb_plus_rep bmb_plus_rep;
    struct _bmb_sub {
        inline void operator()(const BurgerProgram::Array& a, const BurgerProgram::IndexContainer& ix_space, BurgerProgram::Array& b, BurgerProgram::Int& c) {
            BurgerProgram::Index ix = BurgerProgram::get_index_ixc(ix_space, c);
            BurgerProgram::Int32 new_value = BurgerProgram::binary_sub(BurgerProgram::unwrap_scalar(BurgerProgram::get(a, ix)), BurgerProgram::unwrap_scalar(BurgerProgram::get(b, ix)));
            BurgerProgram::set(b, ix, new_value);
            c = BurgerProgram::elem_int(BurgerProgram::binary_add(BurgerProgram::int_elem(c), BurgerProgram::one()));
        };
    };

    static BurgerProgram::_bmb_sub bmb_sub;
    struct _bmb_sub_rep {
        inline void operator()(const BurgerProgram::Array& context1, const BurgerProgram::IndexContainer& context2, BurgerProgram::Array& state1, BurgerProgram::Int& state2) {
            return __while_loop2_24.repeat(context1, context2, state1, state2);
        };
    };

    static BurgerProgram::_bmb_sub_rep bmb_sub_rep;
    struct _cat {
        inline BurgerProgram::Array operator()(const BurgerProgram::Array& array1, const BurgerProgram::Array& array2) {
            BurgerProgram::Int32 take_a1 = BurgerProgram::int_elem(BurgerProgram::get_shape_elem(array1, BurgerProgram::elem_int(BurgerProgram::zero())));
            BurgerProgram::Int32 take_a2 = BurgerProgram::int_elem(BurgerProgram::get_shape_elem(array2, BurgerProgram::elem_int(BurgerProgram::zero())));
            BurgerProgram::Shape drop_a1 = BurgerProgram::drop_shape_elem(array1, BurgerProgram::elem_int(BurgerProgram::zero()));
            BurgerProgram::Shape result_shape = BurgerProgram::cat_shape(BurgerProgram::create_shape1(BurgerProgram::elem_int(BurgerProgram::binary_add(take_a1, take_a2))), drop_a1);
            BurgerProgram::Array res = BurgerProgram::create_array(result_shape);
            BurgerProgram::Int counter = BurgerProgram::elem_int(BurgerProgram::zero());
            BurgerProgram::cat_repeat(array1, array2, counter, res);
            return res;
        };
    };

    static BurgerProgram::_cat cat;
    struct _cat_body {
        inline void operator()(const BurgerProgram::Array& array1, const BurgerProgram::Array& array2, BurgerProgram::Int& counter, BurgerProgram::Array& res) {
            BurgerProgram::Int32 s_0 = BurgerProgram::int_elem(BurgerProgram::total(array1));
            if (BurgerProgram::lt(BurgerProgram::int_elem(counter), s_0))
            {
                BurgerProgram::set(res, counter, BurgerProgram::unwrap_scalar(BurgerProgram::get(array1, counter)));
                counter = BurgerProgram::elem_int(BurgerProgram::binary_add(BurgerProgram::int_elem(counter), BurgerProgram::one()));
            }
            else
            {
                BurgerProgram::Int ix = BurgerProgram::elem_int(BurgerProgram::binary_sub(BurgerProgram::int_elem(counter), s_0));
                BurgerProgram::set(res, counter, BurgerProgram::unwrap_scalar(BurgerProgram::get(array2, ix)));
                counter = BurgerProgram::elem_int(BurgerProgram::binary_add(BurgerProgram::int_elem(counter), BurgerProgram::one()));
            }
        };
    };

    static BurgerProgram::_cat_body cat_body;
    struct _cat_cond {
        inline bool operator()(const BurgerProgram::Array& array1, const BurgerProgram::Array& array2, const BurgerProgram::Int& counter, const BurgerProgram::Array& res) {
            BurgerProgram::Int32 upper_bound = BurgerProgram::int_elem(BurgerProgram::total(res));
            return BurgerProgram::lt(BurgerProgram::int_elem(counter), upper_bound);
        };
    };

private:
    static while_loop2_2<BurgerProgram::Array, BurgerProgram::Array, BurgerProgram::Int, BurgerProgram::Array, BurgerProgram::_cat_body, BurgerProgram::_cat_cond> __while_loop2_20;
public:
    static BurgerProgram::_cat_cond cat_cond;
    struct _cat_repeat {
        inline void operator()(const BurgerProgram::Array& context1, const BurgerProgram::Array& context2, BurgerProgram::Int& state1, BurgerProgram::Array& state2) {
            return __while_loop2_20.repeat(context1, context2, state1, state2);
        };
    };

    static BurgerProgram::_cat_repeat cat_repeat;
    struct _cat_vec {
        inline BurgerProgram::Array operator()(const BurgerProgram::Array& vector1, const BurgerProgram::Array& vector2) {
            BurgerProgram::Shape res_shape = BurgerProgram::create_shape1(BurgerProgram::elem_int(BurgerProgram::binary_add(BurgerProgram::int_elem(BurgerProgram::total(vector1)), BurgerProgram::int_elem(BurgerProgram::total(vector2)))));
            BurgerProgram::Array res = BurgerProgram::create_array(res_shape);
            BurgerProgram::Int counter = BurgerProgram::elem_int(BurgerProgram::zero());
            BurgerProgram::cat_vec_repeat(vector1, vector2, res, counter);
            return res;
        };
    };

    static BurgerProgram::_cat_vec cat_vec;
    struct _cat_vec_body {
        inline void operator()(const BurgerProgram::Array& v1, const BurgerProgram::Array& v2, BurgerProgram::Array& res, BurgerProgram::Int& counter) {
            BurgerProgram::Int32 v1_bound = BurgerProgram::int_elem(BurgerProgram::total(v1));
            BurgerProgram::Index ix;
            if (BurgerProgram::lt(BurgerProgram::int_elem(counter), BurgerProgram::int_elem(BurgerProgram::total(v1))))
            {
                ix = BurgerProgram::create_index1(counter);
                BurgerProgram::set(res, ix, BurgerProgram::unwrap_scalar(BurgerProgram::get(v1, ix)));
            }
            else
            {
                ix = BurgerProgram::create_index1(BurgerProgram::elem_int(BurgerProgram::binary_sub(BurgerProgram::int_elem(counter), v1_bound)));
                BurgerProgram::Index res_ix = BurgerProgram::create_index1(counter);
                BurgerProgram::set(res, res_ix, BurgerProgram::unwrap_scalar(BurgerProgram::get(v2, ix)));
            }
            counter = BurgerProgram::elem_int(BurgerProgram::binary_add(BurgerProgram::int_elem(counter), BurgerProgram::one()));
        };
    };

    static BurgerProgram::_cat_vec_body cat_vec_body;
    struct _cat_vec_cond {
        inline bool operator()(const BurgerProgram::Array& v1, const BurgerProgram::Array& v2, const BurgerProgram::Array& res, const BurgerProgram::Int& counter) {
            return BurgerProgram::lt(BurgerProgram::int_elem(counter), BurgerProgram::int_elem(BurgerProgram::total(res)));
        };
    };

private:
    static while_loop2_2<BurgerProgram::Array, BurgerProgram::Array, BurgerProgram::Array, BurgerProgram::Int, BurgerProgram::_cat_vec_body, BurgerProgram::_cat_vec_cond> __while_loop2_2;
public:
    static BurgerProgram::_cat_vec_cond cat_vec_cond;
    struct _cat_vec_repeat {
        inline void operator()(const BurgerProgram::Array& context1, const BurgerProgram::Array& context2, BurgerProgram::Array& state1, BurgerProgram::Int& state2) {
            return __while_loop2_2.repeat(context1, context2, state1, state2);
        };
    };

    static BurgerProgram::_cat_vec_repeat cat_vec_repeat;
    struct _circular_padl {
        inline BurgerProgram::PaddedArray operator()(const BurgerProgram::PaddedArray& a, const BurgerProgram::Int& ix) {
            BurgerProgram::Array padding = BurgerProgram::get(a, BurgerProgram::create_index1(ix));
            BurgerProgram::Shape reshape_shape = BurgerProgram::cat_shape(BurgerProgram::create_shape1(BurgerProgram::elem_int(BurgerProgram::one())), BurgerProgram::shape(padding));
            BurgerProgram::Array reshaped_padding = BurgerProgram::reshape(padding, reshape_shape);
            BurgerProgram::Array catenated_array = BurgerProgram::cat(reshaped_padding, BurgerProgram::padded_to_unpadded(a));
            BurgerProgram::Shape unpadded_shape = BurgerProgram::shape(a);
            BurgerProgram::Shape padded_shape = BurgerProgram::shape(catenated_array);
            BurgerProgram::PaddedArray res = BurgerProgram::create_padded_array(unpadded_shape, padded_shape, catenated_array);
            return res;
        };
        inline BurgerProgram::PaddedArray operator()(const BurgerProgram::Array& a, const BurgerProgram::Int& ix) {
            BurgerProgram::Array padding = BurgerProgram::get(a, BurgerProgram::create_index1(ix));
            BurgerProgram::Shape reshape_shape = BurgerProgram::cat_shape(BurgerProgram::create_shape1(BurgerProgram::elem_int(BurgerProgram::one())), BurgerProgram::shape(padding));
            BurgerProgram::Array reshaped_padding = BurgerProgram::reshape(padding, reshape_shape);
            BurgerProgram::Array catenated_array = BurgerProgram::cat(reshaped_padding, a);
            BurgerProgram::Shape unpadded_shape = BurgerProgram::shape(a);
            BurgerProgram::Shape padded_shape = BurgerProgram::shape(catenated_array);
            BurgerProgram::PaddedArray res = BurgerProgram::create_padded_array(unpadded_shape, padded_shape, catenated_array);
            return res;
        };
    };

    static BurgerProgram::_circular_padl circular_padl;
    struct _circular_padr {
        inline BurgerProgram::PaddedArray operator()(const BurgerProgram::PaddedArray& a, const BurgerProgram::Int& ix) {
            BurgerProgram::Shape unpadded_shape = BurgerProgram::shape(a);
            BurgerProgram::Array padding = BurgerProgram::get(a, BurgerProgram::create_index1(ix));
            BurgerProgram::Shape reshape_shape = BurgerProgram::cat_shape(BurgerProgram::create_shape1(BurgerProgram::elem_int(BurgerProgram::one())), BurgerProgram::shape(padding));
            BurgerProgram::Array reshaped_padding = BurgerProgram::reshape(padding, reshape_shape);
            BurgerProgram::Array catenated_array = BurgerProgram::cat(BurgerProgram::padded_to_unpadded(a), reshaped_padding);
            BurgerProgram::Shape padded_shape = BurgerProgram::shape(catenated_array);
            BurgerProgram::PaddedArray res = BurgerProgram::create_padded_array(unpadded_shape, padded_shape, catenated_array);
            return res;
        };
        inline BurgerProgram::PaddedArray operator()(const BurgerProgram::Array& a, const BurgerProgram::Int& ix) {
            BurgerProgram::Array padding = BurgerProgram::get(a, BurgerProgram::create_index1(ix));
            BurgerProgram::Shape reshape_shape = BurgerProgram::cat_shape(BurgerProgram::create_shape1(BurgerProgram::elem_int(BurgerProgram::one())), BurgerProgram::shape(padding));
            BurgerProgram::Array reshaped_padding = BurgerProgram::reshape(padding, reshape_shape);
            BurgerProgram::Array catenated_array = BurgerProgram::cat(a, reshaped_padding);
            BurgerProgram::Shape unpadded_shape = BurgerProgram::shape(a);
            BurgerProgram::Shape padded_shape = BurgerProgram::shape(catenated_array);
            BurgerProgram::PaddedArray res = BurgerProgram::create_padded_array(unpadded_shape, padded_shape, catenated_array);
            return res;
        };
    };

    static BurgerProgram::_circular_padr circular_padr;
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
    struct _create_partial_indices {
        inline BurgerProgram::IndexContainer operator()(const BurgerProgram::Array& a, const BurgerProgram::Int& i) {
            return __array.create_partial_indices(a, i);
        };
    };

    static BurgerProgram::_create_partial_indices create_partial_indices;
    struct _create_total_indices {
        inline BurgerProgram::IndexContainer operator()(const BurgerProgram::Array& a) {
            return __array.create_total_indices(a);
        };
        inline BurgerProgram::IndexContainer operator()(const BurgerProgram::PaddedArray& a) {
            return __array.create_total_indices(a);
        };
    };

    static BurgerProgram::_create_total_indices create_total_indices;
    struct _dim {
        inline BurgerProgram::Int operator()(const BurgerProgram::Array& a) {
            return __array.dim(a);
        };
    };

    static BurgerProgram::_dim dim;
    struct _div {
        inline BurgerProgram::Array operator()(const BurgerProgram::Array& a, const BurgerProgram::Array& b) {
            BurgerProgram::IndexContainer ix_space = BurgerProgram::create_total_indices(a);
            BurgerProgram::Array b_upd = b;
            BurgerProgram::Int counter = BurgerProgram::elem_int(BurgerProgram::zero());
            BurgerProgram::bmb_div_rep(a, ix_space, b_upd, counter);
            return b;
        };
        inline BurgerProgram::Int32 operator()(const BurgerProgram::Int32& a, const BurgerProgram::Int32& b) {
            return __int32_utils.div(a, b);
        };
    };

    static BurgerProgram::_div div;
    struct _drop {
        inline BurgerProgram::Array operator()(const BurgerProgram::Int& i, const BurgerProgram::Array& a) {
            BurgerProgram::Shape drop_sh_0 = BurgerProgram::drop_shape_elem(a, BurgerProgram::elem_int(BurgerProgram::zero()));
            BurgerProgram::Array res_array;
            if (BurgerProgram::lt(BurgerProgram::int_elem(i), BurgerProgram::zero()))
            {
                BurgerProgram::Int take_value = BurgerProgram::elem_int(BurgerProgram::binary_add(BurgerProgram::int_elem(BurgerProgram::get_shape_elem(a, BurgerProgram::elem_int(BurgerProgram::zero()))), BurgerProgram::int_elem(i)));
                res_array = BurgerProgram::take(take_value, a);
            }
            else
            {
                res_array = BurgerProgram::get(a, BurgerProgram::create_index1(i));
                res_array = BurgerProgram::reshape(res_array, BurgerProgram::cat_shape(BurgerProgram::create_shape1(BurgerProgram::elem_int(BurgerProgram::one())), BurgerProgram::shape(res_array)));
                BurgerProgram::Int c = BurgerProgram::elem_int(BurgerProgram::binary_add(BurgerProgram::int_elem(i), BurgerProgram::one()));
                BurgerProgram::drop_repeat(a, i, res_array, c);
            }
            return res_array;
        };
    };

    static BurgerProgram::_drop drop;
    struct _drop_body {
        inline void operator()(const BurgerProgram::Array& a, const BurgerProgram::Int& i, BurgerProgram::Array& res, BurgerProgram::Int& c) {
            BurgerProgram::Array slice = BurgerProgram::get(a, BurgerProgram::create_index1(c));
            slice = BurgerProgram::reshape(slice, BurgerProgram::cat_shape(BurgerProgram::create_shape1(BurgerProgram::elem_int(BurgerProgram::one())), BurgerProgram::shape(slice)));
            res = BurgerProgram::cat(res, slice);
            c = BurgerProgram::elem_int(BurgerProgram::binary_add(BurgerProgram::int_elem(c), BurgerProgram::one()));
        };
    };

    static BurgerProgram::_drop_body drop_body;
    struct _drop_cond {
        inline bool operator()(const BurgerProgram::Array& a, const BurgerProgram::Int& i, const BurgerProgram::Array& res, const BurgerProgram::Int& c) {
            return BurgerProgram::lt(BurgerProgram::int_elem(c), BurgerProgram::int_elem(BurgerProgram::get_shape_elem(a, BurgerProgram::elem_int(BurgerProgram::zero()))));
        };
    };

private:
    static while_loop2_2<BurgerProgram::Array, BurgerProgram::Int, BurgerProgram::Array, BurgerProgram::Int, BurgerProgram::_drop_body, BurgerProgram::_drop_cond> __while_loop2_27;
public:
    static BurgerProgram::_drop_cond drop_cond;
    struct _drop_repeat {
        inline void operator()(const BurgerProgram::Array& context1, const BurgerProgram::Int& context2, BurgerProgram::Array& state1, BurgerProgram::Int& state2) {
            return __while_loop2_27.repeat(context1, context2, state1, state2);
        };
    };

    static BurgerProgram::_drop_repeat drop_repeat;
    struct _drop_shape_elem {
        inline BurgerProgram::Shape operator()(const BurgerProgram::Array& a, const BurgerProgram::Int& i) {
            return __array.drop_shape_elem(a, i);
        };
    };

    static BurgerProgram::_drop_shape_elem drop_shape_elem;
    struct _get {
        inline BurgerProgram::Array operator()(const BurgerProgram::Array& a, const BurgerProgram::Int& ix) {
            return __array.get(a, ix);
        };
        inline BurgerProgram::Array operator()(const BurgerProgram::Array& a, const BurgerProgram::Index& ix) {
            return __array.get(a, ix);
        };
        inline BurgerProgram::Array operator()(const BurgerProgram::PaddedArray& a, const BurgerProgram::Index& ix) {
            return __array.get(a, ix);
        };
    };

    static BurgerProgram::_get get;
    struct _get_shape_elem {
        inline BurgerProgram::Int operator()(const BurgerProgram::Array& a, const BurgerProgram::Int& i) {
            return __array.get_shape_elem(a, i);
        };
    };

    static BurgerProgram::_get_shape_elem get_shape_elem;
    struct _mapped_ops_cond {
        inline bool operator()(const BurgerProgram::Array& a, const BurgerProgram::IndexContainer& ix_space, const BurgerProgram::Array& b, const BurgerProgram::Int& c) {
            return BurgerProgram::lt(BurgerProgram::int_elem(c), BurgerProgram::int_elem(BurgerProgram::total(ix_space)));
        };
    };

private:
    static while_loop2_2<BurgerProgram::Array, BurgerProgram::IndexContainer, BurgerProgram::Array, BurgerProgram::Int, BurgerProgram::_bmb_div, BurgerProgram::_mapped_ops_cond> __while_loop2_21;
    static while_loop2_2<BurgerProgram::Array, BurgerProgram::IndexContainer, BurgerProgram::Array, BurgerProgram::Int, BurgerProgram::_bmb_mul, BurgerProgram::_mapped_ops_cond> __while_loop2_22;
    static while_loop2_2<BurgerProgram::Array, BurgerProgram::IndexContainer, BurgerProgram::Array, BurgerProgram::Int, BurgerProgram::_bmb_plus, BurgerProgram::_mapped_ops_cond> __while_loop2_23;
    static while_loop2_2<BurgerProgram::Array, BurgerProgram::IndexContainer, BurgerProgram::Array, BurgerProgram::Int, BurgerProgram::_bmb_sub, BurgerProgram::_mapped_ops_cond> __while_loop2_24;
public:
    static BurgerProgram::_mapped_ops_cond mapped_ops_cond;
    struct _mul {
        inline BurgerProgram::Array operator()(const BurgerProgram::Array& a, const BurgerProgram::Array& b) {
            BurgerProgram::IndexContainer ix_space = BurgerProgram::create_total_indices(a);
            BurgerProgram::Array b_upd = b;
            BurgerProgram::Int counter = BurgerProgram::elem_int(BurgerProgram::zero());
            BurgerProgram::bmb_mul_rep(a, ix_space, b_upd, counter);
            return b;
        };
        inline BurgerProgram::Int32 operator()(const BurgerProgram::Int32& a, const BurgerProgram::Int32& b) {
            return __int32_utils.mul(a, b);
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
    struct _reshape {
        inline BurgerProgram::Array operator()(const BurgerProgram::Array& input_array, const BurgerProgram::Shape& s) {
            BurgerProgram::Array new_array = BurgerProgram::create_array(s);
            BurgerProgram::Int counter = BurgerProgram::elem_int(BurgerProgram::zero());
            BurgerProgram::reshape_repeat(input_array, new_array, counter);
            return new_array;
        };
    };

    static BurgerProgram::_reshape reshape;
    struct _reshape_body {
        inline void operator()(const BurgerProgram::Array& old_array, BurgerProgram::Array& new_array, BurgerProgram::Int& counter) {
            BurgerProgram::set(new_array, counter, BurgerProgram::unwrap_scalar(BurgerProgram::get(old_array, counter)));
            counter = BurgerProgram::elem_int(BurgerProgram::binary_add(BurgerProgram::int_elem(counter), BurgerProgram::one()));
        };
    };

    static BurgerProgram::_reshape_body reshape_body;
    struct _reshape_cond {
        inline bool operator()(const BurgerProgram::Array& old_array, const BurgerProgram::Array& new_array, const BurgerProgram::Int& counter) {
            return BurgerProgram::lt(BurgerProgram::int_elem(counter), BurgerProgram::int_elem(BurgerProgram::total(new_array)));
        };
    };

private:
    static while_loop1_2<BurgerProgram::Array, BurgerProgram::Array, BurgerProgram::Int, BurgerProgram::_reshape_body, BurgerProgram::_reshape_cond> __while_loop1_2;
public:
    static BurgerProgram::_reshape_cond reshape_cond;
    struct _reshape_repeat {
        inline void operator()(const BurgerProgram::Array& context1, BurgerProgram::Array& state1, BurgerProgram::Int& state2) {
            return __while_loop1_2.repeat(context1, state1, state2);
        };
    };

    static BurgerProgram::_reshape_repeat reshape_repeat;
    struct _reverse {
        inline BurgerProgram::Array operator()(const BurgerProgram::Array& a) {
            BurgerProgram::Array res_array = BurgerProgram::create_array(BurgerProgram::shape(a));
            BurgerProgram::IndexContainer valid_indices = BurgerProgram::create_total_indices(res_array);
            BurgerProgram::Int counter = BurgerProgram::elem_int(BurgerProgram::zero());
            BurgerProgram::reverse_repeat(a, valid_indices, res_array, counter);
            return res_array;
        };
    };

    static BurgerProgram::_reverse reverse;
    struct _reverse_body {
        inline void operator()(const BurgerProgram::Array& input, const BurgerProgram::IndexContainer& indices, BurgerProgram::Array& res, BurgerProgram::Int& c) {
            BurgerProgram::Index ix = BurgerProgram::get_index_ixc(indices, c);
            BurgerProgram::Int32 elem = BurgerProgram::unwrap_scalar(BurgerProgram::get(input, ix));
            BurgerProgram::Int sh_0 = BurgerProgram::get_shape_elem(input, BurgerProgram::elem_int(BurgerProgram::zero()));
            BurgerProgram::Int ix_0 = BurgerProgram::get_index_elem(ix, BurgerProgram::elem_int(BurgerProgram::zero()));
            BurgerProgram::Int32 new_ix_0 = BurgerProgram::binary_sub(BurgerProgram::int_elem(sh_0), BurgerProgram::binary_add(BurgerProgram::int_elem(ix_0), BurgerProgram::one()));
            BurgerProgram::Index new_ix = BurgerProgram::cat_index(BurgerProgram::create_index1(BurgerProgram::elem_int(new_ix_0)), BurgerProgram::drop_index_elem(ix, BurgerProgram::elem_int(BurgerProgram::zero())));
            BurgerProgram::set(res, new_ix, elem);
            c = BurgerProgram::elem_int(BurgerProgram::binary_add(BurgerProgram::int_elem(c), BurgerProgram::one()));
        };
    };

    static BurgerProgram::_reverse_body reverse_body;
    struct _reverse_cond {
        inline bool operator()(const BurgerProgram::Array& input, const BurgerProgram::IndexContainer& indices, const BurgerProgram::Array& res, const BurgerProgram::Int& c) {
            return BurgerProgram::lt(BurgerProgram::int_elem(c), BurgerProgram::int_elem(BurgerProgram::total(indices)));
        };
    };

private:
    static while_loop2_2<BurgerProgram::Array, BurgerProgram::IndexContainer, BurgerProgram::Array, BurgerProgram::Int, BurgerProgram::_reverse_body, BurgerProgram::_reverse_cond> __while_loop2_25;
public:
    static BurgerProgram::_reverse_cond reverse_cond;
    struct _reverse_repeat {
        inline void operator()(const BurgerProgram::Array& context1, const BurgerProgram::IndexContainer& context2, BurgerProgram::Array& state1, BurgerProgram::Int& state2) {
            return __while_loop2_25.repeat(context1, context2, state1, state2);
        };
    };

    static BurgerProgram::_reverse_repeat reverse_repeat;
    struct _rotate {
        inline BurgerProgram::Array operator()(const BurgerProgram::Int& sigma, const BurgerProgram::Int& ax, const BurgerProgram::Array& a) {
            BurgerProgram::IndexContainer ix_space = BurgerProgram::create_partial_indices(a, ax);
            BurgerProgram::Int c = BurgerProgram::elem_int(BurgerProgram::zero());
            BurgerProgram::Index i = BurgerProgram::get_index_ixc(ix_space, c);
            BurgerProgram::Array res;
            if (BurgerProgram::lt(BurgerProgram::zero(), BurgerProgram::int_elem(sigma)))
            {
                BurgerProgram::Array e1 = BurgerProgram::drop(sigma, BurgerProgram::get(a, i));
                BurgerProgram::Array e2 = BurgerProgram::take(sigma, BurgerProgram::get(a, i));
                res = BurgerProgram::cat(e1, e2);
            }
            else
            {
                BurgerProgram::Array e1 = BurgerProgram::take(BurgerProgram::elem_int(BurgerProgram::unary_sub(BurgerProgram::int_elem(sigma))), BurgerProgram::get(a, i));
                BurgerProgram::Array e2 = BurgerProgram::drop(BurgerProgram::elem_int(BurgerProgram::unary_sub(BurgerProgram::int_elem(sigma))), BurgerProgram::get(a, i));
                BurgerProgram::print_array(e1);
                BurgerProgram::print_array(e2);
                res = BurgerProgram::cat(e1, e2);
            }
            return res;
        };
    };

    static BurgerProgram::_rotate rotate;
    struct _set {
        inline void operator()(BurgerProgram::Array& a, const BurgerProgram::Int& ix, const BurgerProgram::Int32& e) {
            return __array.set(a, ix, e);
        };
        inline void operator()(BurgerProgram::Array& a, const BurgerProgram::Index& ix, const BurgerProgram::Int32& e) {
            return __array.set(a, ix, e);
        };
        inline void operator()(BurgerProgram::PaddedArray& a, const BurgerProgram::Index& ix, const BurgerProgram::Int32& e) {
            return __array.set(a, ix, e);
        };
    };

    static BurgerProgram::_set set;
    struct _shape {
        inline BurgerProgram::Shape operator()(const BurgerProgram::Array& a) {
            return __array.shape(a);
        };
        inline BurgerProgram::Shape operator()(const BurgerProgram::PaddedArray& a) {
            return __array.shape(a);
        };
    };

    static BurgerProgram::_shape shape;
    struct _take {
        inline BurgerProgram::Array operator()(const BurgerProgram::Int& i, const BurgerProgram::Array& a) {
            BurgerProgram::Shape drop_sh_0 = BurgerProgram::drop_shape_elem(a, BurgerProgram::elem_int(BurgerProgram::zero()));
            BurgerProgram::Array res_array;
            if (BurgerProgram::lt(BurgerProgram::int_elem(i), BurgerProgram::zero()))
            {
                BurgerProgram::Int drop_value = BurgerProgram::elem_int(BurgerProgram::binary_add(BurgerProgram::int_elem(BurgerProgram::get_shape_elem(a, BurgerProgram::elem_int(BurgerProgram::zero()))), BurgerProgram::int_elem(i)));
                res_array = BurgerProgram::drop(drop_value, a);
            }
            else
            {
                res_array = BurgerProgram::get(a, BurgerProgram::create_index1(BurgerProgram::elem_int(BurgerProgram::zero())));
                res_array = BurgerProgram::reshape(res_array, BurgerProgram::cat_shape(BurgerProgram::create_shape1(BurgerProgram::elem_int(BurgerProgram::one())), BurgerProgram::shape(res_array)));
                BurgerProgram::Int c = BurgerProgram::elem_int(BurgerProgram::one());
                BurgerProgram::take_repeat(a, i, res_array, c);
            }
            return res_array;
        };
    };

    static BurgerProgram::_take take;
    struct _take_body {
        inline void operator()(const BurgerProgram::Array& a, const BurgerProgram::Int& i, BurgerProgram::Array& res, BurgerProgram::Int& c) {
            BurgerProgram::Index ix = BurgerProgram::create_index1(c);
            BurgerProgram::Array slice = BurgerProgram::get(a, ix);
            slice = BurgerProgram::reshape(slice, BurgerProgram::cat_shape(BurgerProgram::create_shape1(BurgerProgram::elem_int(BurgerProgram::one())), BurgerProgram::shape(slice)));
            res = BurgerProgram::cat(res, slice);
            c = BurgerProgram::elem_int(BurgerProgram::binary_add(BurgerProgram::int_elem(c), BurgerProgram::one()));
        };
    };

    static BurgerProgram::_take_body take_body;
    struct _take_cond {
        inline bool operator()(const BurgerProgram::Array& a, const BurgerProgram::Int& i, const BurgerProgram::Array& res, const BurgerProgram::Int& c) {
            return BurgerProgram::lt(BurgerProgram::int_elem(c), BurgerProgram::int_elem(i));
        };
    };

private:
    static while_loop2_2<BurgerProgram::Array, BurgerProgram::Int, BurgerProgram::Array, BurgerProgram::Int, BurgerProgram::_take_body, BurgerProgram::_take_cond> __while_loop2_28;
public:
    static BurgerProgram::_take_cond take_cond;
    struct _take_repeat {
        inline void operator()(const BurgerProgram::Array& context1, const BurgerProgram::Int& context2, BurgerProgram::Array& state1, BurgerProgram::Int& state2) {
            return __while_loop2_28.repeat(context1, context2, state1, state2);
        };
    };

    static BurgerProgram::_take_repeat take_repeat;
    struct _test_array3_2_2 {
        inline BurgerProgram::Array operator()() {
            return __array.test_array3_2_2();
        };
    };

    static BurgerProgram::_test_array3_2_2 test_array3_2_2;
    struct _test_array3_2_2F {
        inline BurgerProgram::Array operator()() {
            return __array.test_array3_2_2F();
        };
    };

    static BurgerProgram::_test_array3_2_2F test_array3_2_2F;
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
        inline BurgerProgram::Int operator()(const BurgerProgram::Array& a) {
            return __array.total(a);
        };
        inline BurgerProgram::Int operator()(const BurgerProgram::Shape& s) {
            return __array.total(s);
        };
        inline BurgerProgram::Int operator()(const BurgerProgram::IndexContainer& ixc) {
            return __array.total(ixc);
        };
    };

    static BurgerProgram::_total total;
    struct _transpose {
        inline BurgerProgram::Array operator()(const BurgerProgram::Array& a) {
            BurgerProgram::Array transposed_array = BurgerProgram::create_array(BurgerProgram::reverse_shape(BurgerProgram::shape(a)));
            BurgerProgram::IndexContainer ix_space = BurgerProgram::create_total_indices(transposed_array);
            BurgerProgram::Int counter = BurgerProgram::elem_int(BurgerProgram::zero());
            BurgerProgram::transpose_repeat(a, ix_space, transposed_array, counter);
            return transposed_array;
        };
        inline BurgerProgram::PaddedArray operator()(const BurgerProgram::PaddedArray& a) {
            BurgerProgram::Array reshaped_array = BurgerProgram::create_array(BurgerProgram::padded_shape(a));
            BurgerProgram::PaddedArray transposed_array = BurgerProgram::create_padded_array(BurgerProgram::reverse_shape(BurgerProgram::shape(a)), BurgerProgram::reverse_shape(BurgerProgram::padded_shape(a)), reshaped_array);
            BurgerProgram::IndexContainer ix_space = BurgerProgram::create_total_indices(transposed_array);
            BurgerProgram::Int counter = BurgerProgram::elem_int(BurgerProgram::zero());
            BurgerProgram::padded_transpose_repeat(a, ix_space, transposed_array, counter);
            return transposed_array;
        };
    };

    static BurgerProgram::_transpose transpose;
    struct _transpose_body {
        inline void operator()(const BurgerProgram::Array& a, const BurgerProgram::IndexContainer& ixc, BurgerProgram::Array& res, BurgerProgram::Int& c) {
            BurgerProgram::Index current_ix = BurgerProgram::get_index_ixc(ixc, c);
            BurgerProgram::Int32 current_element = BurgerProgram::unwrap_scalar(BurgerProgram::get(a, BurgerProgram::reverse_index(current_ix)));
            BurgerProgram::set(res, current_ix, current_element);
            c = BurgerProgram::elem_int(BurgerProgram::binary_add(BurgerProgram::int_elem(c), BurgerProgram::one()));
        };
    };

    static BurgerProgram::_transpose_body transpose_body;
    struct _transpose_repeat {
        inline void operator()(const BurgerProgram::Array& context1, const BurgerProgram::IndexContainer& context2, BurgerProgram::Array& state1, BurgerProgram::Int& state2) {
            return __while_loop2_26.repeat(context1, context2, state1, state2);
        };
    };

    static BurgerProgram::_transpose_repeat transpose_repeat;
    struct _unaryMap {
        inline void operator()(const BurgerProgram::Array& a, const BurgerProgram::Index& ix) {
            assert((BurgerProgram::unwrap_scalar(BurgerProgram::get(BurgerProgram::unary_sub(a), ix))) == (BurgerProgram::unary_sub(BurgerProgram::unwrap_scalar(BurgerProgram::get(a, ix)))));
        };
    };

    static BurgerProgram::_unaryMap unaryMap;
    struct _unary_sub {
        inline BurgerProgram::Array operator()(const BurgerProgram::Array& a) {
            BurgerProgram::IndexContainer ix_space = BurgerProgram::create_total_indices(a);
            BurgerProgram::Array a_upd = a;
            BurgerProgram::Int counter = BurgerProgram::elem_int(BurgerProgram::zero());
            BurgerProgram::unary_sub_repeat(ix_space, a_upd, counter);
            return a_upd;
        };
        inline BurgerProgram::Int32 operator()(const BurgerProgram::Int32& a) {
            return __int32_utils.unary_sub(a);
        };
    };

    static BurgerProgram::_unary_sub unary_sub;
    struct _unary_sub_body {
        inline void operator()(const BurgerProgram::IndexContainer& ix_space, BurgerProgram::Array& a, BurgerProgram::Int& c) {
            BurgerProgram::Index ix = BurgerProgram::get_index_ixc(ix_space, c);
            BurgerProgram::Int32 new_value = BurgerProgram::unary_sub(BurgerProgram::unwrap_scalar(BurgerProgram::get(a, ix)));
            BurgerProgram::set(a, ix, new_value);
            c = BurgerProgram::elem_int(BurgerProgram::binary_add(BurgerProgram::int_elem(c), BurgerProgram::one()));
        };
    };

    static BurgerProgram::_unary_sub_body unary_sub_body;
    struct _unary_sub_cond {
        inline bool operator()(const BurgerProgram::IndexContainer& ix_space, const BurgerProgram::Array& a, const BurgerProgram::Int& c) {
            return BurgerProgram::lt(BurgerProgram::int_elem(c), BurgerProgram::int_elem(BurgerProgram::total(ix_space)));
        };
    };

private:
    static while_loop1_2<BurgerProgram::IndexContainer, BurgerProgram::Array, BurgerProgram::Int, BurgerProgram::_unary_sub_body, BurgerProgram::_unary_sub_cond> __while_loop1_20;
public:
    static BurgerProgram::_unary_sub_cond unary_sub_cond;
    struct _unary_sub_repeat {
        inline void operator()(const BurgerProgram::IndexContainer& context1, BurgerProgram::Array& state1, BurgerProgram::Int& state2) {
            return __while_loop1_20.repeat(context1, state1, state2);
        };
    };

    static BurgerProgram::_unary_sub_repeat unary_sub_repeat;
    struct _unwrap_scalar {
        inline BurgerProgram::Int32 operator()(const BurgerProgram::Array& a) {
            return __array.unwrap_scalar(a);
        };
    };

    static BurgerProgram::_unwrap_scalar unwrap_scalar;
    struct _upper_bound {
        inline bool operator()(const BurgerProgram::Array& a, const BurgerProgram::IndexContainer& i, const BurgerProgram::Array& res, const BurgerProgram::Int& c) {
            return BurgerProgram::lt(BurgerProgram::int_elem(c), BurgerProgram::int_elem(BurgerProgram::total(i)));
        };
    };

private:
    static while_loop2_2<BurgerProgram::Array, BurgerProgram::IndexContainer, BurgerProgram::Array, BurgerProgram::Int, BurgerProgram::_transpose_body, BurgerProgram::_upper_bound> __while_loop2_26;
public:
    static BurgerProgram::_upper_bound upper_bound;
};
} // examples
} // moa
} // mg_src
} // moa_cpp

namespace examples {
namespace moa {
namespace mg_src {
namespace moa_cpp {
struct Float64Arrays {
private:
    static float64_utils __float64_utils;
public:
    typedef float64_utils::Float64 Float64;
    typedef array<Float64Arrays::Float64>::Index Index;
    struct _cat_index {
        inline Float64Arrays::Index operator()(const Float64Arrays::Index& i, const Float64Arrays::Index& j) {
            return __array.cat_index(i, j);
        };
    };

    static Float64Arrays::_cat_index cat_index;
    struct _print_index {
        inline void operator()(const Float64Arrays::Index& i) {
            return __array.print_index(i);
        };
    };

    static Float64Arrays::_print_index print_index;
    struct _reverse_index {
        inline Float64Arrays::Index operator()(const Float64Arrays::Index& ix) {
            return __array.reverse_index(ix);
        };
    };

    static Float64Arrays::_reverse_index reverse_index;
    struct _test_index {
        inline Float64Arrays::Index operator()() {
            return __array.test_index();
        };
    };

    static Float64Arrays::_test_index test_index;
    typedef array<Float64Arrays::Float64>::IndexContainer IndexContainer;
    struct _print_index_container {
        inline void operator()(const Float64Arrays::IndexContainer& i) {
            return __array.print_index_container(i);
        };
    };

    static Float64Arrays::_print_index_container print_index_container;
    typedef array<Float64Arrays::Float64>::Int Int;
    struct _create_index1 {
        inline Float64Arrays::Index operator()(const Float64Arrays::Int& a) {
            return __array.create_index1(a);
        };
    };

    static Float64Arrays::_create_index1 create_index1;
    struct _create_index2 {
        inline Float64Arrays::Index operator()(const Float64Arrays::Int& a, const Float64Arrays::Int& b) {
            return __array.create_index2(a, b);
        };
    };

    static Float64Arrays::_create_index2 create_index2;
    struct _create_index3 {
        inline Float64Arrays::Index operator()(const Float64Arrays::Int& a, const Float64Arrays::Int& b, const Float64Arrays::Int& c) {
            return __array.create_index3(a, b, c);
        };
    };

    static Float64Arrays::_create_index3 create_index3;
    struct _drop_index_elem {
        inline Float64Arrays::Index operator()(const Float64Arrays::Index& ix, const Float64Arrays::Int& i) {
            return __array.drop_index_elem(ix, i);
        };
    };

    static Float64Arrays::_drop_index_elem drop_index_elem;
    struct _get_index_elem {
        inline Float64Arrays::Int operator()(const Float64Arrays::Index& ix, const Float64Arrays::Int& i) {
            return __array.get_index_elem(ix, i);
        };
    };

    static Float64Arrays::_get_index_elem get_index_elem;
    struct _get_index_ixc {
        inline Float64Arrays::Index operator()(const Float64Arrays::IndexContainer& ixc, const Float64Arrays::Int& ix) {
            return __array.get_index_ixc(ixc, ix);
        };
    };

    static Float64Arrays::_get_index_ixc get_index_ixc;
    struct _print_int {
        inline void operator()(const Float64Arrays::Int& u) {
            return __array.print_int(u);
        };
    };

    static Float64Arrays::_print_int print_int;
    typedef array<Float64Arrays::Float64>::PaddedArray PaddedArray;
    struct _padded_dim {
        inline Float64Arrays::Int operator()(const Float64Arrays::PaddedArray& a) {
            return __array.padded_dim(a);
        };
    };

    static Float64Arrays::_padded_dim padded_dim;
    struct _padded_get_shape_elem {
        inline Float64Arrays::Int operator()(const Float64Arrays::PaddedArray& a, const Float64Arrays::Int& i) {
            return __array.padded_get_shape_elem(a, i);
        };
    };

    static Float64Arrays::_padded_get_shape_elem padded_get_shape_elem;
    struct _padded_total {
        inline Float64Arrays::Int operator()(const Float64Arrays::PaddedArray& a) {
            return __array.padded_total(a);
        };
    };

    static Float64Arrays::_padded_total padded_total;
    struct _padded_transpose_body {
        inline void operator()(const Float64Arrays::PaddedArray& a, const Float64Arrays::IndexContainer& ixc, Float64Arrays::PaddedArray& res, Float64Arrays::Int& c) {
            Float64Arrays::Index current_ix = Float64Arrays::get_index_ixc(ixc, c);
            Float64Arrays::Float64 current_element = Float64Arrays::unwrap_scalar(Float64Arrays::get(a, Float64Arrays::reverse_index(current_ix)));
            Float64Arrays::set(res, current_ix, current_element);
            c = Float64Arrays::elem_int(Float64Arrays::binary_add(Float64Arrays::int_elem(c), Float64Arrays::one()));
        };
    };

    static Float64Arrays::_padded_transpose_body padded_transpose_body;
    struct _padded_transpose_repeat {
        inline void operator()(const Float64Arrays::PaddedArray& context1, const Float64Arrays::IndexContainer& context2, Float64Arrays::PaddedArray& state1, Float64Arrays::Int& state2) {
            return __while_loop2_29.repeat(context1, context2, state1, state2);
        };
    };

    static Float64Arrays::_padded_transpose_repeat padded_transpose_repeat;
    struct _padded_upper_bound {
        inline bool operator()(const Float64Arrays::PaddedArray& a, const Float64Arrays::IndexContainer& i, const Float64Arrays::PaddedArray& res, const Float64Arrays::Int& c) {
            return Float64Arrays::lt(Float64Arrays::int_elem(c), Float64Arrays::int_elem(Float64Arrays::total(i)));
        };
    };

private:
    static while_loop2_2<Float64Arrays::PaddedArray, Float64Arrays::IndexContainer, Float64Arrays::PaddedArray, Float64Arrays::Int, Float64Arrays::_padded_transpose_body, Float64Arrays::_padded_upper_bound> __while_loop2_29;
public:
    static Float64Arrays::_padded_upper_bound padded_upper_bound;
    struct _print_parray {
        inline void operator()(const Float64Arrays::PaddedArray& a) {
            return __array.print_parray(a);
        };
    };

    static Float64Arrays::_print_parray print_parray;
    typedef array<Float64Arrays::Float64>::Shape Shape;
    struct _cat_shape {
        inline Float64Arrays::Shape operator()(const Float64Arrays::Shape& a, const Float64Arrays::Shape& b) {
            return __array.cat_shape(a, b);
        };
    };

    static Float64Arrays::_cat_shape cat_shape;
    struct _create_shape1 {
        inline Float64Arrays::Shape operator()(const Float64Arrays::Int& a) {
            return __array.create_shape1(a);
        };
    };

    static Float64Arrays::_create_shape1 create_shape1;
    struct _create_shape2 {
        inline Float64Arrays::Shape operator()(const Float64Arrays::Int& a, const Float64Arrays::Int& b) {
            return __array.create_shape2(a, b);
        };
    };

    static Float64Arrays::_create_shape2 create_shape2;
    struct _create_shape3 {
        inline Float64Arrays::Shape operator()(const Float64Arrays::Int& a, const Float64Arrays::Int& b, const Float64Arrays::Int& c) {
            return __array.create_shape3(a, b, c);
        };
    };

    static Float64Arrays::_create_shape3 create_shape3;
    struct _padded_drop_shape_elem {
        inline Float64Arrays::Shape operator()(const Float64Arrays::PaddedArray& a, const Float64Arrays::Int& i) {
            return __array.padded_drop_shape_elem(a, i);
        };
    };

    static Float64Arrays::_padded_drop_shape_elem padded_drop_shape_elem;
    struct _padded_shape {
        inline Float64Arrays::Shape operator()(const Float64Arrays::PaddedArray& a) {
            return __array.padded_shape(a);
        };
    };

    static Float64Arrays::_padded_shape padded_shape;
    struct _print_shape {
        inline void operator()(const Float64Arrays::Shape& sh) {
            return __array.print_shape(sh);
        };
    };

    static Float64Arrays::_print_shape print_shape;
    struct _reverse_shape {
        inline Float64Arrays::Shape operator()(const Float64Arrays::Shape& s) {
            return __array.reverse_shape(s);
        };
    };

    static Float64Arrays::_reverse_shape reverse_shape;
private:
    static array<Float64Arrays::Float64> __array;
public:
    struct _abs {
        inline Float64Arrays::Float64 operator()(const Float64Arrays::Float64& a) {
            return __float64_utils.abs(a);
        };
    };

    static Float64Arrays::_abs abs;
    struct _elem_int {
        inline Float64Arrays::Int operator()(const Float64Arrays::Float64& a) {
            return __array.elem_int(a);
        };
    };

    static Float64Arrays::_elem_int elem_int;
    struct _eq {
        inline bool operator()(const Float64Arrays::Float64& a, const Float64Arrays::Float64& b) {
            return __float64_utils.eq(a, b);
        };
    };

    static Float64Arrays::_eq eq;
    struct _int_elem {
        inline Float64Arrays::Float64 operator()(const Float64Arrays::Int& a) {
            return __array.int_elem(a);
        };
    };

    static Float64Arrays::_int_elem int_elem;
    struct _le {
        inline bool operator()(const Float64Arrays::Float64& a, const Float64Arrays::Float64& b) {
            return __float64_utils.le(a, b);
        };
    };

    static Float64Arrays::_le le;
    struct _lt {
        inline bool operator()(const Float64Arrays::Float64& a, const Float64Arrays::Float64& b) {
            return __float64_utils.lt(a, b);
        };
    };

    static Float64Arrays::_lt lt;
    struct _one {
        inline Float64Arrays::Float64 operator()() {
            return __float64_utils.one();
        };
    };

    static Float64Arrays::_one one;
    struct _print_element {
        inline void operator()(const Float64Arrays::Float64& e) {
            return __array.print_element(e);
        };
    };

    static Float64Arrays::_print_element print_element;
    struct _zero {
        inline Float64Arrays::Float64 operator()() {
            return __float64_utils.zero();
        };
    };

    static Float64Arrays::_zero zero;
    typedef array<Float64Arrays::Float64>::Array Array;
    struct _binaryMap {
        inline void operator()(const Float64Arrays::Array& a, const Float64Arrays::Array& b, const Float64Arrays::Index& ix) {
            assert((Float64Arrays::unwrap_scalar(Float64Arrays::get(Float64Arrays::binary_add(a, b), ix))) == (Float64Arrays::binary_add(Float64Arrays::unwrap_scalar(Float64Arrays::get(a, ix)), Float64Arrays::unwrap_scalar(Float64Arrays::get(b, ix)))));
            assert((Float64Arrays::unwrap_scalar(Float64Arrays::get(Float64Arrays::binary_sub(a, b), ix))) == (Float64Arrays::binary_sub(Float64Arrays::unwrap_scalar(Float64Arrays::get(a, ix)), Float64Arrays::unwrap_scalar(Float64Arrays::get(b, ix)))));
            assert((Float64Arrays::unwrap_scalar(Float64Arrays::get(Float64Arrays::mul(a, b), ix))) == (Float64Arrays::mul(Float64Arrays::unwrap_scalar(Float64Arrays::get(a, ix)), Float64Arrays::unwrap_scalar(Float64Arrays::get(b, ix)))));
            assert((Float64Arrays::unwrap_scalar(Float64Arrays::get(Float64Arrays::div(a, b), ix))) == (Float64Arrays::div(Float64Arrays::unwrap_scalar(Float64Arrays::get(a, ix)), Float64Arrays::unwrap_scalar(Float64Arrays::get(b, ix)))));
        };
    };

    static Float64Arrays::_binaryMap binaryMap;
    struct _binary_add {
        inline Float64Arrays::Array operator()(const Float64Arrays::Array& a, const Float64Arrays::Array& b) {
            Float64Arrays::IndexContainer ix_space = Float64Arrays::create_total_indices(a);
            Float64Arrays::Array b_upd = b;
            Float64Arrays::Int counter = Float64Arrays::elem_int(Float64Arrays::zero());
            Float64Arrays::bmb_plus_rep(a, ix_space, b_upd, counter);
            return b;
        };
        inline Float64Arrays::Float64 operator()(const Float64Arrays::Float64& a, const Float64Arrays::Float64& b) {
            return __float64_utils.binary_add(a, b);
        };
    };

    static Float64Arrays::_binary_add binary_add;
    struct _binary_sub {
        inline Float64Arrays::Array operator()(const Float64Arrays::Array& a, const Float64Arrays::Array& b) {
            Float64Arrays::IndexContainer ix_space = Float64Arrays::create_total_indices(a);
            Float64Arrays::Array b_upd = b;
            Float64Arrays::Int counter = Float64Arrays::elem_int(Float64Arrays::zero());
            Float64Arrays::bmb_sub_rep(a, ix_space, b_upd, counter);
            return b;
        };
        inline Float64Arrays::Float64 operator()(const Float64Arrays::Float64& a, const Float64Arrays::Float64& b) {
            return __float64_utils.binary_sub(a, b);
        };
    };

    static Float64Arrays::_binary_sub binary_sub;
    struct _bmb_div {
        inline void operator()(const Float64Arrays::Array& a, const Float64Arrays::IndexContainer& ix_space, Float64Arrays::Array& b, Float64Arrays::Int& c) {
            Float64Arrays::Index ix = Float64Arrays::get_index_ixc(ix_space, c);
            Float64Arrays::Float64 new_value = Float64Arrays::div(Float64Arrays::unwrap_scalar(Float64Arrays::get(a, ix)), Float64Arrays::unwrap_scalar(Float64Arrays::get(b, ix)));
            Float64Arrays::set(b, ix, new_value);
            c = Float64Arrays::elem_int(Float64Arrays::binary_add(Float64Arrays::int_elem(c), Float64Arrays::one()));
        };
    };

    static Float64Arrays::_bmb_div bmb_div;
    struct _bmb_div_rep {
        inline void operator()(const Float64Arrays::Array& context1, const Float64Arrays::IndexContainer& context2, Float64Arrays::Array& state1, Float64Arrays::Int& state2) {
            return __while_loop2_21.repeat(context1, context2, state1, state2);
        };
    };

    static Float64Arrays::_bmb_div_rep bmb_div_rep;
    struct _bmb_mul {
        inline void operator()(const Float64Arrays::Array& a, const Float64Arrays::IndexContainer& ix_space, Float64Arrays::Array& b, Float64Arrays::Int& c) {
            Float64Arrays::Index ix = Float64Arrays::get_index_ixc(ix_space, c);
            Float64Arrays::Float64 new_value = Float64Arrays::mul(Float64Arrays::unwrap_scalar(Float64Arrays::get(a, ix)), Float64Arrays::unwrap_scalar(Float64Arrays::get(b, ix)));
            Float64Arrays::set(b, ix, new_value);
            c = Float64Arrays::elem_int(Float64Arrays::binary_add(Float64Arrays::int_elem(c), Float64Arrays::one()));
        };
    };

    static Float64Arrays::_bmb_mul bmb_mul;
    struct _bmb_mul_rep {
        inline void operator()(const Float64Arrays::Array& context1, const Float64Arrays::IndexContainer& context2, Float64Arrays::Array& state1, Float64Arrays::Int& state2) {
            return __while_loop2_22.repeat(context1, context2, state1, state2);
        };
    };

    static Float64Arrays::_bmb_mul_rep bmb_mul_rep;
    struct _bmb_plus {
        inline void operator()(const Float64Arrays::Array& a, const Float64Arrays::IndexContainer& ix_space, Float64Arrays::Array& b, Float64Arrays::Int& c) {
            Float64Arrays::Index ix = Float64Arrays::get_index_ixc(ix_space, c);
            Float64Arrays::Float64 new_value = Float64Arrays::binary_add(Float64Arrays::unwrap_scalar(Float64Arrays::get(a, ix)), Float64Arrays::unwrap_scalar(Float64Arrays::get(b, ix)));
            Float64Arrays::set(b, ix, new_value);
            c = Float64Arrays::elem_int(Float64Arrays::binary_add(Float64Arrays::int_elem(c), Float64Arrays::one()));
        };
    };

    static Float64Arrays::_bmb_plus bmb_plus;
    struct _bmb_plus_rep {
        inline void operator()(const Float64Arrays::Array& context1, const Float64Arrays::IndexContainer& context2, Float64Arrays::Array& state1, Float64Arrays::Int& state2) {
            return __while_loop2_23.repeat(context1, context2, state1, state2);
        };
    };

    static Float64Arrays::_bmb_plus_rep bmb_plus_rep;
    struct _bmb_sub {
        inline void operator()(const Float64Arrays::Array& a, const Float64Arrays::IndexContainer& ix_space, Float64Arrays::Array& b, Float64Arrays::Int& c) {
            Float64Arrays::Index ix = Float64Arrays::get_index_ixc(ix_space, c);
            Float64Arrays::Float64 new_value = Float64Arrays::binary_sub(Float64Arrays::unwrap_scalar(Float64Arrays::get(a, ix)), Float64Arrays::unwrap_scalar(Float64Arrays::get(b, ix)));
            Float64Arrays::set(b, ix, new_value);
            c = Float64Arrays::elem_int(Float64Arrays::binary_add(Float64Arrays::int_elem(c), Float64Arrays::one()));
        };
    };

    static Float64Arrays::_bmb_sub bmb_sub;
    struct _bmb_sub_rep {
        inline void operator()(const Float64Arrays::Array& context1, const Float64Arrays::IndexContainer& context2, Float64Arrays::Array& state1, Float64Arrays::Int& state2) {
            return __while_loop2_24.repeat(context1, context2, state1, state2);
        };
    };

    static Float64Arrays::_bmb_sub_rep bmb_sub_rep;
    struct _cat {
        inline Float64Arrays::Array operator()(const Float64Arrays::Array& array1, const Float64Arrays::Array& array2) {
            Float64Arrays::Float64 take_a1 = Float64Arrays::int_elem(Float64Arrays::get_shape_elem(array1, Float64Arrays::elem_int(Float64Arrays::zero())));
            Float64Arrays::Float64 take_a2 = Float64Arrays::int_elem(Float64Arrays::get_shape_elem(array2, Float64Arrays::elem_int(Float64Arrays::zero())));
            Float64Arrays::Shape drop_a1 = Float64Arrays::drop_shape_elem(array1, Float64Arrays::elem_int(Float64Arrays::zero()));
            Float64Arrays::Shape result_shape = Float64Arrays::cat_shape(Float64Arrays::create_shape1(Float64Arrays::elem_int(Float64Arrays::binary_add(take_a1, take_a2))), drop_a1);
            Float64Arrays::Array res = Float64Arrays::create_array(result_shape);
            Float64Arrays::Int counter = Float64Arrays::elem_int(Float64Arrays::zero());
            Float64Arrays::cat_repeat(array1, array2, counter, res);
            return res;
        };
    };

    static Float64Arrays::_cat cat;
    struct _cat_body {
        inline void operator()(const Float64Arrays::Array& array1, const Float64Arrays::Array& array2, Float64Arrays::Int& counter, Float64Arrays::Array& res) {
            Float64Arrays::Float64 s_0 = Float64Arrays::int_elem(Float64Arrays::total(array1));
            if (Float64Arrays::lt(Float64Arrays::int_elem(counter), s_0))
            {
                Float64Arrays::set(res, counter, Float64Arrays::unwrap_scalar(Float64Arrays::get(array1, counter)));
                counter = Float64Arrays::elem_int(Float64Arrays::binary_add(Float64Arrays::int_elem(counter), Float64Arrays::one()));
            }
            else
            {
                Float64Arrays::Int ix = Float64Arrays::elem_int(Float64Arrays::binary_sub(Float64Arrays::int_elem(counter), s_0));
                Float64Arrays::set(res, counter, Float64Arrays::unwrap_scalar(Float64Arrays::get(array2, ix)));
                counter = Float64Arrays::elem_int(Float64Arrays::binary_add(Float64Arrays::int_elem(counter), Float64Arrays::one()));
            }
        };
    };

    static Float64Arrays::_cat_body cat_body;
    struct _cat_cond {
        inline bool operator()(const Float64Arrays::Array& array1, const Float64Arrays::Array& array2, const Float64Arrays::Int& counter, const Float64Arrays::Array& res) {
            Float64Arrays::Float64 upper_bound = Float64Arrays::int_elem(Float64Arrays::total(res));
            return Float64Arrays::lt(Float64Arrays::int_elem(counter), upper_bound);
        };
    };

private:
    static while_loop2_2<Float64Arrays::Array, Float64Arrays::Array, Float64Arrays::Int, Float64Arrays::Array, Float64Arrays::_cat_body, Float64Arrays::_cat_cond> __while_loop2_20;
public:
    static Float64Arrays::_cat_cond cat_cond;
    struct _cat_repeat {
        inline void operator()(const Float64Arrays::Array& context1, const Float64Arrays::Array& context2, Float64Arrays::Int& state1, Float64Arrays::Array& state2) {
            return __while_loop2_20.repeat(context1, context2, state1, state2);
        };
    };

    static Float64Arrays::_cat_repeat cat_repeat;
    struct _cat_vec {
        inline Float64Arrays::Array operator()(const Float64Arrays::Array& vector1, const Float64Arrays::Array& vector2) {
            Float64Arrays::Shape res_shape = Float64Arrays::create_shape1(Float64Arrays::elem_int(Float64Arrays::binary_add(Float64Arrays::int_elem(Float64Arrays::total(vector1)), Float64Arrays::int_elem(Float64Arrays::total(vector2)))));
            Float64Arrays::Array res = Float64Arrays::create_array(res_shape);
            Float64Arrays::Int counter = Float64Arrays::elem_int(Float64Arrays::zero());
            Float64Arrays::cat_vec_repeat(vector1, vector2, res, counter);
            return res;
        };
    };

    static Float64Arrays::_cat_vec cat_vec;
    struct _cat_vec_body {
        inline void operator()(const Float64Arrays::Array& v1, const Float64Arrays::Array& v2, Float64Arrays::Array& res, Float64Arrays::Int& counter) {
            Float64Arrays::Float64 v1_bound = Float64Arrays::int_elem(Float64Arrays::total(v1));
            Float64Arrays::Index ix;
            if (Float64Arrays::lt(Float64Arrays::int_elem(counter), Float64Arrays::int_elem(Float64Arrays::total(v1))))
            {
                ix = Float64Arrays::create_index1(counter);
                Float64Arrays::set(res, ix, Float64Arrays::unwrap_scalar(Float64Arrays::get(v1, ix)));
            }
            else
            {
                ix = Float64Arrays::create_index1(Float64Arrays::elem_int(Float64Arrays::binary_sub(Float64Arrays::int_elem(counter), v1_bound)));
                Float64Arrays::Index res_ix = Float64Arrays::create_index1(counter);
                Float64Arrays::set(res, res_ix, Float64Arrays::unwrap_scalar(Float64Arrays::get(v2, ix)));
            }
            counter = Float64Arrays::elem_int(Float64Arrays::binary_add(Float64Arrays::int_elem(counter), Float64Arrays::one()));
        };
    };

    static Float64Arrays::_cat_vec_body cat_vec_body;
    struct _cat_vec_cond {
        inline bool operator()(const Float64Arrays::Array& v1, const Float64Arrays::Array& v2, const Float64Arrays::Array& res, const Float64Arrays::Int& counter) {
            return Float64Arrays::lt(Float64Arrays::int_elem(counter), Float64Arrays::int_elem(Float64Arrays::total(res)));
        };
    };

private:
    static while_loop2_2<Float64Arrays::Array, Float64Arrays::Array, Float64Arrays::Array, Float64Arrays::Int, Float64Arrays::_cat_vec_body, Float64Arrays::_cat_vec_cond> __while_loop2_2;
public:
    static Float64Arrays::_cat_vec_cond cat_vec_cond;
    struct _cat_vec_repeat {
        inline void operator()(const Float64Arrays::Array& context1, const Float64Arrays::Array& context2, Float64Arrays::Array& state1, Float64Arrays::Int& state2) {
            return __while_loop2_2.repeat(context1, context2, state1, state2);
        };
    };

    static Float64Arrays::_cat_vec_repeat cat_vec_repeat;
    struct _circular_padl {
        inline Float64Arrays::PaddedArray operator()(const Float64Arrays::PaddedArray& a, const Float64Arrays::Int& ix) {
            Float64Arrays::Array padding = Float64Arrays::get(a, Float64Arrays::create_index1(ix));
            Float64Arrays::Shape reshape_shape = Float64Arrays::cat_shape(Float64Arrays::create_shape1(Float64Arrays::elem_int(Float64Arrays::one())), Float64Arrays::shape(padding));
            Float64Arrays::Array reshaped_padding = Float64Arrays::reshape(padding, reshape_shape);
            Float64Arrays::Array catenated_array = Float64Arrays::cat(reshaped_padding, Float64Arrays::padded_to_unpadded(a));
            Float64Arrays::Shape unpadded_shape = Float64Arrays::shape(a);
            Float64Arrays::Shape padded_shape = Float64Arrays::shape(catenated_array);
            Float64Arrays::PaddedArray res = Float64Arrays::create_padded_array(unpadded_shape, padded_shape, catenated_array);
            return res;
        };
        inline Float64Arrays::PaddedArray operator()(const Float64Arrays::Array& a, const Float64Arrays::Int& ix) {
            Float64Arrays::Array padding = Float64Arrays::get(a, Float64Arrays::create_index1(ix));
            Float64Arrays::Shape reshape_shape = Float64Arrays::cat_shape(Float64Arrays::create_shape1(Float64Arrays::elem_int(Float64Arrays::one())), Float64Arrays::shape(padding));
            Float64Arrays::Array reshaped_padding = Float64Arrays::reshape(padding, reshape_shape);
            Float64Arrays::Array catenated_array = Float64Arrays::cat(reshaped_padding, a);
            Float64Arrays::Shape unpadded_shape = Float64Arrays::shape(a);
            Float64Arrays::Shape padded_shape = Float64Arrays::shape(catenated_array);
            Float64Arrays::PaddedArray res = Float64Arrays::create_padded_array(unpadded_shape, padded_shape, catenated_array);
            return res;
        };
    };

    static Float64Arrays::_circular_padl circular_padl;
    struct _circular_padr {
        inline Float64Arrays::PaddedArray operator()(const Float64Arrays::PaddedArray& a, const Float64Arrays::Int& ix) {
            Float64Arrays::Shape unpadded_shape = Float64Arrays::shape(a);
            Float64Arrays::Array padding = Float64Arrays::get(a, Float64Arrays::create_index1(ix));
            Float64Arrays::Shape reshape_shape = Float64Arrays::cat_shape(Float64Arrays::create_shape1(Float64Arrays::elem_int(Float64Arrays::one())), Float64Arrays::shape(padding));
            Float64Arrays::Array reshaped_padding = Float64Arrays::reshape(padding, reshape_shape);
            Float64Arrays::Array catenated_array = Float64Arrays::cat(Float64Arrays::padded_to_unpadded(a), reshaped_padding);
            Float64Arrays::Shape padded_shape = Float64Arrays::shape(catenated_array);
            Float64Arrays::PaddedArray res = Float64Arrays::create_padded_array(unpadded_shape, padded_shape, catenated_array);
            return res;
        };
        inline Float64Arrays::PaddedArray operator()(const Float64Arrays::Array& a, const Float64Arrays::Int& ix) {
            Float64Arrays::Array padding = Float64Arrays::get(a, Float64Arrays::create_index1(ix));
            Float64Arrays::Shape reshape_shape = Float64Arrays::cat_shape(Float64Arrays::create_shape1(Float64Arrays::elem_int(Float64Arrays::one())), Float64Arrays::shape(padding));
            Float64Arrays::Array reshaped_padding = Float64Arrays::reshape(padding, reshape_shape);
            Float64Arrays::Array catenated_array = Float64Arrays::cat(a, reshaped_padding);
            Float64Arrays::Shape unpadded_shape = Float64Arrays::shape(a);
            Float64Arrays::Shape padded_shape = Float64Arrays::shape(catenated_array);
            Float64Arrays::PaddedArray res = Float64Arrays::create_padded_array(unpadded_shape, padded_shape, catenated_array);
            return res;
        };
    };

    static Float64Arrays::_circular_padr circular_padr;
    struct _create_array {
        inline Float64Arrays::Array operator()(const Float64Arrays::Shape& sh) {
            return __array.create_array(sh);
        };
    };

    static Float64Arrays::_create_array create_array;
    struct _create_padded_array {
        inline Float64Arrays::PaddedArray operator()(const Float64Arrays::Shape& unpadded_shape, const Float64Arrays::Shape& padded_shape, const Float64Arrays::Array& padded_array) {
            return __array.create_padded_array(unpadded_shape, padded_shape, padded_array);
        };
    };

    static Float64Arrays::_create_padded_array create_padded_array;
    struct _create_partial_indices {
        inline Float64Arrays::IndexContainer operator()(const Float64Arrays::Array& a, const Float64Arrays::Int& i) {
            return __array.create_partial_indices(a, i);
        };
    };

    static Float64Arrays::_create_partial_indices create_partial_indices;
    struct _create_total_indices {
        inline Float64Arrays::IndexContainer operator()(const Float64Arrays::Array& a) {
            return __array.create_total_indices(a);
        };
        inline Float64Arrays::IndexContainer operator()(const Float64Arrays::PaddedArray& a) {
            return __array.create_total_indices(a);
        };
    };

    static Float64Arrays::_create_total_indices create_total_indices;
    struct _dim {
        inline Float64Arrays::Int operator()(const Float64Arrays::Array& a) {
            return __array.dim(a);
        };
    };

    static Float64Arrays::_dim dim;
    struct _div {
        inline Float64Arrays::Array operator()(const Float64Arrays::Array& a, const Float64Arrays::Array& b) {
            Float64Arrays::IndexContainer ix_space = Float64Arrays::create_total_indices(a);
            Float64Arrays::Array b_upd = b;
            Float64Arrays::Int counter = Float64Arrays::elem_int(Float64Arrays::zero());
            Float64Arrays::bmb_div_rep(a, ix_space, b_upd, counter);
            return b;
        };
        inline Float64Arrays::Float64 operator()(const Float64Arrays::Float64& a, const Float64Arrays::Float64& b) {
            return __float64_utils.div(a, b);
        };
    };

    static Float64Arrays::_div div;
    struct _drop {
        inline Float64Arrays::Array operator()(const Float64Arrays::Int& i, const Float64Arrays::Array& a) {
            Float64Arrays::Shape drop_sh_0 = Float64Arrays::drop_shape_elem(a, Float64Arrays::elem_int(Float64Arrays::zero()));
            Float64Arrays::Array res_array;
            if (Float64Arrays::lt(Float64Arrays::int_elem(i), Float64Arrays::zero()))
            {
                Float64Arrays::Int take_value = Float64Arrays::elem_int(Float64Arrays::binary_add(Float64Arrays::int_elem(Float64Arrays::get_shape_elem(a, Float64Arrays::elem_int(Float64Arrays::zero()))), Float64Arrays::int_elem(i)));
                res_array = Float64Arrays::take(take_value, a);
            }
            else
            {
                res_array = Float64Arrays::get(a, Float64Arrays::create_index1(i));
                res_array = Float64Arrays::reshape(res_array, Float64Arrays::cat_shape(Float64Arrays::create_shape1(Float64Arrays::elem_int(Float64Arrays::one())), Float64Arrays::shape(res_array)));
                Float64Arrays::Int c = Float64Arrays::elem_int(Float64Arrays::binary_add(Float64Arrays::int_elem(i), Float64Arrays::one()));
                Float64Arrays::drop_repeat(a, i, res_array, c);
            }
            return res_array;
        };
    };

    static Float64Arrays::_drop drop;
    struct _drop_body {
        inline void operator()(const Float64Arrays::Array& a, const Float64Arrays::Int& i, Float64Arrays::Array& res, Float64Arrays::Int& c) {
            Float64Arrays::Array slice = Float64Arrays::get(a, Float64Arrays::create_index1(c));
            slice = Float64Arrays::reshape(slice, Float64Arrays::cat_shape(Float64Arrays::create_shape1(Float64Arrays::elem_int(Float64Arrays::one())), Float64Arrays::shape(slice)));
            res = Float64Arrays::cat(res, slice);
            c = Float64Arrays::elem_int(Float64Arrays::binary_add(Float64Arrays::int_elem(c), Float64Arrays::one()));
        };
    };

    static Float64Arrays::_drop_body drop_body;
    struct _drop_cond {
        inline bool operator()(const Float64Arrays::Array& a, const Float64Arrays::Int& i, const Float64Arrays::Array& res, const Float64Arrays::Int& c) {
            return Float64Arrays::lt(Float64Arrays::int_elem(c), Float64Arrays::int_elem(Float64Arrays::get_shape_elem(a, Float64Arrays::elem_int(Float64Arrays::zero()))));
        };
    };

private:
    static while_loop2_2<Float64Arrays::Array, Float64Arrays::Int, Float64Arrays::Array, Float64Arrays::Int, Float64Arrays::_drop_body, Float64Arrays::_drop_cond> __while_loop2_27;
public:
    static Float64Arrays::_drop_cond drop_cond;
    struct _drop_repeat {
        inline void operator()(const Float64Arrays::Array& context1, const Float64Arrays::Int& context2, Float64Arrays::Array& state1, Float64Arrays::Int& state2) {
            return __while_loop2_27.repeat(context1, context2, state1, state2);
        };
    };

    static Float64Arrays::_drop_repeat drop_repeat;
    struct _drop_shape_elem {
        inline Float64Arrays::Shape operator()(const Float64Arrays::Array& a, const Float64Arrays::Int& i) {
            return __array.drop_shape_elem(a, i);
        };
    };

    static Float64Arrays::_drop_shape_elem drop_shape_elem;
    struct _get {
        inline Float64Arrays::Array operator()(const Float64Arrays::Array& a, const Float64Arrays::Int& ix) {
            return __array.get(a, ix);
        };
        inline Float64Arrays::Array operator()(const Float64Arrays::Array& a, const Float64Arrays::Index& ix) {
            return __array.get(a, ix);
        };
        inline Float64Arrays::Array operator()(const Float64Arrays::PaddedArray& a, const Float64Arrays::Index& ix) {
            return __array.get(a, ix);
        };
    };

    static Float64Arrays::_get get;
    struct _get_shape_elem {
        inline Float64Arrays::Int operator()(const Float64Arrays::Array& a, const Float64Arrays::Int& i) {
            return __array.get_shape_elem(a, i);
        };
    };

    static Float64Arrays::_get_shape_elem get_shape_elem;
    struct _mapped_ops_cond {
        inline bool operator()(const Float64Arrays::Array& a, const Float64Arrays::IndexContainer& ix_space, const Float64Arrays::Array& b, const Float64Arrays::Int& c) {
            return Float64Arrays::lt(Float64Arrays::int_elem(c), Float64Arrays::int_elem(Float64Arrays::total(ix_space)));
        };
    };

private:
    static while_loop2_2<Float64Arrays::Array, Float64Arrays::IndexContainer, Float64Arrays::Array, Float64Arrays::Int, Float64Arrays::_bmb_div, Float64Arrays::_mapped_ops_cond> __while_loop2_21;
    static while_loop2_2<Float64Arrays::Array, Float64Arrays::IndexContainer, Float64Arrays::Array, Float64Arrays::Int, Float64Arrays::_bmb_mul, Float64Arrays::_mapped_ops_cond> __while_loop2_22;
    static while_loop2_2<Float64Arrays::Array, Float64Arrays::IndexContainer, Float64Arrays::Array, Float64Arrays::Int, Float64Arrays::_bmb_plus, Float64Arrays::_mapped_ops_cond> __while_loop2_23;
    static while_loop2_2<Float64Arrays::Array, Float64Arrays::IndexContainer, Float64Arrays::Array, Float64Arrays::Int, Float64Arrays::_bmb_sub, Float64Arrays::_mapped_ops_cond> __while_loop2_24;
public:
    static Float64Arrays::_mapped_ops_cond mapped_ops_cond;
    struct _mul {
        inline Float64Arrays::Array operator()(const Float64Arrays::Array& a, const Float64Arrays::Array& b) {
            Float64Arrays::IndexContainer ix_space = Float64Arrays::create_total_indices(a);
            Float64Arrays::Array b_upd = b;
            Float64Arrays::Int counter = Float64Arrays::elem_int(Float64Arrays::zero());
            Float64Arrays::bmb_mul_rep(a, ix_space, b_upd, counter);
            return b;
        };
        inline Float64Arrays::Float64 operator()(const Float64Arrays::Float64& a, const Float64Arrays::Float64& b) {
            return __float64_utils.mul(a, b);
        };
    };

    static Float64Arrays::_mul mul;
    struct _padded_to_unpadded {
        inline Float64Arrays::Array operator()(const Float64Arrays::PaddedArray& a) {
            return __array.padded_to_unpadded(a);
        };
    };

    static Float64Arrays::_padded_to_unpadded padded_to_unpadded;
    struct _print_array {
        inline void operator()(const Float64Arrays::Array& a) {
            return __array.print_array(a);
        };
    };

    static Float64Arrays::_print_array print_array;
    struct _reshape {
        inline Float64Arrays::Array operator()(const Float64Arrays::Array& input_array, const Float64Arrays::Shape& s) {
            Float64Arrays::Array new_array = Float64Arrays::create_array(s);
            Float64Arrays::Int counter = Float64Arrays::elem_int(Float64Arrays::zero());
            Float64Arrays::reshape_repeat(input_array, new_array, counter);
            return new_array;
        };
    };

    static Float64Arrays::_reshape reshape;
    struct _reshape_body {
        inline void operator()(const Float64Arrays::Array& old_array, Float64Arrays::Array& new_array, Float64Arrays::Int& counter) {
            Float64Arrays::set(new_array, counter, Float64Arrays::unwrap_scalar(Float64Arrays::get(old_array, counter)));
            counter = Float64Arrays::elem_int(Float64Arrays::binary_add(Float64Arrays::int_elem(counter), Float64Arrays::one()));
        };
    };

    static Float64Arrays::_reshape_body reshape_body;
    struct _reshape_cond {
        inline bool operator()(const Float64Arrays::Array& old_array, const Float64Arrays::Array& new_array, const Float64Arrays::Int& counter) {
            return Float64Arrays::lt(Float64Arrays::int_elem(counter), Float64Arrays::int_elem(Float64Arrays::total(new_array)));
        };
    };

private:
    static while_loop1_2<Float64Arrays::Array, Float64Arrays::Array, Float64Arrays::Int, Float64Arrays::_reshape_body, Float64Arrays::_reshape_cond> __while_loop1_2;
public:
    static Float64Arrays::_reshape_cond reshape_cond;
    struct _reshape_repeat {
        inline void operator()(const Float64Arrays::Array& context1, Float64Arrays::Array& state1, Float64Arrays::Int& state2) {
            return __while_loop1_2.repeat(context1, state1, state2);
        };
    };

    static Float64Arrays::_reshape_repeat reshape_repeat;
    struct _reverse {
        inline Float64Arrays::Array operator()(const Float64Arrays::Array& a) {
            Float64Arrays::Array res_array = Float64Arrays::create_array(Float64Arrays::shape(a));
            Float64Arrays::IndexContainer valid_indices = Float64Arrays::create_total_indices(res_array);
            Float64Arrays::Int counter = Float64Arrays::elem_int(Float64Arrays::zero());
            Float64Arrays::reverse_repeat(a, valid_indices, res_array, counter);
            return res_array;
        };
    };

    static Float64Arrays::_reverse reverse;
    struct _reverse_body {
        inline void operator()(const Float64Arrays::Array& input, const Float64Arrays::IndexContainer& indices, Float64Arrays::Array& res, Float64Arrays::Int& c) {
            Float64Arrays::Index ix = Float64Arrays::get_index_ixc(indices, c);
            Float64Arrays::Float64 elem = Float64Arrays::unwrap_scalar(Float64Arrays::get(input, ix));
            Float64Arrays::Int sh_0 = Float64Arrays::get_shape_elem(input, Float64Arrays::elem_int(Float64Arrays::zero()));
            Float64Arrays::Int ix_0 = Float64Arrays::get_index_elem(ix, Float64Arrays::elem_int(Float64Arrays::zero()));
            Float64Arrays::Float64 new_ix_0 = Float64Arrays::binary_sub(Float64Arrays::int_elem(sh_0), Float64Arrays::binary_add(Float64Arrays::int_elem(ix_0), Float64Arrays::one()));
            Float64Arrays::Index new_ix = Float64Arrays::cat_index(Float64Arrays::create_index1(Float64Arrays::elem_int(new_ix_0)), Float64Arrays::drop_index_elem(ix, Float64Arrays::elem_int(Float64Arrays::zero())));
            Float64Arrays::set(res, new_ix, elem);
            c = Float64Arrays::elem_int(Float64Arrays::binary_add(Float64Arrays::int_elem(c), Float64Arrays::one()));
        };
    };

    static Float64Arrays::_reverse_body reverse_body;
    struct _reverse_cond {
        inline bool operator()(const Float64Arrays::Array& input, const Float64Arrays::IndexContainer& indices, const Float64Arrays::Array& res, const Float64Arrays::Int& c) {
            return Float64Arrays::lt(Float64Arrays::int_elem(c), Float64Arrays::int_elem(Float64Arrays::total(indices)));
        };
    };

private:
    static while_loop2_2<Float64Arrays::Array, Float64Arrays::IndexContainer, Float64Arrays::Array, Float64Arrays::Int, Float64Arrays::_reverse_body, Float64Arrays::_reverse_cond> __while_loop2_25;
public:
    static Float64Arrays::_reverse_cond reverse_cond;
    struct _reverse_repeat {
        inline void operator()(const Float64Arrays::Array& context1, const Float64Arrays::IndexContainer& context2, Float64Arrays::Array& state1, Float64Arrays::Int& state2) {
            return __while_loop2_25.repeat(context1, context2, state1, state2);
        };
    };

    static Float64Arrays::_reverse_repeat reverse_repeat;
    struct _rotate {
        inline Float64Arrays::Array operator()(const Float64Arrays::Int& sigma, const Float64Arrays::Int& ax, const Float64Arrays::Array& a) {
            Float64Arrays::IndexContainer ix_space = Float64Arrays::create_partial_indices(a, ax);
            Float64Arrays::Int c = Float64Arrays::elem_int(Float64Arrays::zero());
            Float64Arrays::Index i = Float64Arrays::get_index_ixc(ix_space, c);
            Float64Arrays::Array res;
            if (Float64Arrays::lt(Float64Arrays::zero(), Float64Arrays::int_elem(sigma)))
            {
                Float64Arrays::Array e1 = Float64Arrays::drop(sigma, Float64Arrays::get(a, i));
                Float64Arrays::Array e2 = Float64Arrays::take(sigma, Float64Arrays::get(a, i));
                res = Float64Arrays::cat(e1, e2);
            }
            else
            {
                Float64Arrays::Array e1 = Float64Arrays::take(Float64Arrays::elem_int(Float64Arrays::unary_sub(Float64Arrays::int_elem(sigma))), Float64Arrays::get(a, i));
                Float64Arrays::Array e2 = Float64Arrays::drop(Float64Arrays::elem_int(Float64Arrays::unary_sub(Float64Arrays::int_elem(sigma))), Float64Arrays::get(a, i));
                Float64Arrays::print_array(e1);
                Float64Arrays::print_array(e2);
                res = Float64Arrays::cat(e1, e2);
            }
            return res;
        };
    };

    static Float64Arrays::_rotate rotate;
    struct _set {
        inline void operator()(Float64Arrays::Array& a, const Float64Arrays::Int& ix, const Float64Arrays::Float64& e) {
            return __array.set(a, ix, e);
        };
        inline void operator()(Float64Arrays::Array& a, const Float64Arrays::Index& ix, const Float64Arrays::Float64& e) {
            return __array.set(a, ix, e);
        };
        inline void operator()(Float64Arrays::PaddedArray& a, const Float64Arrays::Index& ix, const Float64Arrays::Float64& e) {
            return __array.set(a, ix, e);
        };
    };

    static Float64Arrays::_set set;
    struct _shape {
        inline Float64Arrays::Shape operator()(const Float64Arrays::Array& a) {
            return __array.shape(a);
        };
        inline Float64Arrays::Shape operator()(const Float64Arrays::PaddedArray& a) {
            return __array.shape(a);
        };
    };

    static Float64Arrays::_shape shape;
    struct _take {
        inline Float64Arrays::Array operator()(const Float64Arrays::Int& i, const Float64Arrays::Array& a) {
            Float64Arrays::Shape drop_sh_0 = Float64Arrays::drop_shape_elem(a, Float64Arrays::elem_int(Float64Arrays::zero()));
            Float64Arrays::Array res_array;
            if (Float64Arrays::lt(Float64Arrays::int_elem(i), Float64Arrays::zero()))
            {
                Float64Arrays::Int drop_value = Float64Arrays::elem_int(Float64Arrays::binary_add(Float64Arrays::int_elem(Float64Arrays::get_shape_elem(a, Float64Arrays::elem_int(Float64Arrays::zero()))), Float64Arrays::int_elem(i)));
                res_array = Float64Arrays::drop(drop_value, a);
            }
            else
            {
                res_array = Float64Arrays::get(a, Float64Arrays::create_index1(Float64Arrays::elem_int(Float64Arrays::zero())));
                res_array = Float64Arrays::reshape(res_array, Float64Arrays::cat_shape(Float64Arrays::create_shape1(Float64Arrays::elem_int(Float64Arrays::one())), Float64Arrays::shape(res_array)));
                Float64Arrays::Int c = Float64Arrays::elem_int(Float64Arrays::one());
                Float64Arrays::take_repeat(a, i, res_array, c);
            }
            return res_array;
        };
    };

    static Float64Arrays::_take take;
    struct _take_body {
        inline void operator()(const Float64Arrays::Array& a, const Float64Arrays::Int& i, Float64Arrays::Array& res, Float64Arrays::Int& c) {
            Float64Arrays::Index ix = Float64Arrays::create_index1(c);
            Float64Arrays::Array slice = Float64Arrays::get(a, ix);
            slice = Float64Arrays::reshape(slice, Float64Arrays::cat_shape(Float64Arrays::create_shape1(Float64Arrays::elem_int(Float64Arrays::one())), Float64Arrays::shape(slice)));
            res = Float64Arrays::cat(res, slice);
            c = Float64Arrays::elem_int(Float64Arrays::binary_add(Float64Arrays::int_elem(c), Float64Arrays::one()));
        };
    };

    static Float64Arrays::_take_body take_body;
    struct _take_cond {
        inline bool operator()(const Float64Arrays::Array& a, const Float64Arrays::Int& i, const Float64Arrays::Array& res, const Float64Arrays::Int& c) {
            return Float64Arrays::lt(Float64Arrays::int_elem(c), Float64Arrays::int_elem(i));
        };
    };

private:
    static while_loop2_2<Float64Arrays::Array, Float64Arrays::Int, Float64Arrays::Array, Float64Arrays::Int, Float64Arrays::_take_body, Float64Arrays::_take_cond> __while_loop2_28;
public:
    static Float64Arrays::_take_cond take_cond;
    struct _take_repeat {
        inline void operator()(const Float64Arrays::Array& context1, const Float64Arrays::Int& context2, Float64Arrays::Array& state1, Float64Arrays::Int& state2) {
            return __while_loop2_28.repeat(context1, context2, state1, state2);
        };
    };

    static Float64Arrays::_take_repeat take_repeat;
    struct _test_array3_2_2 {
        inline Float64Arrays::Array operator()() {
            return __array.test_array3_2_2();
        };
    };

    static Float64Arrays::_test_array3_2_2 test_array3_2_2;
    struct _test_array3_2_2F {
        inline Float64Arrays::Array operator()() {
            return __array.test_array3_2_2F();
        };
    };

    static Float64Arrays::_test_array3_2_2F test_array3_2_2F;
    struct _test_array3_3 {
        inline Float64Arrays::Array operator()() {
            return __array.test_array3_3();
        };
    };

    static Float64Arrays::_test_array3_3 test_array3_3;
    struct _test_vector2 {
        inline Float64Arrays::Array operator()() {
            return __array.test_vector2();
        };
    };

    static Float64Arrays::_test_vector2 test_vector2;
    struct _test_vector3 {
        inline Float64Arrays::Array operator()() {
            return __array.test_vector3();
        };
    };

    static Float64Arrays::_test_vector3 test_vector3;
    struct _test_vector5 {
        inline Float64Arrays::Array operator()() {
            return __array.test_vector5();
        };
    };

    static Float64Arrays::_test_vector5 test_vector5;
    struct _total {
        inline Float64Arrays::Int operator()(const Float64Arrays::Array& a) {
            return __array.total(a);
        };
        inline Float64Arrays::Int operator()(const Float64Arrays::Shape& s) {
            return __array.total(s);
        };
        inline Float64Arrays::Int operator()(const Float64Arrays::IndexContainer& ixc) {
            return __array.total(ixc);
        };
    };

    static Float64Arrays::_total total;
    struct _transpose {
        inline Float64Arrays::Array operator()(const Float64Arrays::Array& a) {
            Float64Arrays::Array transposed_array = Float64Arrays::create_array(Float64Arrays::reverse_shape(Float64Arrays::shape(a)));
            Float64Arrays::IndexContainer ix_space = Float64Arrays::create_total_indices(transposed_array);
            Float64Arrays::Int counter = Float64Arrays::elem_int(Float64Arrays::zero());
            Float64Arrays::transpose_repeat(a, ix_space, transposed_array, counter);
            return transposed_array;
        };
        inline Float64Arrays::PaddedArray operator()(const Float64Arrays::PaddedArray& a) {
            Float64Arrays::Array reshaped_array = Float64Arrays::create_array(Float64Arrays::padded_shape(a));
            Float64Arrays::PaddedArray transposed_array = Float64Arrays::create_padded_array(Float64Arrays::reverse_shape(Float64Arrays::shape(a)), Float64Arrays::reverse_shape(Float64Arrays::padded_shape(a)), reshaped_array);
            Float64Arrays::IndexContainer ix_space = Float64Arrays::create_total_indices(transposed_array);
            Float64Arrays::Int counter = Float64Arrays::elem_int(Float64Arrays::zero());
            Float64Arrays::padded_transpose_repeat(a, ix_space, transposed_array, counter);
            return transposed_array;
        };
    };

    static Float64Arrays::_transpose transpose;
    struct _transpose_body {
        inline void operator()(const Float64Arrays::Array& a, const Float64Arrays::IndexContainer& ixc, Float64Arrays::Array& res, Float64Arrays::Int& c) {
            Float64Arrays::Index current_ix = Float64Arrays::get_index_ixc(ixc, c);
            Float64Arrays::Float64 current_element = Float64Arrays::unwrap_scalar(Float64Arrays::get(a, Float64Arrays::reverse_index(current_ix)));
            Float64Arrays::set(res, current_ix, current_element);
            c = Float64Arrays::elem_int(Float64Arrays::binary_add(Float64Arrays::int_elem(c), Float64Arrays::one()));
        };
    };

    static Float64Arrays::_transpose_body transpose_body;
    struct _transpose_repeat {
        inline void operator()(const Float64Arrays::Array& context1, const Float64Arrays::IndexContainer& context2, Float64Arrays::Array& state1, Float64Arrays::Int& state2) {
            return __while_loop2_26.repeat(context1, context2, state1, state2);
        };
    };

    static Float64Arrays::_transpose_repeat transpose_repeat;
    struct _unaryMap {
        inline void operator()(const Float64Arrays::Array& a, const Float64Arrays::Index& ix) {
            assert((Float64Arrays::unwrap_scalar(Float64Arrays::get(Float64Arrays::unary_sub(a), ix))) == (Float64Arrays::unary_sub(Float64Arrays::unwrap_scalar(Float64Arrays::get(a, ix)))));
        };
    };

    static Float64Arrays::_unaryMap unaryMap;
    struct _unary_sub {
        inline Float64Arrays::Array operator()(const Float64Arrays::Array& a) {
            Float64Arrays::IndexContainer ix_space = Float64Arrays::create_total_indices(a);
            Float64Arrays::Array a_upd = a;
            Float64Arrays::Int counter = Float64Arrays::elem_int(Float64Arrays::zero());
            Float64Arrays::unary_sub_repeat(ix_space, a_upd, counter);
            return a_upd;
        };
        inline Float64Arrays::Float64 operator()(const Float64Arrays::Float64& a) {
            return __float64_utils.unary_sub(a);
        };
    };

    static Float64Arrays::_unary_sub unary_sub;
    struct _unary_sub_body {
        inline void operator()(const Float64Arrays::IndexContainer& ix_space, Float64Arrays::Array& a, Float64Arrays::Int& c) {
            Float64Arrays::Index ix = Float64Arrays::get_index_ixc(ix_space, c);
            Float64Arrays::Float64 new_value = Float64Arrays::unary_sub(Float64Arrays::unwrap_scalar(Float64Arrays::get(a, ix)));
            Float64Arrays::set(a, ix, new_value);
            c = Float64Arrays::elem_int(Float64Arrays::binary_add(Float64Arrays::int_elem(c), Float64Arrays::one()));
        };
    };

    static Float64Arrays::_unary_sub_body unary_sub_body;
    struct _unary_sub_cond {
        inline bool operator()(const Float64Arrays::IndexContainer& ix_space, const Float64Arrays::Array& a, const Float64Arrays::Int& c) {
            return Float64Arrays::lt(Float64Arrays::int_elem(c), Float64Arrays::int_elem(Float64Arrays::total(ix_space)));
        };
    };

private:
    static while_loop1_2<Float64Arrays::IndexContainer, Float64Arrays::Array, Float64Arrays::Int, Float64Arrays::_unary_sub_body, Float64Arrays::_unary_sub_cond> __while_loop1_20;
public:
    static Float64Arrays::_unary_sub_cond unary_sub_cond;
    struct _unary_sub_repeat {
        inline void operator()(const Float64Arrays::IndexContainer& context1, Float64Arrays::Array& state1, Float64Arrays::Int& state2) {
            return __while_loop1_20.repeat(context1, state1, state2);
        };
    };

    static Float64Arrays::_unary_sub_repeat unary_sub_repeat;
    struct _unwrap_scalar {
        inline Float64Arrays::Float64 operator()(const Float64Arrays::Array& a) {
            return __array.unwrap_scalar(a);
        };
    };

    static Float64Arrays::_unwrap_scalar unwrap_scalar;
    struct _upper_bound {
        inline bool operator()(const Float64Arrays::Array& a, const Float64Arrays::IndexContainer& i, const Float64Arrays::Array& res, const Float64Arrays::Int& c) {
            return Float64Arrays::lt(Float64Arrays::int_elem(c), Float64Arrays::int_elem(Float64Arrays::total(i)));
        };
    };

private:
    static while_loop2_2<Float64Arrays::Array, Float64Arrays::IndexContainer, Float64Arrays::Array, Float64Arrays::Int, Float64Arrays::_transpose_body, Float64Arrays::_upper_bound> __while_loop2_26;
public:
    static Float64Arrays::_upper_bound upper_bound;
};
} // examples
} // moa
} // mg_src
} // moa_cpp

namespace examples {
namespace moa {
namespace mg_src {
namespace moa_cpp {
struct Int32Arrays {
private:
    static int32_utils __int32_utils;
public:
    typedef int32_utils::Int32 Int32;
    typedef array<Int32Arrays::Int32>::PaddedArray PaddedArray;
    struct _print_parray {
        inline void operator()(const Int32Arrays::PaddedArray& a) {
            return __array.print_parray(a);
        };
    };

    static Int32Arrays::_print_parray print_parray;
    typedef array<Int32Arrays::Int32>::Shape Shape;
    struct _cat_shape {
        inline Int32Arrays::Shape operator()(const Int32Arrays::Shape& a, const Int32Arrays::Shape& b) {
            return __array.cat_shape(a, b);
        };
    };

    static Int32Arrays::_cat_shape cat_shape;
    struct _padded_shape {
        inline Int32Arrays::Shape operator()(const Int32Arrays::PaddedArray& a) {
            return __array.padded_shape(a);
        };
    };

    static Int32Arrays::_padded_shape padded_shape;
    struct _print_shape {
        inline void operator()(const Int32Arrays::Shape& sh) {
            return __array.print_shape(sh);
        };
    };

    static Int32Arrays::_print_shape print_shape;
    struct _reverse_shape {
        inline Int32Arrays::Shape operator()(const Int32Arrays::Shape& s) {
            return __array.reverse_shape(s);
        };
    };

    static Int32Arrays::_reverse_shape reverse_shape;
private:
    static array<Int32Arrays::Int32> __array;
public:
    struct _abs {
        inline Int32Arrays::Int32 operator()(const Int32Arrays::Int32& a) {
            return __int32_utils.abs(a);
        };
    };

    static Int32Arrays::_abs abs;
    struct _eq {
        inline bool operator()(const Int32Arrays::Int32& a, const Int32Arrays::Int32& b) {
            return __int32_utils.eq(a, b);
        };
    };

    static Int32Arrays::_eq eq;
    struct _le {
        inline bool operator()(const Int32Arrays::Int32& a, const Int32Arrays::Int32& b) {
            return __int32_utils.le(a, b);
        };
    };

    static Int32Arrays::_le le;
    struct _lt {
        inline bool operator()(const Int32Arrays::Int32& a, const Int32Arrays::Int32& b) {
            return __int32_utils.lt(a, b);
        };
    };

    static Int32Arrays::_lt lt;
    struct _one {
        inline Int32Arrays::Int32 operator()() {
            return __int32_utils.one();
        };
    };

    static Int32Arrays::_one one;
    struct _print_element {
        inline void operator()(const Int32Arrays::Int32& e) {
            return __array.print_element(e);
        };
    };

    static Int32Arrays::_print_element print_element;
    struct _zero {
        inline Int32Arrays::Int32 operator()() {
            return __int32_utils.zero();
        };
    };

    static Int32Arrays::_zero zero;
    typedef array<Int32Arrays::Int32>::Int Int;
    struct _create_shape1 {
        inline Int32Arrays::Shape operator()(const Int32Arrays::Int& a) {
            return __array.create_shape1(a);
        };
    };

    static Int32Arrays::_create_shape1 create_shape1;
    struct _create_shape2 {
        inline Int32Arrays::Shape operator()(const Int32Arrays::Int& a, const Int32Arrays::Int& b) {
            return __array.create_shape2(a, b);
        };
    };

    static Int32Arrays::_create_shape2 create_shape2;
    struct _create_shape3 {
        inline Int32Arrays::Shape operator()(const Int32Arrays::Int& a, const Int32Arrays::Int& b, const Int32Arrays::Int& c) {
            return __array.create_shape3(a, b, c);
        };
    };

    static Int32Arrays::_create_shape3 create_shape3;
    struct _elem_int {
        inline Int32Arrays::Int operator()(const Int32Arrays::Int32& a) {
            return __array.elem_int(a);
        };
    };

    static Int32Arrays::_elem_int elem_int;
    struct _int_elem {
        inline Int32Arrays::Int32 operator()(const Int32Arrays::Int& a) {
            return __array.int_elem(a);
        };
    };

    static Int32Arrays::_int_elem int_elem;
    struct _padded_dim {
        inline Int32Arrays::Int operator()(const Int32Arrays::PaddedArray& a) {
            return __array.padded_dim(a);
        };
    };

    static Int32Arrays::_padded_dim padded_dim;
    struct _padded_drop_shape_elem {
        inline Int32Arrays::Shape operator()(const Int32Arrays::PaddedArray& a, const Int32Arrays::Int& i) {
            return __array.padded_drop_shape_elem(a, i);
        };
    };

    static Int32Arrays::_padded_drop_shape_elem padded_drop_shape_elem;
    struct _padded_get_shape_elem {
        inline Int32Arrays::Int operator()(const Int32Arrays::PaddedArray& a, const Int32Arrays::Int& i) {
            return __array.padded_get_shape_elem(a, i);
        };
    };

    static Int32Arrays::_padded_get_shape_elem padded_get_shape_elem;
    struct _padded_total {
        inline Int32Arrays::Int operator()(const Int32Arrays::PaddedArray& a) {
            return __array.padded_total(a);
        };
    };

    static Int32Arrays::_padded_total padded_total;
    struct _print_int {
        inline void operator()(const Int32Arrays::Int& u) {
            return __array.print_int(u);
        };
    };

    static Int32Arrays::_print_int print_int;
    typedef array<Int32Arrays::Int32>::IndexContainer IndexContainer;
    struct _padded_transpose_body {
        inline void operator()(const Int32Arrays::PaddedArray& a, const Int32Arrays::IndexContainer& ixc, Int32Arrays::PaddedArray& res, Int32Arrays::Int& c) {
            Int32Arrays::Index current_ix = Int32Arrays::get_index_ixc(ixc, c);
            Int32Arrays::Int32 current_element = Int32Arrays::unwrap_scalar(Int32Arrays::get(a, Int32Arrays::reverse_index(current_ix)));
            Int32Arrays::set(res, current_ix, current_element);
            c = Int32Arrays::elem_int(Int32Arrays::binary_add(Int32Arrays::int_elem(c), Int32Arrays::one()));
        };
    };

    static Int32Arrays::_padded_transpose_body padded_transpose_body;
    struct _padded_transpose_repeat {
        inline void operator()(const Int32Arrays::PaddedArray& context1, const Int32Arrays::IndexContainer& context2, Int32Arrays::PaddedArray& state1, Int32Arrays::Int& state2) {
            return __while_loop2_29.repeat(context1, context2, state1, state2);
        };
    };

    static Int32Arrays::_padded_transpose_repeat padded_transpose_repeat;
    struct _padded_upper_bound {
        inline bool operator()(const Int32Arrays::PaddedArray& a, const Int32Arrays::IndexContainer& i, const Int32Arrays::PaddedArray& res, const Int32Arrays::Int& c) {
            return Int32Arrays::lt(Int32Arrays::int_elem(c), Int32Arrays::int_elem(Int32Arrays::total(i)));
        };
    };

private:
    static while_loop2_2<Int32Arrays::PaddedArray, Int32Arrays::IndexContainer, Int32Arrays::PaddedArray, Int32Arrays::Int, Int32Arrays::_padded_transpose_body, Int32Arrays::_padded_upper_bound> __while_loop2_29;
public:
    static Int32Arrays::_padded_upper_bound padded_upper_bound;
    struct _print_index_container {
        inline void operator()(const Int32Arrays::IndexContainer& i) {
            return __array.print_index_container(i);
        };
    };

    static Int32Arrays::_print_index_container print_index_container;
    typedef array<Int32Arrays::Int32>::Index Index;
    struct _cat_index {
        inline Int32Arrays::Index operator()(const Int32Arrays::Index& i, const Int32Arrays::Index& j) {
            return __array.cat_index(i, j);
        };
    };

    static Int32Arrays::_cat_index cat_index;
    struct _create_index1 {
        inline Int32Arrays::Index operator()(const Int32Arrays::Int& a) {
            return __array.create_index1(a);
        };
    };

    static Int32Arrays::_create_index1 create_index1;
    struct _create_index2 {
        inline Int32Arrays::Index operator()(const Int32Arrays::Int& a, const Int32Arrays::Int& b) {
            return __array.create_index2(a, b);
        };
    };

    static Int32Arrays::_create_index2 create_index2;
    struct _create_index3 {
        inline Int32Arrays::Index operator()(const Int32Arrays::Int& a, const Int32Arrays::Int& b, const Int32Arrays::Int& c) {
            return __array.create_index3(a, b, c);
        };
    };

    static Int32Arrays::_create_index3 create_index3;
    struct _drop_index_elem {
        inline Int32Arrays::Index operator()(const Int32Arrays::Index& ix, const Int32Arrays::Int& i) {
            return __array.drop_index_elem(ix, i);
        };
    };

    static Int32Arrays::_drop_index_elem drop_index_elem;
    struct _get_index_elem {
        inline Int32Arrays::Int operator()(const Int32Arrays::Index& ix, const Int32Arrays::Int& i) {
            return __array.get_index_elem(ix, i);
        };
    };

    static Int32Arrays::_get_index_elem get_index_elem;
    struct _get_index_ixc {
        inline Int32Arrays::Index operator()(const Int32Arrays::IndexContainer& ixc, const Int32Arrays::Int& ix) {
            return __array.get_index_ixc(ixc, ix);
        };
    };

    static Int32Arrays::_get_index_ixc get_index_ixc;
    struct _print_index {
        inline void operator()(const Int32Arrays::Index& i) {
            return __array.print_index(i);
        };
    };

    static Int32Arrays::_print_index print_index;
    struct _reverse_index {
        inline Int32Arrays::Index operator()(const Int32Arrays::Index& ix) {
            return __array.reverse_index(ix);
        };
    };

    static Int32Arrays::_reverse_index reverse_index;
    struct _test_index {
        inline Int32Arrays::Index operator()() {
            return __array.test_index();
        };
    };

    static Int32Arrays::_test_index test_index;
    typedef array<Int32Arrays::Int32>::Array Array;
    struct _binaryMap {
        inline void operator()(const Int32Arrays::Array& a, const Int32Arrays::Array& b, const Int32Arrays::Index& ix) {
            assert((Int32Arrays::unwrap_scalar(Int32Arrays::get(Int32Arrays::binary_add(a, b), ix))) == (Int32Arrays::binary_add(Int32Arrays::unwrap_scalar(Int32Arrays::get(a, ix)), Int32Arrays::unwrap_scalar(Int32Arrays::get(b, ix)))));
            assert((Int32Arrays::unwrap_scalar(Int32Arrays::get(Int32Arrays::binary_sub(a, b), ix))) == (Int32Arrays::binary_sub(Int32Arrays::unwrap_scalar(Int32Arrays::get(a, ix)), Int32Arrays::unwrap_scalar(Int32Arrays::get(b, ix)))));
            assert((Int32Arrays::unwrap_scalar(Int32Arrays::get(Int32Arrays::mul(a, b), ix))) == (Int32Arrays::mul(Int32Arrays::unwrap_scalar(Int32Arrays::get(a, ix)), Int32Arrays::unwrap_scalar(Int32Arrays::get(b, ix)))));
            assert((Int32Arrays::unwrap_scalar(Int32Arrays::get(Int32Arrays::div(a, b), ix))) == (Int32Arrays::div(Int32Arrays::unwrap_scalar(Int32Arrays::get(a, ix)), Int32Arrays::unwrap_scalar(Int32Arrays::get(b, ix)))));
        };
    };

    static Int32Arrays::_binaryMap binaryMap;
    struct _binary_add {
        inline Int32Arrays::Array operator()(const Int32Arrays::Array& a, const Int32Arrays::Array& b) {
            Int32Arrays::IndexContainer ix_space = Int32Arrays::create_total_indices(a);
            Int32Arrays::Array b_upd = b;
            Int32Arrays::Int counter = Int32Arrays::elem_int(Int32Arrays::zero());
            Int32Arrays::bmb_plus_rep(a, ix_space, b_upd, counter);
            return b;
        };
        inline Int32Arrays::Int32 operator()(const Int32Arrays::Int32& a, const Int32Arrays::Int32& b) {
            return __int32_utils.binary_add(a, b);
        };
    };

    static Int32Arrays::_binary_add binary_add;
    struct _binary_sub {
        inline Int32Arrays::Array operator()(const Int32Arrays::Array& a, const Int32Arrays::Array& b) {
            Int32Arrays::IndexContainer ix_space = Int32Arrays::create_total_indices(a);
            Int32Arrays::Array b_upd = b;
            Int32Arrays::Int counter = Int32Arrays::elem_int(Int32Arrays::zero());
            Int32Arrays::bmb_sub_rep(a, ix_space, b_upd, counter);
            return b;
        };
        inline Int32Arrays::Int32 operator()(const Int32Arrays::Int32& a, const Int32Arrays::Int32& b) {
            return __int32_utils.binary_sub(a, b);
        };
    };

    static Int32Arrays::_binary_sub binary_sub;
    struct _bmb_div {
        inline void operator()(const Int32Arrays::Array& a, const Int32Arrays::IndexContainer& ix_space, Int32Arrays::Array& b, Int32Arrays::Int& c) {
            Int32Arrays::Index ix = Int32Arrays::get_index_ixc(ix_space, c);
            Int32Arrays::Int32 new_value = Int32Arrays::div(Int32Arrays::unwrap_scalar(Int32Arrays::get(a, ix)), Int32Arrays::unwrap_scalar(Int32Arrays::get(b, ix)));
            Int32Arrays::set(b, ix, new_value);
            c = Int32Arrays::elem_int(Int32Arrays::binary_add(Int32Arrays::int_elem(c), Int32Arrays::one()));
        };
    };

    static Int32Arrays::_bmb_div bmb_div;
    struct _bmb_div_rep {
        inline void operator()(const Int32Arrays::Array& context1, const Int32Arrays::IndexContainer& context2, Int32Arrays::Array& state1, Int32Arrays::Int& state2) {
            return __while_loop2_21.repeat(context1, context2, state1, state2);
        };
    };

    static Int32Arrays::_bmb_div_rep bmb_div_rep;
    struct _bmb_mul {
        inline void operator()(const Int32Arrays::Array& a, const Int32Arrays::IndexContainer& ix_space, Int32Arrays::Array& b, Int32Arrays::Int& c) {
            Int32Arrays::Index ix = Int32Arrays::get_index_ixc(ix_space, c);
            Int32Arrays::Int32 new_value = Int32Arrays::mul(Int32Arrays::unwrap_scalar(Int32Arrays::get(a, ix)), Int32Arrays::unwrap_scalar(Int32Arrays::get(b, ix)));
            Int32Arrays::set(b, ix, new_value);
            c = Int32Arrays::elem_int(Int32Arrays::binary_add(Int32Arrays::int_elem(c), Int32Arrays::one()));
        };
    };

    static Int32Arrays::_bmb_mul bmb_mul;
    struct _bmb_mul_rep {
        inline void operator()(const Int32Arrays::Array& context1, const Int32Arrays::IndexContainer& context2, Int32Arrays::Array& state1, Int32Arrays::Int& state2) {
            return __while_loop2_22.repeat(context1, context2, state1, state2);
        };
    };

    static Int32Arrays::_bmb_mul_rep bmb_mul_rep;
    struct _bmb_plus {
        inline void operator()(const Int32Arrays::Array& a, const Int32Arrays::IndexContainer& ix_space, Int32Arrays::Array& b, Int32Arrays::Int& c) {
            Int32Arrays::Index ix = Int32Arrays::get_index_ixc(ix_space, c);
            Int32Arrays::Int32 new_value = Int32Arrays::binary_add(Int32Arrays::unwrap_scalar(Int32Arrays::get(a, ix)), Int32Arrays::unwrap_scalar(Int32Arrays::get(b, ix)));
            Int32Arrays::set(b, ix, new_value);
            c = Int32Arrays::elem_int(Int32Arrays::binary_add(Int32Arrays::int_elem(c), Int32Arrays::one()));
        };
    };

    static Int32Arrays::_bmb_plus bmb_plus;
    struct _bmb_plus_rep {
        inline void operator()(const Int32Arrays::Array& context1, const Int32Arrays::IndexContainer& context2, Int32Arrays::Array& state1, Int32Arrays::Int& state2) {
            return __while_loop2_23.repeat(context1, context2, state1, state2);
        };
    };

    static Int32Arrays::_bmb_plus_rep bmb_plus_rep;
    struct _bmb_sub {
        inline void operator()(const Int32Arrays::Array& a, const Int32Arrays::IndexContainer& ix_space, Int32Arrays::Array& b, Int32Arrays::Int& c) {
            Int32Arrays::Index ix = Int32Arrays::get_index_ixc(ix_space, c);
            Int32Arrays::Int32 new_value = Int32Arrays::binary_sub(Int32Arrays::unwrap_scalar(Int32Arrays::get(a, ix)), Int32Arrays::unwrap_scalar(Int32Arrays::get(b, ix)));
            Int32Arrays::set(b, ix, new_value);
            c = Int32Arrays::elem_int(Int32Arrays::binary_add(Int32Arrays::int_elem(c), Int32Arrays::one()));
        };
    };

    static Int32Arrays::_bmb_sub bmb_sub;
    struct _bmb_sub_rep {
        inline void operator()(const Int32Arrays::Array& context1, const Int32Arrays::IndexContainer& context2, Int32Arrays::Array& state1, Int32Arrays::Int& state2) {
            return __while_loop2_24.repeat(context1, context2, state1, state2);
        };
    };

    static Int32Arrays::_bmb_sub_rep bmb_sub_rep;
    struct _cat {
        inline Int32Arrays::Array operator()(const Int32Arrays::Array& array1, const Int32Arrays::Array& array2) {
            Int32Arrays::Int32 take_a1 = Int32Arrays::int_elem(Int32Arrays::get_shape_elem(array1, Int32Arrays::elem_int(Int32Arrays::zero())));
            Int32Arrays::Int32 take_a2 = Int32Arrays::int_elem(Int32Arrays::get_shape_elem(array2, Int32Arrays::elem_int(Int32Arrays::zero())));
            Int32Arrays::Shape drop_a1 = Int32Arrays::drop_shape_elem(array1, Int32Arrays::elem_int(Int32Arrays::zero()));
            Int32Arrays::Shape result_shape = Int32Arrays::cat_shape(Int32Arrays::create_shape1(Int32Arrays::elem_int(Int32Arrays::binary_add(take_a1, take_a2))), drop_a1);
            Int32Arrays::Array res = Int32Arrays::create_array(result_shape);
            Int32Arrays::Int counter = Int32Arrays::elem_int(Int32Arrays::zero());
            Int32Arrays::cat_repeat(array1, array2, counter, res);
            return res;
        };
    };

    static Int32Arrays::_cat cat;
    struct _cat_body {
        inline void operator()(const Int32Arrays::Array& array1, const Int32Arrays::Array& array2, Int32Arrays::Int& counter, Int32Arrays::Array& res) {
            Int32Arrays::Int32 s_0 = Int32Arrays::int_elem(Int32Arrays::total(array1));
            if (Int32Arrays::lt(Int32Arrays::int_elem(counter), s_0))
            {
                Int32Arrays::set(res, counter, Int32Arrays::unwrap_scalar(Int32Arrays::get(array1, counter)));
                counter = Int32Arrays::elem_int(Int32Arrays::binary_add(Int32Arrays::int_elem(counter), Int32Arrays::one()));
            }
            else
            {
                Int32Arrays::Int ix = Int32Arrays::elem_int(Int32Arrays::binary_sub(Int32Arrays::int_elem(counter), s_0));
                Int32Arrays::set(res, counter, Int32Arrays::unwrap_scalar(Int32Arrays::get(array2, ix)));
                counter = Int32Arrays::elem_int(Int32Arrays::binary_add(Int32Arrays::int_elem(counter), Int32Arrays::one()));
            }
        };
    };

    static Int32Arrays::_cat_body cat_body;
    struct _cat_cond {
        inline bool operator()(const Int32Arrays::Array& array1, const Int32Arrays::Array& array2, const Int32Arrays::Int& counter, const Int32Arrays::Array& res) {
            Int32Arrays::Int32 upper_bound = Int32Arrays::int_elem(Int32Arrays::total(res));
            return Int32Arrays::lt(Int32Arrays::int_elem(counter), upper_bound);
        };
    };

private:
    static while_loop2_2<Int32Arrays::Array, Int32Arrays::Array, Int32Arrays::Int, Int32Arrays::Array, Int32Arrays::_cat_body, Int32Arrays::_cat_cond> __while_loop2_20;
public:
    static Int32Arrays::_cat_cond cat_cond;
    struct _cat_repeat {
        inline void operator()(const Int32Arrays::Array& context1, const Int32Arrays::Array& context2, Int32Arrays::Int& state1, Int32Arrays::Array& state2) {
            return __while_loop2_20.repeat(context1, context2, state1, state2);
        };
    };

    static Int32Arrays::_cat_repeat cat_repeat;
    struct _cat_vec {
        inline Int32Arrays::Array operator()(const Int32Arrays::Array& vector1, const Int32Arrays::Array& vector2) {
            Int32Arrays::Shape res_shape = Int32Arrays::create_shape1(Int32Arrays::elem_int(Int32Arrays::binary_add(Int32Arrays::int_elem(Int32Arrays::total(vector1)), Int32Arrays::int_elem(Int32Arrays::total(vector2)))));
            Int32Arrays::Array res = Int32Arrays::create_array(res_shape);
            Int32Arrays::Int counter = Int32Arrays::elem_int(Int32Arrays::zero());
            Int32Arrays::cat_vec_repeat(vector1, vector2, res, counter);
            return res;
        };
    };

    static Int32Arrays::_cat_vec cat_vec;
    struct _cat_vec_body {
        inline void operator()(const Int32Arrays::Array& v1, const Int32Arrays::Array& v2, Int32Arrays::Array& res, Int32Arrays::Int& counter) {
            Int32Arrays::Int32 v1_bound = Int32Arrays::int_elem(Int32Arrays::total(v1));
            Int32Arrays::Index ix;
            if (Int32Arrays::lt(Int32Arrays::int_elem(counter), Int32Arrays::int_elem(Int32Arrays::total(v1))))
            {
                ix = Int32Arrays::create_index1(counter);
                Int32Arrays::set(res, ix, Int32Arrays::unwrap_scalar(Int32Arrays::get(v1, ix)));
            }
            else
            {
                ix = Int32Arrays::create_index1(Int32Arrays::elem_int(Int32Arrays::binary_sub(Int32Arrays::int_elem(counter), v1_bound)));
                Int32Arrays::Index res_ix = Int32Arrays::create_index1(counter);
                Int32Arrays::set(res, res_ix, Int32Arrays::unwrap_scalar(Int32Arrays::get(v2, ix)));
            }
            counter = Int32Arrays::elem_int(Int32Arrays::binary_add(Int32Arrays::int_elem(counter), Int32Arrays::one()));
        };
    };

    static Int32Arrays::_cat_vec_body cat_vec_body;
    struct _cat_vec_cond {
        inline bool operator()(const Int32Arrays::Array& v1, const Int32Arrays::Array& v2, const Int32Arrays::Array& res, const Int32Arrays::Int& counter) {
            return Int32Arrays::lt(Int32Arrays::int_elem(counter), Int32Arrays::int_elem(Int32Arrays::total(res)));
        };
    };

private:
    static while_loop2_2<Int32Arrays::Array, Int32Arrays::Array, Int32Arrays::Array, Int32Arrays::Int, Int32Arrays::_cat_vec_body, Int32Arrays::_cat_vec_cond> __while_loop2_2;
public:
    static Int32Arrays::_cat_vec_cond cat_vec_cond;
    struct _cat_vec_repeat {
        inline void operator()(const Int32Arrays::Array& context1, const Int32Arrays::Array& context2, Int32Arrays::Array& state1, Int32Arrays::Int& state2) {
            return __while_loop2_2.repeat(context1, context2, state1, state2);
        };
    };

    static Int32Arrays::_cat_vec_repeat cat_vec_repeat;
    struct _circular_padl {
        inline Int32Arrays::PaddedArray operator()(const Int32Arrays::PaddedArray& a, const Int32Arrays::Int& ix) {
            Int32Arrays::Array padding = Int32Arrays::get(a, Int32Arrays::create_index1(ix));
            Int32Arrays::Shape reshape_shape = Int32Arrays::cat_shape(Int32Arrays::create_shape1(Int32Arrays::elem_int(Int32Arrays::one())), Int32Arrays::shape(padding));
            Int32Arrays::Array reshaped_padding = Int32Arrays::reshape(padding, reshape_shape);
            Int32Arrays::Array catenated_array = Int32Arrays::cat(reshaped_padding, Int32Arrays::padded_to_unpadded(a));
            Int32Arrays::Shape unpadded_shape = Int32Arrays::shape(a);
            Int32Arrays::Shape padded_shape = Int32Arrays::shape(catenated_array);
            Int32Arrays::PaddedArray res = Int32Arrays::create_padded_array(unpadded_shape, padded_shape, catenated_array);
            return res;
        };
        inline Int32Arrays::PaddedArray operator()(const Int32Arrays::Array& a, const Int32Arrays::Int& ix) {
            Int32Arrays::Array padding = Int32Arrays::get(a, Int32Arrays::create_index1(ix));
            Int32Arrays::Shape reshape_shape = Int32Arrays::cat_shape(Int32Arrays::create_shape1(Int32Arrays::elem_int(Int32Arrays::one())), Int32Arrays::shape(padding));
            Int32Arrays::Array reshaped_padding = Int32Arrays::reshape(padding, reshape_shape);
            Int32Arrays::Array catenated_array = Int32Arrays::cat(reshaped_padding, a);
            Int32Arrays::Shape unpadded_shape = Int32Arrays::shape(a);
            Int32Arrays::Shape padded_shape = Int32Arrays::shape(catenated_array);
            Int32Arrays::PaddedArray res = Int32Arrays::create_padded_array(unpadded_shape, padded_shape, catenated_array);
            return res;
        };
    };

    static Int32Arrays::_circular_padl circular_padl;
    struct _circular_padr {
        inline Int32Arrays::PaddedArray operator()(const Int32Arrays::PaddedArray& a, const Int32Arrays::Int& ix) {
            Int32Arrays::Shape unpadded_shape = Int32Arrays::shape(a);
            Int32Arrays::Array padding = Int32Arrays::get(a, Int32Arrays::create_index1(ix));
            Int32Arrays::Shape reshape_shape = Int32Arrays::cat_shape(Int32Arrays::create_shape1(Int32Arrays::elem_int(Int32Arrays::one())), Int32Arrays::shape(padding));
            Int32Arrays::Array reshaped_padding = Int32Arrays::reshape(padding, reshape_shape);
            Int32Arrays::Array catenated_array = Int32Arrays::cat(Int32Arrays::padded_to_unpadded(a), reshaped_padding);
            Int32Arrays::Shape padded_shape = Int32Arrays::shape(catenated_array);
            Int32Arrays::PaddedArray res = Int32Arrays::create_padded_array(unpadded_shape, padded_shape, catenated_array);
            return res;
        };
        inline Int32Arrays::PaddedArray operator()(const Int32Arrays::Array& a, const Int32Arrays::Int& ix) {
            Int32Arrays::Array padding = Int32Arrays::get(a, Int32Arrays::create_index1(ix));
            Int32Arrays::Shape reshape_shape = Int32Arrays::cat_shape(Int32Arrays::create_shape1(Int32Arrays::elem_int(Int32Arrays::one())), Int32Arrays::shape(padding));
            Int32Arrays::Array reshaped_padding = Int32Arrays::reshape(padding, reshape_shape);
            Int32Arrays::Array catenated_array = Int32Arrays::cat(a, reshaped_padding);
            Int32Arrays::Shape unpadded_shape = Int32Arrays::shape(a);
            Int32Arrays::Shape padded_shape = Int32Arrays::shape(catenated_array);
            Int32Arrays::PaddedArray res = Int32Arrays::create_padded_array(unpadded_shape, padded_shape, catenated_array);
            return res;
        };
    };

    static Int32Arrays::_circular_padr circular_padr;
    struct _create_array {
        inline Int32Arrays::Array operator()(const Int32Arrays::Shape& sh) {
            return __array.create_array(sh);
        };
    };

    static Int32Arrays::_create_array create_array;
    struct _create_padded_array {
        inline Int32Arrays::PaddedArray operator()(const Int32Arrays::Shape& unpadded_shape, const Int32Arrays::Shape& padded_shape, const Int32Arrays::Array& padded_array) {
            return __array.create_padded_array(unpadded_shape, padded_shape, padded_array);
        };
    };

    static Int32Arrays::_create_padded_array create_padded_array;
    struct _create_partial_indices {
        inline Int32Arrays::IndexContainer operator()(const Int32Arrays::Array& a, const Int32Arrays::Int& i) {
            return __array.create_partial_indices(a, i);
        };
    };

    static Int32Arrays::_create_partial_indices create_partial_indices;
    struct _create_total_indices {
        inline Int32Arrays::IndexContainer operator()(const Int32Arrays::Array& a) {
            return __array.create_total_indices(a);
        };
        inline Int32Arrays::IndexContainer operator()(const Int32Arrays::PaddedArray& a) {
            return __array.create_total_indices(a);
        };
    };

    static Int32Arrays::_create_total_indices create_total_indices;
    struct _dim {
        inline Int32Arrays::Int operator()(const Int32Arrays::Array& a) {
            return __array.dim(a);
        };
    };

    static Int32Arrays::_dim dim;
    struct _div {
        inline Int32Arrays::Array operator()(const Int32Arrays::Array& a, const Int32Arrays::Array& b) {
            Int32Arrays::IndexContainer ix_space = Int32Arrays::create_total_indices(a);
            Int32Arrays::Array b_upd = b;
            Int32Arrays::Int counter = Int32Arrays::elem_int(Int32Arrays::zero());
            Int32Arrays::bmb_div_rep(a, ix_space, b_upd, counter);
            return b;
        };
        inline Int32Arrays::Int32 operator()(const Int32Arrays::Int32& a, const Int32Arrays::Int32& b) {
            return __int32_utils.div(a, b);
        };
    };

    static Int32Arrays::_div div;
    struct _drop {
        inline Int32Arrays::Array operator()(const Int32Arrays::Int& i, const Int32Arrays::Array& a) {
            Int32Arrays::Shape drop_sh_0 = Int32Arrays::drop_shape_elem(a, Int32Arrays::elem_int(Int32Arrays::zero()));
            Int32Arrays::Array res_array;
            if (Int32Arrays::lt(Int32Arrays::int_elem(i), Int32Arrays::zero()))
            {
                Int32Arrays::Int take_value = Int32Arrays::elem_int(Int32Arrays::binary_add(Int32Arrays::int_elem(Int32Arrays::get_shape_elem(a, Int32Arrays::elem_int(Int32Arrays::zero()))), Int32Arrays::int_elem(i)));
                res_array = Int32Arrays::take(take_value, a);
            }
            else
            {
                res_array = Int32Arrays::get(a, Int32Arrays::create_index1(i));
                res_array = Int32Arrays::reshape(res_array, Int32Arrays::cat_shape(Int32Arrays::create_shape1(Int32Arrays::elem_int(Int32Arrays::one())), Int32Arrays::shape(res_array)));
                Int32Arrays::Int c = Int32Arrays::elem_int(Int32Arrays::binary_add(Int32Arrays::int_elem(i), Int32Arrays::one()));
                Int32Arrays::drop_repeat(a, i, res_array, c);
            }
            return res_array;
        };
    };

    static Int32Arrays::_drop drop;
    struct _drop_body {
        inline void operator()(const Int32Arrays::Array& a, const Int32Arrays::Int& i, Int32Arrays::Array& res, Int32Arrays::Int& c) {
            Int32Arrays::Array slice = Int32Arrays::get(a, Int32Arrays::create_index1(c));
            slice = Int32Arrays::reshape(slice, Int32Arrays::cat_shape(Int32Arrays::create_shape1(Int32Arrays::elem_int(Int32Arrays::one())), Int32Arrays::shape(slice)));
            res = Int32Arrays::cat(res, slice);
            c = Int32Arrays::elem_int(Int32Arrays::binary_add(Int32Arrays::int_elem(c), Int32Arrays::one()));
        };
    };

    static Int32Arrays::_drop_body drop_body;
    struct _drop_cond {
        inline bool operator()(const Int32Arrays::Array& a, const Int32Arrays::Int& i, const Int32Arrays::Array& res, const Int32Arrays::Int& c) {
            return Int32Arrays::lt(Int32Arrays::int_elem(c), Int32Arrays::int_elem(Int32Arrays::get_shape_elem(a, Int32Arrays::elem_int(Int32Arrays::zero()))));
        };
    };

private:
    static while_loop2_2<Int32Arrays::Array, Int32Arrays::Int, Int32Arrays::Array, Int32Arrays::Int, Int32Arrays::_drop_body, Int32Arrays::_drop_cond> __while_loop2_27;
public:
    static Int32Arrays::_drop_cond drop_cond;
    struct _drop_repeat {
        inline void operator()(const Int32Arrays::Array& context1, const Int32Arrays::Int& context2, Int32Arrays::Array& state1, Int32Arrays::Int& state2) {
            return __while_loop2_27.repeat(context1, context2, state1, state2);
        };
    };

    static Int32Arrays::_drop_repeat drop_repeat;
    struct _drop_shape_elem {
        inline Int32Arrays::Shape operator()(const Int32Arrays::Array& a, const Int32Arrays::Int& i) {
            return __array.drop_shape_elem(a, i);
        };
    };

    static Int32Arrays::_drop_shape_elem drop_shape_elem;
    struct _get {
        inline Int32Arrays::Array operator()(const Int32Arrays::Array& a, const Int32Arrays::Int& ix) {
            return __array.get(a, ix);
        };
        inline Int32Arrays::Array operator()(const Int32Arrays::Array& a, const Int32Arrays::Index& ix) {
            return __array.get(a, ix);
        };
        inline Int32Arrays::Array operator()(const Int32Arrays::PaddedArray& a, const Int32Arrays::Index& ix) {
            return __array.get(a, ix);
        };
    };

    static Int32Arrays::_get get;
    struct _get_shape_elem {
        inline Int32Arrays::Int operator()(const Int32Arrays::Array& a, const Int32Arrays::Int& i) {
            return __array.get_shape_elem(a, i);
        };
    };

    static Int32Arrays::_get_shape_elem get_shape_elem;
    struct _mapped_ops_cond {
        inline bool operator()(const Int32Arrays::Array& a, const Int32Arrays::IndexContainer& ix_space, const Int32Arrays::Array& b, const Int32Arrays::Int& c) {
            return Int32Arrays::lt(Int32Arrays::int_elem(c), Int32Arrays::int_elem(Int32Arrays::total(ix_space)));
        };
    };

private:
    static while_loop2_2<Int32Arrays::Array, Int32Arrays::IndexContainer, Int32Arrays::Array, Int32Arrays::Int, Int32Arrays::_bmb_div, Int32Arrays::_mapped_ops_cond> __while_loop2_21;
    static while_loop2_2<Int32Arrays::Array, Int32Arrays::IndexContainer, Int32Arrays::Array, Int32Arrays::Int, Int32Arrays::_bmb_mul, Int32Arrays::_mapped_ops_cond> __while_loop2_22;
    static while_loop2_2<Int32Arrays::Array, Int32Arrays::IndexContainer, Int32Arrays::Array, Int32Arrays::Int, Int32Arrays::_bmb_plus, Int32Arrays::_mapped_ops_cond> __while_loop2_23;
    static while_loop2_2<Int32Arrays::Array, Int32Arrays::IndexContainer, Int32Arrays::Array, Int32Arrays::Int, Int32Arrays::_bmb_sub, Int32Arrays::_mapped_ops_cond> __while_loop2_24;
public:
    static Int32Arrays::_mapped_ops_cond mapped_ops_cond;
    struct _mul {
        inline Int32Arrays::Array operator()(const Int32Arrays::Array& a, const Int32Arrays::Array& b) {
            Int32Arrays::IndexContainer ix_space = Int32Arrays::create_total_indices(a);
            Int32Arrays::Array b_upd = b;
            Int32Arrays::Int counter = Int32Arrays::elem_int(Int32Arrays::zero());
            Int32Arrays::bmb_mul_rep(a, ix_space, b_upd, counter);
            return b;
        };
        inline Int32Arrays::Int32 operator()(const Int32Arrays::Int32& a, const Int32Arrays::Int32& b) {
            return __int32_utils.mul(a, b);
        };
    };

    static Int32Arrays::_mul mul;
    struct _padded_to_unpadded {
        inline Int32Arrays::Array operator()(const Int32Arrays::PaddedArray& a) {
            return __array.padded_to_unpadded(a);
        };
    };

    static Int32Arrays::_padded_to_unpadded padded_to_unpadded;
    struct _print_array {
        inline void operator()(const Int32Arrays::Array& a) {
            return __array.print_array(a);
        };
    };

    static Int32Arrays::_print_array print_array;
    struct _reshape {
        inline Int32Arrays::Array operator()(const Int32Arrays::Array& input_array, const Int32Arrays::Shape& s) {
            Int32Arrays::Array new_array = Int32Arrays::create_array(s);
            Int32Arrays::Int counter = Int32Arrays::elem_int(Int32Arrays::zero());
            Int32Arrays::reshape_repeat(input_array, new_array, counter);
            return new_array;
        };
    };

    static Int32Arrays::_reshape reshape;
    struct _reshape_body {
        inline void operator()(const Int32Arrays::Array& old_array, Int32Arrays::Array& new_array, Int32Arrays::Int& counter) {
            Int32Arrays::set(new_array, counter, Int32Arrays::unwrap_scalar(Int32Arrays::get(old_array, counter)));
            counter = Int32Arrays::elem_int(Int32Arrays::binary_add(Int32Arrays::int_elem(counter), Int32Arrays::one()));
        };
    };

    static Int32Arrays::_reshape_body reshape_body;
    struct _reshape_cond {
        inline bool operator()(const Int32Arrays::Array& old_array, const Int32Arrays::Array& new_array, const Int32Arrays::Int& counter) {
            return Int32Arrays::lt(Int32Arrays::int_elem(counter), Int32Arrays::int_elem(Int32Arrays::total(new_array)));
        };
    };

private:
    static while_loop1_2<Int32Arrays::Array, Int32Arrays::Array, Int32Arrays::Int, Int32Arrays::_reshape_body, Int32Arrays::_reshape_cond> __while_loop1_2;
public:
    static Int32Arrays::_reshape_cond reshape_cond;
    struct _reshape_repeat {
        inline void operator()(const Int32Arrays::Array& context1, Int32Arrays::Array& state1, Int32Arrays::Int& state2) {
            return __while_loop1_2.repeat(context1, state1, state2);
        };
    };

    static Int32Arrays::_reshape_repeat reshape_repeat;
    struct _reverse {
        inline Int32Arrays::Array operator()(const Int32Arrays::Array& a) {
            Int32Arrays::Array res_array = Int32Arrays::create_array(Int32Arrays::shape(a));
            Int32Arrays::IndexContainer valid_indices = Int32Arrays::create_total_indices(res_array);
            Int32Arrays::Int counter = Int32Arrays::elem_int(Int32Arrays::zero());
            Int32Arrays::reverse_repeat(a, valid_indices, res_array, counter);
            return res_array;
        };
    };

    static Int32Arrays::_reverse reverse;
    struct _reverse_body {
        inline void operator()(const Int32Arrays::Array& input, const Int32Arrays::IndexContainer& indices, Int32Arrays::Array& res, Int32Arrays::Int& c) {
            Int32Arrays::Index ix = Int32Arrays::get_index_ixc(indices, c);
            Int32Arrays::Int32 elem = Int32Arrays::unwrap_scalar(Int32Arrays::get(input, ix));
            Int32Arrays::Int sh_0 = Int32Arrays::get_shape_elem(input, Int32Arrays::elem_int(Int32Arrays::zero()));
            Int32Arrays::Int ix_0 = Int32Arrays::get_index_elem(ix, Int32Arrays::elem_int(Int32Arrays::zero()));
            Int32Arrays::Int32 new_ix_0 = Int32Arrays::binary_sub(Int32Arrays::int_elem(sh_0), Int32Arrays::binary_add(Int32Arrays::int_elem(ix_0), Int32Arrays::one()));
            Int32Arrays::Index new_ix = Int32Arrays::cat_index(Int32Arrays::create_index1(Int32Arrays::elem_int(new_ix_0)), Int32Arrays::drop_index_elem(ix, Int32Arrays::elem_int(Int32Arrays::zero())));
            Int32Arrays::set(res, new_ix, elem);
            c = Int32Arrays::elem_int(Int32Arrays::binary_add(Int32Arrays::int_elem(c), Int32Arrays::one()));
        };
    };

    static Int32Arrays::_reverse_body reverse_body;
    struct _reverse_cond {
        inline bool operator()(const Int32Arrays::Array& input, const Int32Arrays::IndexContainer& indices, const Int32Arrays::Array& res, const Int32Arrays::Int& c) {
            return Int32Arrays::lt(Int32Arrays::int_elem(c), Int32Arrays::int_elem(Int32Arrays::total(indices)));
        };
    };

private:
    static while_loop2_2<Int32Arrays::Array, Int32Arrays::IndexContainer, Int32Arrays::Array, Int32Arrays::Int, Int32Arrays::_reverse_body, Int32Arrays::_reverse_cond> __while_loop2_25;
public:
    static Int32Arrays::_reverse_cond reverse_cond;
    struct _reverse_repeat {
        inline void operator()(const Int32Arrays::Array& context1, const Int32Arrays::IndexContainer& context2, Int32Arrays::Array& state1, Int32Arrays::Int& state2) {
            return __while_loop2_25.repeat(context1, context2, state1, state2);
        };
    };

    static Int32Arrays::_reverse_repeat reverse_repeat;
    struct _rotate {
        inline Int32Arrays::Array operator()(const Int32Arrays::Int& sigma, const Int32Arrays::Int& ax, const Int32Arrays::Array& a) {
            Int32Arrays::IndexContainer ix_space = Int32Arrays::create_partial_indices(a, ax);
            Int32Arrays::Int c = Int32Arrays::elem_int(Int32Arrays::zero());
            Int32Arrays::Index i = Int32Arrays::get_index_ixc(ix_space, c);
            Int32Arrays::Array res;
            if (Int32Arrays::lt(Int32Arrays::zero(), Int32Arrays::int_elem(sigma)))
            {
                Int32Arrays::Array e1 = Int32Arrays::drop(sigma, Int32Arrays::get(a, i));
                Int32Arrays::Array e2 = Int32Arrays::take(sigma, Int32Arrays::get(a, i));
                res = Int32Arrays::cat(e1, e2);
            }
            else
            {
                Int32Arrays::Array e1 = Int32Arrays::take(Int32Arrays::elem_int(Int32Arrays::unary_sub(Int32Arrays::int_elem(sigma))), Int32Arrays::get(a, i));
                Int32Arrays::Array e2 = Int32Arrays::drop(Int32Arrays::elem_int(Int32Arrays::unary_sub(Int32Arrays::int_elem(sigma))), Int32Arrays::get(a, i));
                Int32Arrays::print_array(e1);
                Int32Arrays::print_array(e2);
                res = Int32Arrays::cat(e1, e2);
            }
            return res;
        };
    };

    static Int32Arrays::_rotate rotate;
    struct _set {
        inline void operator()(Int32Arrays::Array& a, const Int32Arrays::Int& ix, const Int32Arrays::Int32& e) {
            return __array.set(a, ix, e);
        };
        inline void operator()(Int32Arrays::Array& a, const Int32Arrays::Index& ix, const Int32Arrays::Int32& e) {
            return __array.set(a, ix, e);
        };
        inline void operator()(Int32Arrays::PaddedArray& a, const Int32Arrays::Index& ix, const Int32Arrays::Int32& e) {
            return __array.set(a, ix, e);
        };
    };

    static Int32Arrays::_set set;
    struct _shape {
        inline Int32Arrays::Shape operator()(const Int32Arrays::Array& a) {
            return __array.shape(a);
        };
        inline Int32Arrays::Shape operator()(const Int32Arrays::PaddedArray& a) {
            return __array.shape(a);
        };
    };

    static Int32Arrays::_shape shape;
    struct _take {
        inline Int32Arrays::Array operator()(const Int32Arrays::Int& i, const Int32Arrays::Array& a) {
            Int32Arrays::Shape drop_sh_0 = Int32Arrays::drop_shape_elem(a, Int32Arrays::elem_int(Int32Arrays::zero()));
            Int32Arrays::Array res_array;
            if (Int32Arrays::lt(Int32Arrays::int_elem(i), Int32Arrays::zero()))
            {
                Int32Arrays::Int drop_value = Int32Arrays::elem_int(Int32Arrays::binary_add(Int32Arrays::int_elem(Int32Arrays::get_shape_elem(a, Int32Arrays::elem_int(Int32Arrays::zero()))), Int32Arrays::int_elem(i)));
                res_array = Int32Arrays::drop(drop_value, a);
            }
            else
            {
                res_array = Int32Arrays::get(a, Int32Arrays::create_index1(Int32Arrays::elem_int(Int32Arrays::zero())));
                res_array = Int32Arrays::reshape(res_array, Int32Arrays::cat_shape(Int32Arrays::create_shape1(Int32Arrays::elem_int(Int32Arrays::one())), Int32Arrays::shape(res_array)));
                Int32Arrays::Int c = Int32Arrays::elem_int(Int32Arrays::one());
                Int32Arrays::take_repeat(a, i, res_array, c);
            }
            return res_array;
        };
    };

    static Int32Arrays::_take take;
    struct _take_body {
        inline void operator()(const Int32Arrays::Array& a, const Int32Arrays::Int& i, Int32Arrays::Array& res, Int32Arrays::Int& c) {
            Int32Arrays::Index ix = Int32Arrays::create_index1(c);
            Int32Arrays::Array slice = Int32Arrays::get(a, ix);
            slice = Int32Arrays::reshape(slice, Int32Arrays::cat_shape(Int32Arrays::create_shape1(Int32Arrays::elem_int(Int32Arrays::one())), Int32Arrays::shape(slice)));
            res = Int32Arrays::cat(res, slice);
            c = Int32Arrays::elem_int(Int32Arrays::binary_add(Int32Arrays::int_elem(c), Int32Arrays::one()));
        };
    };

    static Int32Arrays::_take_body take_body;
    struct _take_cond {
        inline bool operator()(const Int32Arrays::Array& a, const Int32Arrays::Int& i, const Int32Arrays::Array& res, const Int32Arrays::Int& c) {
            return Int32Arrays::lt(Int32Arrays::int_elem(c), Int32Arrays::int_elem(i));
        };
    };

private:
    static while_loop2_2<Int32Arrays::Array, Int32Arrays::Int, Int32Arrays::Array, Int32Arrays::Int, Int32Arrays::_take_body, Int32Arrays::_take_cond> __while_loop2_28;
public:
    static Int32Arrays::_take_cond take_cond;
    struct _take_repeat {
        inline void operator()(const Int32Arrays::Array& context1, const Int32Arrays::Int& context2, Int32Arrays::Array& state1, Int32Arrays::Int& state2) {
            return __while_loop2_28.repeat(context1, context2, state1, state2);
        };
    };

    static Int32Arrays::_take_repeat take_repeat;
    struct _test_array3_2_2 {
        inline Int32Arrays::Array operator()() {
            return __array.test_array3_2_2();
        };
    };

    static Int32Arrays::_test_array3_2_2 test_array3_2_2;
    struct _test_array3_2_2F {
        inline Int32Arrays::Array operator()() {
            return __array.test_array3_2_2F();
        };
    };

    static Int32Arrays::_test_array3_2_2F test_array3_2_2F;
    struct _test_array3_3 {
        inline Int32Arrays::Array operator()() {
            return __array.test_array3_3();
        };
    };

    static Int32Arrays::_test_array3_3 test_array3_3;
    struct _test_vector2 {
        inline Int32Arrays::Array operator()() {
            return __array.test_vector2();
        };
    };

    static Int32Arrays::_test_vector2 test_vector2;
    struct _test_vector3 {
        inline Int32Arrays::Array operator()() {
            return __array.test_vector3();
        };
    };

    static Int32Arrays::_test_vector3 test_vector3;
    struct _test_vector5 {
        inline Int32Arrays::Array operator()() {
            return __array.test_vector5();
        };
    };

    static Int32Arrays::_test_vector5 test_vector5;
    struct _total {
        inline Int32Arrays::Int operator()(const Int32Arrays::Array& a) {
            return __array.total(a);
        };
        inline Int32Arrays::Int operator()(const Int32Arrays::Shape& s) {
            return __array.total(s);
        };
        inline Int32Arrays::Int operator()(const Int32Arrays::IndexContainer& ixc) {
            return __array.total(ixc);
        };
    };

    static Int32Arrays::_total total;
    struct _transpose {
        inline Int32Arrays::Array operator()(const Int32Arrays::Array& a) {
            Int32Arrays::Array transposed_array = Int32Arrays::create_array(Int32Arrays::reverse_shape(Int32Arrays::shape(a)));
            Int32Arrays::IndexContainer ix_space = Int32Arrays::create_total_indices(transposed_array);
            Int32Arrays::Int counter = Int32Arrays::elem_int(Int32Arrays::zero());
            Int32Arrays::transpose_repeat(a, ix_space, transposed_array, counter);
            return transposed_array;
        };
        inline Int32Arrays::PaddedArray operator()(const Int32Arrays::PaddedArray& a) {
            Int32Arrays::Array reshaped_array = Int32Arrays::create_array(Int32Arrays::padded_shape(a));
            Int32Arrays::PaddedArray transposed_array = Int32Arrays::create_padded_array(Int32Arrays::reverse_shape(Int32Arrays::shape(a)), Int32Arrays::reverse_shape(Int32Arrays::padded_shape(a)), reshaped_array);
            Int32Arrays::IndexContainer ix_space = Int32Arrays::create_total_indices(transposed_array);
            Int32Arrays::Int counter = Int32Arrays::elem_int(Int32Arrays::zero());
            Int32Arrays::padded_transpose_repeat(a, ix_space, transposed_array, counter);
            return transposed_array;
        };
    };

    static Int32Arrays::_transpose transpose;
    struct _transpose_body {
        inline void operator()(const Int32Arrays::Array& a, const Int32Arrays::IndexContainer& ixc, Int32Arrays::Array& res, Int32Arrays::Int& c) {
            Int32Arrays::Index current_ix = Int32Arrays::get_index_ixc(ixc, c);
            Int32Arrays::Int32 current_element = Int32Arrays::unwrap_scalar(Int32Arrays::get(a, Int32Arrays::reverse_index(current_ix)));
            Int32Arrays::set(res, current_ix, current_element);
            c = Int32Arrays::elem_int(Int32Arrays::binary_add(Int32Arrays::int_elem(c), Int32Arrays::one()));
        };
    };

    static Int32Arrays::_transpose_body transpose_body;
    struct _transpose_repeat {
        inline void operator()(const Int32Arrays::Array& context1, const Int32Arrays::IndexContainer& context2, Int32Arrays::Array& state1, Int32Arrays::Int& state2) {
            return __while_loop2_26.repeat(context1, context2, state1, state2);
        };
    };

    static Int32Arrays::_transpose_repeat transpose_repeat;
    struct _unaryMap {
        inline void operator()(const Int32Arrays::Array& a, const Int32Arrays::Index& ix) {
            assert((Int32Arrays::unwrap_scalar(Int32Arrays::get(Int32Arrays::unary_sub(a), ix))) == (Int32Arrays::unary_sub(Int32Arrays::unwrap_scalar(Int32Arrays::get(a, ix)))));
        };
    };

    static Int32Arrays::_unaryMap unaryMap;
    struct _unary_sub {
        inline Int32Arrays::Array operator()(const Int32Arrays::Array& a) {
            Int32Arrays::IndexContainer ix_space = Int32Arrays::create_total_indices(a);
            Int32Arrays::Array a_upd = a;
            Int32Arrays::Int counter = Int32Arrays::elem_int(Int32Arrays::zero());
            Int32Arrays::unary_sub_repeat(ix_space, a_upd, counter);
            return a_upd;
        };
        inline Int32Arrays::Int32 operator()(const Int32Arrays::Int32& a) {
            return __int32_utils.unary_sub(a);
        };
    };

    static Int32Arrays::_unary_sub unary_sub;
    struct _unary_sub_body {
        inline void operator()(const Int32Arrays::IndexContainer& ix_space, Int32Arrays::Array& a, Int32Arrays::Int& c) {
            Int32Arrays::Index ix = Int32Arrays::get_index_ixc(ix_space, c);
            Int32Arrays::Int32 new_value = Int32Arrays::unary_sub(Int32Arrays::unwrap_scalar(Int32Arrays::get(a, ix)));
            Int32Arrays::set(a, ix, new_value);
            c = Int32Arrays::elem_int(Int32Arrays::binary_add(Int32Arrays::int_elem(c), Int32Arrays::one()));
        };
    };

    static Int32Arrays::_unary_sub_body unary_sub_body;
    struct _unary_sub_cond {
        inline bool operator()(const Int32Arrays::IndexContainer& ix_space, const Int32Arrays::Array& a, const Int32Arrays::Int& c) {
            return Int32Arrays::lt(Int32Arrays::int_elem(c), Int32Arrays::int_elem(Int32Arrays::total(ix_space)));
        };
    };

private:
    static while_loop1_2<Int32Arrays::IndexContainer, Int32Arrays::Array, Int32Arrays::Int, Int32Arrays::_unary_sub_body, Int32Arrays::_unary_sub_cond> __while_loop1_20;
public:
    static Int32Arrays::_unary_sub_cond unary_sub_cond;
    struct _unary_sub_repeat {
        inline void operator()(const Int32Arrays::IndexContainer& context1, Int32Arrays::Array& state1, Int32Arrays::Int& state2) {
            return __while_loop1_20.repeat(context1, state1, state2);
        };
    };

    static Int32Arrays::_unary_sub_repeat unary_sub_repeat;
    struct _unwrap_scalar {
        inline Int32Arrays::Int32 operator()(const Int32Arrays::Array& a) {
            return __array.unwrap_scalar(a);
        };
    };

    static Int32Arrays::_unwrap_scalar unwrap_scalar;
    struct _upper_bound {
        inline bool operator()(const Int32Arrays::Array& a, const Int32Arrays::IndexContainer& i, const Int32Arrays::Array& res, const Int32Arrays::Int& c) {
            return Int32Arrays::lt(Int32Arrays::int_elem(c), Int32Arrays::int_elem(Int32Arrays::total(i)));
        };
    };

private:
    static while_loop2_2<Int32Arrays::Array, Int32Arrays::IndexContainer, Int32Arrays::Array, Int32Arrays::Int, Int32Arrays::_transpose_body, Int32Arrays::_upper_bound> __while_loop2_26;
public:
    static Int32Arrays::_upper_bound upper_bound;
};
} // examples
} // moa
} // mg_src
} // moa_cpp