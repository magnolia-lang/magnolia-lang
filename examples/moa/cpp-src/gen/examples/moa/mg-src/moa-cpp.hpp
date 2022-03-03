#pragma once

#include "base.hpp"
#include <cassert>


namespace examples {
namespace moa {
namespace mg_src {
namespace moa_cpp {
struct BurgerProgram {
    struct _zero {
        template <typename T>
        inline T operator()() {
            T o;
            BurgerProgram::zero0(o);
            return o;
        };
    };

    static BurgerProgram::_zero zero;
    struct _one {
        template <typename T>
        inline T operator()() {
            T o;
            BurgerProgram::one0(o);
            return o;
        };
    };

    static BurgerProgram::_one one;
private:
    static float64_utils __float64_utils;
public:
    typedef float64_utils::Float64 Float64;
    typedef array<BurgerProgram::Float64>::Index Index;
    struct _cat_index {
        inline BurgerProgram::Index operator()(const BurgerProgram::Index& i, const BurgerProgram::Index& j) {
            return __array.cat_index(i, j);
        };
    };

    static BurgerProgram::_cat_index cat_index;
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
    typedef array<BurgerProgram::Float64>::IndexContainer IndexContainer;
    struct _print_index_container {
        inline void operator()(const BurgerProgram::IndexContainer& i) {
            return __array.print_index_container(i);
        };
    };

    static BurgerProgram::_print_index_container print_index_container;
    typedef array<BurgerProgram::Float64>::Int Int;
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
    struct _print_int {
        inline void operator()(const BurgerProgram::Int& u) {
            return __array.print_int(u);
        };
    };

    static BurgerProgram::_print_int print_int;
    struct _two {
        inline BurgerProgram::Int operator()() {
            return BurgerProgram::binary_add(BurgerProgram::one.operator()<Int>(), BurgerProgram::one.operator()<Int>());
        };
    };

    static BurgerProgram::_two two;
private:
    static inline void one0(BurgerProgram::Int& o) {
        o = __array.one();
    };
    static inline void zero0(BurgerProgram::Int& o) {
        o = __array.zero();
    };
public:
    typedef array<BurgerProgram::Float64>::PaddedArray PaddedArray;
    struct _padded_dim {
        inline BurgerProgram::Int operator()(const BurgerProgram::PaddedArray& a) {
            return __array.padded_dim(a);
        };
    };

    static BurgerProgram::_padded_dim padded_dim;
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
    struct _padded_transpose_body {
        inline void operator()(const BurgerProgram::PaddedArray& a, const BurgerProgram::IndexContainer& ixc, BurgerProgram::PaddedArray& res, BurgerProgram::Int& c) {
            BurgerProgram::Index current_ix = BurgerProgram::get_index_ixc(ixc, c);
            BurgerProgram::Float64 current_element = BurgerProgram::unwrap_scalar(BurgerProgram::get(a, BurgerProgram::reverse_index(current_ix)));
            BurgerProgram::set(res, current_ix, current_element);
            c = BurgerProgram::binary_add(c, BurgerProgram::one.operator()<Int>());
        };
    };

    static BurgerProgram::_padded_transpose_body padded_transpose_body;
    struct _padded_transpose_repeat {
        inline void operator()(const BurgerProgram::PaddedArray& context1, const BurgerProgram::IndexContainer& context2, BurgerProgram::PaddedArray& state1, BurgerProgram::Int& state2) {
            return __while_loop2_25.repeat(context1, context2, state1, state2);
        };
    };

    static BurgerProgram::_padded_transpose_repeat padded_transpose_repeat;
    struct _padded_upper_bound {
        inline bool operator()(const BurgerProgram::PaddedArray& a, const BurgerProgram::IndexContainer& i, const BurgerProgram::PaddedArray& res, const BurgerProgram::Int& c) {
            return BurgerProgram::lt(c, BurgerProgram::total(i));
        };
    };

private:
    static while_loop2_2<BurgerProgram::PaddedArray, BurgerProgram::IndexContainer, BurgerProgram::PaddedArray, BurgerProgram::Int, BurgerProgram::_padded_transpose_body, BurgerProgram::_padded_upper_bound> __while_loop2_25;
public:
    static BurgerProgram::_padded_upper_bound padded_upper_bound;
    struct _print_parray {
        inline void operator()(const BurgerProgram::PaddedArray& a) {
            return __array.print_parray(a);
        };
    };

    static BurgerProgram::_print_parray print_parray;
    typedef array<BurgerProgram::Float64>::Shape Shape;
    struct _cat_shape {
        inline BurgerProgram::Shape operator()(const BurgerProgram::Shape& a, const BurgerProgram::Shape& b) {
            return __array.cat_shape(a, b);
        };
    };

    static BurgerProgram::_cat_shape cat_shape;
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
    struct _padded_drop_shape_elem {
        inline BurgerProgram::Shape operator()(const BurgerProgram::PaddedArray& a, const BurgerProgram::Int& i) {
            return __array.padded_drop_shape_elem(a, i);
        };
    };

    static BurgerProgram::_padded_drop_shape_elem padded_drop_shape_elem;
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
    static array<BurgerProgram::Float64> __array;
public:
    struct _elem_int {
        inline BurgerProgram::Int operator()(const BurgerProgram::Float64& a) {
            return __array.elem_int(a);
        };
    };

    static BurgerProgram::_elem_int elem_int;
    struct _eq {
        inline bool operator()(const BurgerProgram::Float64& a, const BurgerProgram::Float64& b) {
            return __float64_utils.eq(a, b);
        };
    };

    static BurgerProgram::_eq eq;
    struct _int_elem {
        inline BurgerProgram::Float64 operator()(const BurgerProgram::Int& a) {
            return __array.int_elem(a);
        };
    };

    static BurgerProgram::_int_elem int_elem;
    struct _print_element {
        inline void operator()(const BurgerProgram::Float64& e) {
            return __array.print_element(e);
        };
    };

    static BurgerProgram::_print_element print_element;
private:
    static inline void one0(BurgerProgram::Float64& o) {
        o = __float64_utils.one();
    };
    static inline void zero0(BurgerProgram::Float64& o) {
        o = __float64_utils.zero();
    };
public:
    typedef array<BurgerProgram::Float64>::Float Float;
    struct _abs {
        inline BurgerProgram::Float operator()(const BurgerProgram::Float& a) {
            return __array.abs(a);
        };
        inline BurgerProgram::Int operator()(const BurgerProgram::Int& a) {
            return __array.abs(a);
        };
        inline BurgerProgram::Float64 operator()(const BurgerProgram::Float64& a) {
            return __float64_utils.abs(a);
        };
    };

    static BurgerProgram::_abs abs;
    struct _elem_float {
        inline BurgerProgram::Float operator()(const BurgerProgram::Float64& e) {
            return __array.elem_float(e);
        };
    };

    static BurgerProgram::_elem_float elem_float;
    struct _float_elem {
        inline BurgerProgram::Float64 operator()(const BurgerProgram::Float& f) {
            return __array.float_elem(f);
        };
    };

    static BurgerProgram::_float_elem float_elem;
    struct _le {
        inline bool operator()(const BurgerProgram::Float& a, const BurgerProgram::Float& b) {
            return __array.le(a, b);
        };
        inline bool operator()(const BurgerProgram::Int& a, const BurgerProgram::Int& b) {
            return __array.le(a, b);
        };
        inline bool operator()(const BurgerProgram::Float64& a, const BurgerProgram::Float64& b) {
            return __float64_utils.le(a, b);
        };
    };

    static BurgerProgram::_le le;
    struct _lt {
        inline bool operator()(const BurgerProgram::Float& a, const BurgerProgram::Float& b) {
            return __array.lt(a, b);
        };
        inline bool operator()(const BurgerProgram::Int& a, const BurgerProgram::Int& b) {
            return __array.lt(a, b);
        };
        inline bool operator()(const BurgerProgram::Float64& a, const BurgerProgram::Float64& b) {
            return __float64_utils.lt(a, b);
        };
    };

    static BurgerProgram::_lt lt;
    struct _oneF {
        inline BurgerProgram::Float operator()() {
            return __array.oneF();
        };
    };

    static BurgerProgram::_oneF oneF;
    struct _print_float {
        inline void operator()(const BurgerProgram::Float& f) {
            return __array.print_float(f);
        };
    };

    static BurgerProgram::_print_float print_float;
    struct _twoF {
        inline BurgerProgram::Float operator()() {
            return BurgerProgram::binary_add(BurgerProgram::oneF(), BurgerProgram::oneF());
        };
    };

    static BurgerProgram::_twoF twoF;
    struct _zeroF {
        inline BurgerProgram::Float operator()() {
            return __array.zeroF();
        };
    };

    static BurgerProgram::_zeroF zeroF;
    typedef array<BurgerProgram::Float64>::Array Array;
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
        inline BurgerProgram::Float operator()(const BurgerProgram::Float& a, const BurgerProgram::Float& b) {
            return __array.binary_add(a, b);
        };
        inline BurgerProgram::Int operator()(const BurgerProgram::Int& a, const BurgerProgram::Int& b) {
            return __array.binary_add(a, b);
        };
        inline BurgerProgram::Float64 operator()(const BurgerProgram::Float64& a, const BurgerProgram::Float64& b) {
            return __float64_utils.binary_add(a, b);
        };
        inline BurgerProgram::Array operator()(const BurgerProgram::Float64& e, const BurgerProgram::Array& a) {
            BurgerProgram::IndexContainer ix_space = BurgerProgram::create_total_indices(a);
            BurgerProgram::Array upd_a = a;
            BurgerProgram::Int counter = BurgerProgram::zero.operator()<Int>();
            BurgerProgram::lm_plus_rep(e, ix_space, upd_a, counter);
            return upd_a;
        };
        inline BurgerProgram::Array operator()(const BurgerProgram::Array& a, const BurgerProgram::Array& b) {
            BurgerProgram::IndexContainer ix_space = BurgerProgram::create_total_indices(a);
            BurgerProgram::Array res = BurgerProgram::create_array(BurgerProgram::shape(a));
            BurgerProgram::Int counter = BurgerProgram::zero.operator()<Int>();
            BurgerProgram::bmb_plus_rep(a, b, ix_space, res, counter);
            return res;
        };
    };

    static BurgerProgram::_binary_add binary_add;
    struct _binary_sub {
        inline BurgerProgram::Float operator()(const BurgerProgram::Float& a, const BurgerProgram::Float& b) {
            return __array.binary_sub(a, b);
        };
        inline BurgerProgram::Int operator()(const BurgerProgram::Int& a, const BurgerProgram::Int& b) {
            return __array.binary_sub(a, b);
        };
        inline BurgerProgram::Float64 operator()(const BurgerProgram::Float64& a, const BurgerProgram::Float64& b) {
            return __float64_utils.binary_sub(a, b);
        };
        inline BurgerProgram::Array operator()(const BurgerProgram::Float64& e, const BurgerProgram::Array& a) {
            BurgerProgram::IndexContainer ix_space = BurgerProgram::create_total_indices(a);
            BurgerProgram::Array upd_a = a;
            BurgerProgram::Int counter = BurgerProgram::zero.operator()<Int>();
            BurgerProgram::lm_sub_rep(e, ix_space, upd_a, counter);
            return upd_a;
        };
        inline BurgerProgram::Array operator()(const BurgerProgram::Array& a, const BurgerProgram::Array& b) {
            BurgerProgram::IndexContainer ix_space = BurgerProgram::create_total_indices(a);
            BurgerProgram::Array res = BurgerProgram::create_array(BurgerProgram::shape(a));
            BurgerProgram::Int counter = BurgerProgram::zero.operator()<Int>();
            BurgerProgram::bmb_sub_rep(a, b, ix_space, res, counter);
            return res;
        };
    };

    static BurgerProgram::_binary_sub binary_sub;
    struct _bmb_div {
        inline void operator()(const BurgerProgram::Array& a, const BurgerProgram::Array& b, const BurgerProgram::IndexContainer& ix_space, BurgerProgram::Array& res, BurgerProgram::Int& c) {
            BurgerProgram::Index ix = BurgerProgram::get_index_ixc(ix_space, c);
            BurgerProgram::Float64 new_value = BurgerProgram::div(BurgerProgram::unwrap_scalar(BurgerProgram::get(a, ix)), BurgerProgram::unwrap_scalar(BurgerProgram::get(b, ix)));
            BurgerProgram::set(res, ix, new_value);
            c = BurgerProgram::binary_add(c, BurgerProgram::one.operator()<Int>());
        };
    };

    static BurgerProgram::_bmb_div bmb_div;
    struct _bmb_div_rep {
        inline void operator()(const BurgerProgram::Array& context1, const BurgerProgram::Array& context2, const BurgerProgram::IndexContainer& context3, BurgerProgram::Array& state1, BurgerProgram::Int& state2) {
            return __while_loop3_2.repeat(context1, context2, context3, state1, state2);
        };
    };

    static BurgerProgram::_bmb_div_rep bmb_div_rep;
    struct _bmb_mul {
        inline void operator()(const BurgerProgram::Array& a, const BurgerProgram::Array& b, const BurgerProgram::IndexContainer& ix_space, BurgerProgram::Array& res, BurgerProgram::Int& c) {
            BurgerProgram::Index ix = BurgerProgram::get_index_ixc(ix_space, c);
            BurgerProgram::Float64 new_value = BurgerProgram::mul(BurgerProgram::unwrap_scalar(BurgerProgram::get(a, ix)), BurgerProgram::unwrap_scalar(BurgerProgram::get(b, ix)));
            BurgerProgram::set(res, ix, new_value);
            c = BurgerProgram::binary_add(c, BurgerProgram::one.operator()<Int>());
        };
    };

    static BurgerProgram::_bmb_mul bmb_mul;
    struct _bmb_mul_rep {
        inline void operator()(const BurgerProgram::Array& context1, const BurgerProgram::Array& context2, const BurgerProgram::IndexContainer& context3, BurgerProgram::Array& state1, BurgerProgram::Int& state2) {
            return __while_loop3_20.repeat(context1, context2, context3, state1, state2);
        };
    };

    static BurgerProgram::_bmb_mul_rep bmb_mul_rep;
    struct _bmb_plus {
        inline void operator()(const BurgerProgram::Array& a, const BurgerProgram::Array& b, const BurgerProgram::IndexContainer& ix_space, BurgerProgram::Array& res, BurgerProgram::Int& c) {
            BurgerProgram::Index ix = BurgerProgram::get_index_ixc(ix_space, c);
            BurgerProgram::Float64 new_value = BurgerProgram::binary_add(BurgerProgram::unwrap_scalar(BurgerProgram::get(a, ix)), BurgerProgram::unwrap_scalar(BurgerProgram::get(b, ix)));
            BurgerProgram::set(res, ix, new_value);
            c = BurgerProgram::binary_add(c, BurgerProgram::one.operator()<Int>());
        };
    };

    static BurgerProgram::_bmb_plus bmb_plus;
    struct _bmb_plus_rep {
        inline void operator()(const BurgerProgram::Array& context1, const BurgerProgram::Array& context2, const BurgerProgram::IndexContainer& context3, BurgerProgram::Array& state1, BurgerProgram::Int& state2) {
            return __while_loop3_21.repeat(context1, context2, context3, state1, state2);
        };
    };

    static BurgerProgram::_bmb_plus_rep bmb_plus_rep;
    struct _bmb_sub {
        inline void operator()(const BurgerProgram::Array& a, const BurgerProgram::Array& b, const BurgerProgram::IndexContainer& ix_space, BurgerProgram::Array& res, BurgerProgram::Int& c) {
            BurgerProgram::Index ix = BurgerProgram::get_index_ixc(ix_space, c);
            BurgerProgram::Float64 new_value = BurgerProgram::binary_sub(BurgerProgram::unwrap_scalar(BurgerProgram::get(a, ix)), BurgerProgram::unwrap_scalar(BurgerProgram::get(b, ix)));
            BurgerProgram::set(res, ix, new_value);
            c = BurgerProgram::binary_add(c, BurgerProgram::one.operator()<Int>());
        };
    };

    static BurgerProgram::_bmb_sub bmb_sub;
    struct _bmb_sub_rep {
        inline void operator()(const BurgerProgram::Array& context1, const BurgerProgram::Array& context2, const BurgerProgram::IndexContainer& context3, BurgerProgram::Array& state1, BurgerProgram::Int& state2) {
            return __while_loop3_22.repeat(context1, context2, context3, state1, state2);
        };
    };

    static BurgerProgram::_bmb_sub_rep bmb_sub_rep;
    struct _bstep {
        inline void operator()(BurgerProgram::Array& u0, BurgerProgram::Array& u1, BurgerProgram::Array& u2, const BurgerProgram::Float& nu, const BurgerProgram::Float& dx, const BurgerProgram::Float& dt) {
            BurgerProgram::Float c0 = BurgerProgram::div(BurgerProgram::div(BurgerProgram::oneF(), BurgerProgram::twoF()), dx);
            BurgerProgram::print_float(c0);
            BurgerProgram::Float c1 = BurgerProgram::div(BurgerProgram::div(BurgerProgram::oneF(), dx), dx);
            BurgerProgram::print_float(c1);
            BurgerProgram::Float c2 = BurgerProgram::div(BurgerProgram::div(BurgerProgram::twoF(), dx), dx);
            BurgerProgram::print_float(c2);
            BurgerProgram::Float c3 = nu;
            BurgerProgram::print_float(c3);
            BurgerProgram::Float c4 = BurgerProgram::div(dt, BurgerProgram::twoF());
            BurgerProgram::print_float(c4);
            BurgerProgram::Array v0 = u0;
            BurgerProgram::Array v1 = u1;
            BurgerProgram::Array v2 = u2;
            BurgerProgram::snippet(v0, u0, u0, u1, u2, c0, c1, c2, c3, c4);
            BurgerProgram::snippet(v1, u1, u0, u1, u2, c0, c1, c2, c3, c4);
            BurgerProgram::snippet(v2, u2, u0, u1, u2, c0, c1, c2, c3, c4);
            BurgerProgram::snippet(u0, v0, v0, v1, v2, c0, c1, c2, c3, c4);
            BurgerProgram::snippet(u1, v1, v0, v1, v2, c0, c1, c2, c3, c4);
            BurgerProgram::snippet(u2, v2, v0, v1, v2, c0, c1, c2, c3, c4);
        };
    };

    static BurgerProgram::_bstep bstep;
    struct _cat {
        inline BurgerProgram::Array operator()(const BurgerProgram::Array& a, const BurgerProgram::Array& b) {
            BurgerProgram::Shape drop_s0 = BurgerProgram::drop_shape_elem(a, BurgerProgram::zero.operator()<Int>());
            BurgerProgram::Shape s0a_s0b = BurgerProgram::create_shape1(BurgerProgram::binary_add(BurgerProgram::get_shape_elem(a, BurgerProgram::zero.operator()<Int>()), BurgerProgram::get_shape_elem(b, BurgerProgram::zero.operator()<Int>())));
            BurgerProgram::Shape res_shape = BurgerProgram::cat_shape(s0a_s0b, drop_s0);
            BurgerProgram::Array res = BurgerProgram::create_array(res_shape);
            BurgerProgram::IndexContainer ixc = BurgerProgram::create_partial_indices(res, BurgerProgram::one.operator()<Int>());
            BurgerProgram::Int c = BurgerProgram::zero.operator()<Int>();
            BurgerProgram::cat_repeat(a, b, ixc, res, c);
            return res;
        };
    };

    static BurgerProgram::_cat cat;
    struct _cat_body {
        inline void operator()(const BurgerProgram::Array& a, const BurgerProgram::Array& b, const BurgerProgram::IndexContainer& ixc, BurgerProgram::Array& res, BurgerProgram::Int& c) {
            BurgerProgram::Index ix = BurgerProgram::get_index_ixc(ixc, c);
            BurgerProgram::Int s0 = BurgerProgram::get_shape_elem(a, BurgerProgram::zero.operator()<Int>());
            BurgerProgram::Int i0 = BurgerProgram::get_index_elem(ix, BurgerProgram::zero.operator()<Int>());
            if (BurgerProgram::lt(i0, s0))
            {
                BurgerProgram::set(res, ix, BurgerProgram::get(a, ix));
            }
            else
            {
                BurgerProgram::Index new_ix = BurgerProgram::create_index1(BurgerProgram::binary_sub(i0, s0));
                BurgerProgram::set(res, ix, BurgerProgram::get(b, new_ix));
            }
            c = BurgerProgram::binary_add(c, BurgerProgram::one.operator()<Int>());
        };
    };

    static BurgerProgram::_cat_body cat_body;
    struct _cat_cond {
        inline bool operator()(const BurgerProgram::Array& a, const BurgerProgram::Array& b, const BurgerProgram::IndexContainer& ixc, const BurgerProgram::Array& res, const BurgerProgram::Int& c) {
            return BurgerProgram::lt(c, BurgerProgram::total(ixc));
        };
    };

private:
    static while_loop3_2<BurgerProgram::Array, BurgerProgram::Array, BurgerProgram::IndexContainer, BurgerProgram::Array, BurgerProgram::Int, BurgerProgram::_cat_body, BurgerProgram::_cat_cond> __while_loop3_23;
public:
    static BurgerProgram::_cat_cond cat_cond;
    struct _cat_repeat {
        inline void operator()(const BurgerProgram::Array& context1, const BurgerProgram::Array& context2, const BurgerProgram::IndexContainer& context3, BurgerProgram::Array& state1, BurgerProgram::Int& state2) {
            return __while_loop3_23.repeat(context1, context2, context3, state1, state2);
        };
    };

    static BurgerProgram::_cat_repeat cat_repeat;
    struct _circular_padl {
        inline BurgerProgram::PaddedArray operator()(const BurgerProgram::Array& a, const BurgerProgram::Int& ix) {
            BurgerProgram::Array padding = BurgerProgram::get(a, BurgerProgram::create_index1(ix));
            BurgerProgram::Shape reshape_shape = BurgerProgram::cat_shape(BurgerProgram::create_shape1(BurgerProgram::one.operator()<Int>()), BurgerProgram::shape(padding));
            BurgerProgram::Array reshaped_padding = BurgerProgram::reshape(padding, reshape_shape);
            BurgerProgram::Array catenated_array = BurgerProgram::cat(reshaped_padding, a);
            BurgerProgram::Shape unpadded_shape = BurgerProgram::shape(a);
            BurgerProgram::Shape padded_shape = BurgerProgram::shape(catenated_array);
            BurgerProgram::PaddedArray res = BurgerProgram::create_padded_array(unpadded_shape, padded_shape, catenated_array);
            return res;
        };
        inline BurgerProgram::PaddedArray operator()(const BurgerProgram::PaddedArray& a, const BurgerProgram::Int& ix) {
            BurgerProgram::Array padding = BurgerProgram::get(a, BurgerProgram::create_index1(ix));
            BurgerProgram::Shape reshape_shape = BurgerProgram::cat_shape(BurgerProgram::create_shape1(BurgerProgram::one.operator()<Int>()), BurgerProgram::shape(padding));
            BurgerProgram::Array reshaped_padding = BurgerProgram::reshape(padding, reshape_shape);
            BurgerProgram::Array catenated_array = BurgerProgram::cat(reshaped_padding, BurgerProgram::padded_to_unpadded(a));
            BurgerProgram::Shape unpadded_shape = BurgerProgram::shape(a);
            BurgerProgram::Shape padded_shape = BurgerProgram::shape(catenated_array);
            BurgerProgram::PaddedArray res = BurgerProgram::create_padded_array(unpadded_shape, padded_shape, catenated_array);
            return res;
        };
    };

    static BurgerProgram::_circular_padl circular_padl;
    struct _circular_padr {
        inline BurgerProgram::PaddedArray operator()(const BurgerProgram::Array& a, const BurgerProgram::Int& ix) {
            BurgerProgram::Array padding = BurgerProgram::get(a, BurgerProgram::create_index1(ix));
            BurgerProgram::Shape reshape_shape = BurgerProgram::cat_shape(BurgerProgram::create_shape1(BurgerProgram::one.operator()<Int>()), BurgerProgram::shape(padding));
            BurgerProgram::Array reshaped_padding = BurgerProgram::reshape(padding, reshape_shape);
            BurgerProgram::Array catenated_array = BurgerProgram::cat(a, reshaped_padding);
            BurgerProgram::Shape unpadded_shape = BurgerProgram::shape(a);
            BurgerProgram::Shape padded_shape = BurgerProgram::shape(catenated_array);
            BurgerProgram::PaddedArray res = BurgerProgram::create_padded_array(unpadded_shape, padded_shape, catenated_array);
            return res;
        };
        inline BurgerProgram::PaddedArray operator()(const BurgerProgram::PaddedArray& a, const BurgerProgram::Int& ix) {
            BurgerProgram::Shape unpadded_shape = BurgerProgram::shape(a);
            BurgerProgram::Array padding = BurgerProgram::get(a, BurgerProgram::create_index1(ix));
            BurgerProgram::Shape reshape_shape = BurgerProgram::cat_shape(BurgerProgram::create_shape1(BurgerProgram::one.operator()<Int>()), BurgerProgram::shape(padding));
            BurgerProgram::Array reshaped_padding = BurgerProgram::reshape(padding, reshape_shape);
            BurgerProgram::Array catenated_array = BurgerProgram::cat(BurgerProgram::padded_to_unpadded(a), reshaped_padding);
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
        inline BurgerProgram::IndexContainer operator()(const BurgerProgram::PaddedArray& a) {
            return __array.create_total_indices(a);
        };
        inline BurgerProgram::IndexContainer operator()(const BurgerProgram::Array& a) {
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
        inline BurgerProgram::Float operator()(const BurgerProgram::Float& a, const BurgerProgram::Float& b) {
            return __array.div(a, b);
        };
        inline BurgerProgram::Int operator()(const BurgerProgram::Int& a, const BurgerProgram::Int& b) {
            return __array.div(a, b);
        };
        inline BurgerProgram::Float64 operator()(const BurgerProgram::Float64& a, const BurgerProgram::Float64& b) {
            return __float64_utils.div(a, b);
        };
        inline BurgerProgram::Array operator()(const BurgerProgram::Float64& e, const BurgerProgram::Array& a) {
            BurgerProgram::IndexContainer ix_space = BurgerProgram::create_total_indices(a);
            BurgerProgram::Array upd_a = a;
            BurgerProgram::Int counter = BurgerProgram::zero.operator()<Int>();
            BurgerProgram::lm_div_rep(e, ix_space, upd_a, counter);
            return upd_a;
        };
        inline BurgerProgram::Array operator()(const BurgerProgram::Array& a, const BurgerProgram::Array& b) {
            BurgerProgram::IndexContainer ix_space = BurgerProgram::create_total_indices(a);
            BurgerProgram::Array res = BurgerProgram::create_array(BurgerProgram::shape(a));
            BurgerProgram::Int counter = BurgerProgram::zero.operator()<Int>();
            BurgerProgram::bmb_div_rep(a, b, ix_space, res, counter);
            return res;
        };
    };

    static BurgerProgram::_div div;
    struct _drop {
        inline BurgerProgram::Array operator()(const BurgerProgram::Int& t, const BurgerProgram::Array& a) {
            BurgerProgram::Int s0 = BurgerProgram::get_shape_elem(a, BurgerProgram::zero.operator()<Int>());
            BurgerProgram::Shape drop_sh_0 = BurgerProgram::drop_shape_elem(a, BurgerProgram::zero.operator()<Int>());
            BurgerProgram::Shape res_shape = BurgerProgram::cat_shape(BurgerProgram::create_shape1(BurgerProgram::binary_sub(s0, BurgerProgram::abs(t))), drop_sh_0);
            BurgerProgram::Array res = BurgerProgram::create_array(res_shape);
            BurgerProgram::IndexContainer ixc = BurgerProgram::create_partial_indices(res, BurgerProgram::one.operator()<Int>());
            BurgerProgram::Int c = BurgerProgram::zero.operator()<Int>();
            BurgerProgram::drop_repeat(a, t, ixc, res, c);
            return res;
        };
    };

    static BurgerProgram::_drop drop;
    struct _drop_body {
        inline void operator()(const BurgerProgram::Array& a, const BurgerProgram::Int& t, const BurgerProgram::IndexContainer& ixc, BurgerProgram::Array& res, BurgerProgram::Int& c) {
            BurgerProgram::Index ix = BurgerProgram::get_index_ixc(ixc, c);
            if (BurgerProgram::le(BurgerProgram::zero.operator()<Int>(), t))
            {
                BurgerProgram::Int i0 = BurgerProgram::get_index_elem(ix, BurgerProgram::zero.operator()<Int>());
                BurgerProgram::Index new_ix = BurgerProgram::create_index1(BurgerProgram::binary_add(i0, t));
                BurgerProgram::set(res, ix, BurgerProgram::get(a, new_ix));
            }
            else
            {
                BurgerProgram::set(res, ix, BurgerProgram::get(a, ix));
            }
            c = BurgerProgram::binary_add(c, BurgerProgram::one.operator()<Int>());
        };
    };

    static BurgerProgram::_drop_body drop_body;
    struct _drop_cond {
        inline bool operator()(const BurgerProgram::Array& a, const BurgerProgram::Int& t, const BurgerProgram::IndexContainer& ixc, const BurgerProgram::Array& res, const BurgerProgram::Int& c) {
            return BurgerProgram::lt(c, BurgerProgram::total(ixc));
        };
    };

private:
    static while_loop3_2<BurgerProgram::Array, BurgerProgram::Int, BurgerProgram::IndexContainer, BurgerProgram::Array, BurgerProgram::Int, BurgerProgram::_drop_body, BurgerProgram::_drop_cond> __while_loop3_25;
public:
    static BurgerProgram::_drop_cond drop_cond;
    struct _drop_repeat {
        inline void operator()(const BurgerProgram::Array& context1, const BurgerProgram::Int& context2, const BurgerProgram::IndexContainer& context3, BurgerProgram::Array& state1, BurgerProgram::Int& state2) {
            return __while_loop3_25.repeat(context1, context2, context3, state1, state2);
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
        inline BurgerProgram::Array operator()(const BurgerProgram::PaddedArray& a, const BurgerProgram::Index& ix) {
            return __array.get(a, ix);
        };
        inline BurgerProgram::Array operator()(const BurgerProgram::Array& a, const BurgerProgram::Index& ix) {
            return __array.get(a, ix);
        };
        inline BurgerProgram::Array operator()(const BurgerProgram::Array& a, const BurgerProgram::Int& ix) {
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
    struct _leftmap_cond {
        inline bool operator()(const BurgerProgram::Float64& e, const BurgerProgram::IndexContainer& ix_space, const BurgerProgram::Array& a, const BurgerProgram::Int& c) {
            return BurgerProgram::lt(c, BurgerProgram::total(ix_space));
        };
    };

    static BurgerProgram::_leftmap_cond leftmap_cond;
    struct _lm_div_rep {
        inline void operator()(const BurgerProgram::Float64& context1, const BurgerProgram::IndexContainer& context2, BurgerProgram::Array& state1, BurgerProgram::Int& state2) {
            return __while_loop2_21.repeat(context1, context2, state1, state2);
        };
    };

    static BurgerProgram::_lm_div_rep lm_div_rep;
    struct _lm_mul_rep {
        inline void operator()(const BurgerProgram::Float64& context1, const BurgerProgram::IndexContainer& context2, BurgerProgram::Array& state1, BurgerProgram::Int& state2) {
            return __while_loop2_22.repeat(context1, context2, state1, state2);
        };
    };

    static BurgerProgram::_lm_mul_rep lm_mul_rep;
    struct _lm_plus_rep {
        inline void operator()(const BurgerProgram::Float64& context1, const BurgerProgram::IndexContainer& context2, BurgerProgram::Array& state1, BurgerProgram::Int& state2) {
            return __while_loop2_23.repeat(context1, context2, state1, state2);
        };
    };

    static BurgerProgram::_lm_plus_rep lm_plus_rep;
    struct _lm_sub_rep {
        inline void operator()(const BurgerProgram::Float64& context1, const BurgerProgram::IndexContainer& context2, BurgerProgram::Array& state1, BurgerProgram::Int& state2) {
            return __while_loop2_24.repeat(context1, context2, state1, state2);
        };
    };

    static BurgerProgram::_lm_sub_rep lm_sub_rep;
    struct _lmb_div {
        inline void operator()(const BurgerProgram::Float64& e, const BurgerProgram::IndexContainer& ix_space, BurgerProgram::Array& a, BurgerProgram::Int& c) {
            BurgerProgram::Index ix = BurgerProgram::get_index_ixc(ix_space, c);
            BurgerProgram::Float64 new_value = BurgerProgram::div(e, BurgerProgram::unwrap_scalar(BurgerProgram::get(a, ix)));
            BurgerProgram::set(a, ix, new_value);
            c = BurgerProgram::binary_add(c, BurgerProgram::one.operator()<Int>());
        };
    };

private:
    static while_loop2_2<BurgerProgram::Float64, BurgerProgram::IndexContainer, BurgerProgram::Array, BurgerProgram::Int, BurgerProgram::_lmb_div, BurgerProgram::_leftmap_cond> __while_loop2_21;
public:
    static BurgerProgram::_lmb_div lmb_div;
    struct _lmb_mul {
        inline void operator()(const BurgerProgram::Float64& e, const BurgerProgram::IndexContainer& ix_space, BurgerProgram::Array& a, BurgerProgram::Int& c) {
            BurgerProgram::Index ix = BurgerProgram::get_index_ixc(ix_space, c);
            BurgerProgram::Float64 new_value = BurgerProgram::mul(e, BurgerProgram::unwrap_scalar(BurgerProgram::get(a, ix)));
            BurgerProgram::set(a, ix, new_value);
            c = BurgerProgram::binary_add(c, BurgerProgram::one.operator()<Int>());
        };
    };

private:
    static while_loop2_2<BurgerProgram::Float64, BurgerProgram::IndexContainer, BurgerProgram::Array, BurgerProgram::Int, BurgerProgram::_lmb_mul, BurgerProgram::_leftmap_cond> __while_loop2_22;
public:
    static BurgerProgram::_lmb_mul lmb_mul;
    struct _lmb_plus {
        inline void operator()(const BurgerProgram::Float64& e, const BurgerProgram::IndexContainer& ix_space, BurgerProgram::Array& a, BurgerProgram::Int& c) {
            BurgerProgram::Index ix = BurgerProgram::get_index_ixc(ix_space, c);
            BurgerProgram::Float64 new_value = BurgerProgram::binary_add(e, BurgerProgram::unwrap_scalar(BurgerProgram::get(a, ix)));
            BurgerProgram::set(a, ix, new_value);
            c = BurgerProgram::binary_add(c, BurgerProgram::one.operator()<Int>());
        };
    };

private:
    static while_loop2_2<BurgerProgram::Float64, BurgerProgram::IndexContainer, BurgerProgram::Array, BurgerProgram::Int, BurgerProgram::_lmb_plus, BurgerProgram::_leftmap_cond> __while_loop2_23;
public:
    static BurgerProgram::_lmb_plus lmb_plus;
    struct _lmb_sub {
        inline void operator()(const BurgerProgram::Float64& e, const BurgerProgram::IndexContainer& ix_space, BurgerProgram::Array& a, BurgerProgram::Int& c) {
            BurgerProgram::Index ix = BurgerProgram::get_index_ixc(ix_space, c);
            BurgerProgram::Float64 new_value = BurgerProgram::binary_sub(e, BurgerProgram::unwrap_scalar(BurgerProgram::get(a, ix)));
            BurgerProgram::set(a, ix, new_value);
            c = BurgerProgram::binary_add(c, BurgerProgram::one.operator()<Int>());
        };
    };

private:
    static while_loop2_2<BurgerProgram::Float64, BurgerProgram::IndexContainer, BurgerProgram::Array, BurgerProgram::Int, BurgerProgram::_lmb_sub, BurgerProgram::_leftmap_cond> __while_loop2_24;
public:
    static BurgerProgram::_lmb_sub lmb_sub;
    struct _mapped_ops_cond {
        inline bool operator()(const BurgerProgram::Array& a, const BurgerProgram::Array& b, const BurgerProgram::IndexContainer& ix_space, const BurgerProgram::Array& res, const BurgerProgram::Int& c) {
            return BurgerProgram::lt(c, BurgerProgram::total(ix_space));
        };
    };

private:
    static while_loop3_2<BurgerProgram::Array, BurgerProgram::Array, BurgerProgram::IndexContainer, BurgerProgram::Array, BurgerProgram::Int, BurgerProgram::_bmb_div, BurgerProgram::_mapped_ops_cond> __while_loop3_2;
    static while_loop3_2<BurgerProgram::Array, BurgerProgram::Array, BurgerProgram::IndexContainer, BurgerProgram::Array, BurgerProgram::Int, BurgerProgram::_bmb_mul, BurgerProgram::_mapped_ops_cond> __while_loop3_20;
    static while_loop3_2<BurgerProgram::Array, BurgerProgram::Array, BurgerProgram::IndexContainer, BurgerProgram::Array, BurgerProgram::Int, BurgerProgram::_bmb_plus, BurgerProgram::_mapped_ops_cond> __while_loop3_21;
    static while_loop3_2<BurgerProgram::Array, BurgerProgram::Array, BurgerProgram::IndexContainer, BurgerProgram::Array, BurgerProgram::Int, BurgerProgram::_bmb_sub, BurgerProgram::_mapped_ops_cond> __while_loop3_22;
public:
    static BurgerProgram::_mapped_ops_cond mapped_ops_cond;
    struct _mul {
        inline BurgerProgram::Float operator()(const BurgerProgram::Float& a, const BurgerProgram::Float& b) {
            return __array.mul(a, b);
        };
        inline BurgerProgram::Int operator()(const BurgerProgram::Int& a, const BurgerProgram::Int& b) {
            return __array.mul(a, b);
        };
        inline BurgerProgram::Float64 operator()(const BurgerProgram::Float64& a, const BurgerProgram::Float64& b) {
            return __float64_utils.mul(a, b);
        };
        inline BurgerProgram::Array operator()(const BurgerProgram::Float64& e, const BurgerProgram::Array& a) {
            BurgerProgram::IndexContainer ix_space = BurgerProgram::create_total_indices(a);
            BurgerProgram::Array upd_a = a;
            BurgerProgram::Int counter = BurgerProgram::zero.operator()<Int>();
            BurgerProgram::lm_mul_rep(e, ix_space, upd_a, counter);
            return upd_a;
        };
        inline BurgerProgram::Array operator()(const BurgerProgram::Array& a, const BurgerProgram::Array& b) {
            BurgerProgram::IndexContainer ix_space = BurgerProgram::create_total_indices(a);
            BurgerProgram::Array res = BurgerProgram::create_array(BurgerProgram::shape(a));
            BurgerProgram::Int counter = BurgerProgram::zero.operator()<Int>();
            BurgerProgram::bmb_mul_rep(a, b, ix_space, res, counter);
            return res;
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
            BurgerProgram::Int counter = BurgerProgram::zero.operator()<Int>();
            BurgerProgram::reshape_repeat(input_array, new_array, counter);
            return new_array;
        };
    };

    static BurgerProgram::_reshape reshape;
    struct _reshape_body {
        inline void operator()(const BurgerProgram::Array& old_array, BurgerProgram::Array& new_array, BurgerProgram::Int& counter) {
            BurgerProgram::set(new_array, counter, BurgerProgram::unwrap_scalar(BurgerProgram::get(old_array, counter)));
            counter = BurgerProgram::binary_add(counter, BurgerProgram::one.operator()<Int>());
        };
    };

    static BurgerProgram::_reshape_body reshape_body;
    struct _reshape_cond {
        inline bool operator()(const BurgerProgram::Array& old_array, const BurgerProgram::Array& new_array, const BurgerProgram::Int& counter) {
            return BurgerProgram::lt(counter, BurgerProgram::total(new_array));
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
            BurgerProgram::Int counter = BurgerProgram::zero.operator()<Int>();
            BurgerProgram::reverse_repeat(a, valid_indices, res_array, counter);
            return res_array;
        };
    };

    static BurgerProgram::_reverse reverse;
    struct _reverse_body {
        inline void operator()(const BurgerProgram::Array& input, const BurgerProgram::IndexContainer& indices, BurgerProgram::Array& res, BurgerProgram::Int& c) {
            BurgerProgram::Index ix = BurgerProgram::get_index_ixc(indices, c);
            BurgerProgram::Float64 elem = BurgerProgram::unwrap_scalar(BurgerProgram::get(input, ix));
            BurgerProgram::Int sh_0 = BurgerProgram::get_shape_elem(input, BurgerProgram::zero.operator()<Int>());
            BurgerProgram::Int ix_0 = BurgerProgram::get_index_elem(ix, BurgerProgram::zero.operator()<Int>());
            BurgerProgram::Int new_ix_0 = BurgerProgram::binary_sub(sh_0, BurgerProgram::binary_add(ix_0, BurgerProgram::one.operator()<Int>()));
            BurgerProgram::Index new_ix = BurgerProgram::cat_index(BurgerProgram::create_index1(new_ix_0), BurgerProgram::drop_index_elem(ix, BurgerProgram::zero.operator()<Int>()));
            BurgerProgram::set(res, new_ix, elem);
            c = BurgerProgram::binary_add(c, BurgerProgram::one.operator()<Int>());
        };
    };

    static BurgerProgram::_reverse_body reverse_body;
    struct _reverse_cond {
        inline bool operator()(const BurgerProgram::Array& input, const BurgerProgram::IndexContainer& indices, const BurgerProgram::Array& res, const BurgerProgram::Int& c) {
            return BurgerProgram::lt(c, BurgerProgram::total(indices));
        };
    };

private:
    static while_loop2_2<BurgerProgram::Array, BurgerProgram::IndexContainer, BurgerProgram::Array, BurgerProgram::Int, BurgerProgram::_reverse_body, BurgerProgram::_reverse_cond> __while_loop2_2;
public:
    static BurgerProgram::_reverse_cond reverse_cond;
    struct _reverse_repeat {
        inline void operator()(const BurgerProgram::Array& context1, const BurgerProgram::IndexContainer& context2, BurgerProgram::Array& state1, BurgerProgram::Int& state2) {
            return __while_loop2_2.repeat(context1, context2, state1, state2);
        };
    };

    static BurgerProgram::_reverse_repeat reverse_repeat;
    struct _rotate {
        inline BurgerProgram::Array operator()(const BurgerProgram::Int& sigma, const BurgerProgram::Int& j, const BurgerProgram::Array& a) {
            BurgerProgram::IndexContainer ix_space = BurgerProgram::create_partial_indices(a, j);
            BurgerProgram::Array res = BurgerProgram::create_array(BurgerProgram::shape(a));
            BurgerProgram::Int c = BurgerProgram::zero.operator()<Int>();
            BurgerProgram::rotate_repeat(a, ix_space, sigma, res, c);
            return res;
        };
    };

    static BurgerProgram::_rotate rotate;
    struct _rotate_body {
        inline void operator()(const BurgerProgram::Array& a, const BurgerProgram::IndexContainer& ixc, const BurgerProgram::Int& sigma, BurgerProgram::Array& res, BurgerProgram::Int& c) {
            BurgerProgram::Index ix = BurgerProgram::get_index_ixc(ixc, c);
            if (BurgerProgram::le(BurgerProgram::zero.operator()<Int>(), sigma))
            {
                BurgerProgram::Array e1 = BurgerProgram::take(BurgerProgram::unary_sub(sigma), BurgerProgram::get(a, ix));
                BurgerProgram::Array e2 = BurgerProgram::drop(BurgerProgram::unary_sub(sigma), BurgerProgram::get(a, ix));
                BurgerProgram::set(res, ix, BurgerProgram::cat(e1, e2));
            }
            else
            {
                BurgerProgram::Array e1 = BurgerProgram::drop(sigma, BurgerProgram::get(a, ix));
                BurgerProgram::Array e2 = BurgerProgram::take(sigma, BurgerProgram::get(a, ix));
                BurgerProgram::set(res, ix, BurgerProgram::cat(e1, e2));
            }
            c = BurgerProgram::binary_add(c, BurgerProgram::one.operator()<Int>());
        };
    };

    static BurgerProgram::_rotate_body rotate_body;
    struct _rotate_cond {
        inline bool operator()(const BurgerProgram::Array& a, const BurgerProgram::IndexContainer& ixc, const BurgerProgram::Int& sigma, const BurgerProgram::Array& res, const BurgerProgram::Int& c) {
            return BurgerProgram::lt(c, BurgerProgram::total(ixc));
        };
    };

private:
    static while_loop3_2<BurgerProgram::Array, BurgerProgram::IndexContainer, BurgerProgram::Int, BurgerProgram::Array, BurgerProgram::Int, BurgerProgram::_rotate_body, BurgerProgram::_rotate_cond> __while_loop3_24;
public:
    static BurgerProgram::_rotate_cond rotate_cond;
    struct _rotate_repeat {
        inline void operator()(const BurgerProgram::Array& context1, const BurgerProgram::IndexContainer& context2, const BurgerProgram::Int& context3, BurgerProgram::Array& state1, BurgerProgram::Int& state2) {
            return __while_loop3_24.repeat(context1, context2, context3, state1, state2);
        };
    };

    static BurgerProgram::_rotate_repeat rotate_repeat;
    struct _scalarLeftMap {
        inline void operator()(const BurgerProgram::Float64& e, const BurgerProgram::Array& a, const BurgerProgram::Index& ix) {
            assert((BurgerProgram::unwrap_scalar(BurgerProgram::get(BurgerProgram::binary_add(e, a), ix))) == (BurgerProgram::binary_add(e, BurgerProgram::unwrap_scalar(BurgerProgram::get(a, ix)))));
            assert((BurgerProgram::unwrap_scalar(BurgerProgram::get(BurgerProgram::binary_sub(e, a), ix))) == (BurgerProgram::binary_sub(e, BurgerProgram::unwrap_scalar(BurgerProgram::get(a, ix)))));
            assert((BurgerProgram::unwrap_scalar(BurgerProgram::get(BurgerProgram::mul(e, a), ix))) == (BurgerProgram::mul(e, BurgerProgram::unwrap_scalar(BurgerProgram::get(a, ix)))));
            assert((BurgerProgram::unwrap_scalar(BurgerProgram::get(BurgerProgram::div(e, a), ix))) == (BurgerProgram::div(e, BurgerProgram::unwrap_scalar(BurgerProgram::get(a, ix)))));
        };
    };

    static BurgerProgram::_scalarLeftMap scalarLeftMap;
    struct _set {
        inline void operator()(BurgerProgram::PaddedArray& a, const BurgerProgram::Index& ix, const BurgerProgram::Float64& e) {
            return __array.set(a, ix, e);
        };
        inline void operator()(BurgerProgram::Array& a, const BurgerProgram::Index& ix, const BurgerProgram::Array& e) {
            return __array.set(a, ix, e);
        };
        inline void operator()(BurgerProgram::Array& a, const BurgerProgram::Index& ix, const BurgerProgram::Float64& e) {
            return __array.set(a, ix, e);
        };
        inline void operator()(BurgerProgram::Array& a, const BurgerProgram::Int& ix, const BurgerProgram::Float64& e) {
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
    struct _snippet {
        inline void operator()(BurgerProgram::Array& u, const BurgerProgram::Array& v, const BurgerProgram::Array& u0, const BurgerProgram::Array& u1, const BurgerProgram::Array& u2, const BurgerProgram::Float& c0, const BurgerProgram::Float& c1, const BurgerProgram::Float& c2, const BurgerProgram::Float& c3, const BurgerProgram::Float& c4) {
            BurgerProgram::Array shift_v = BurgerProgram::rotate(BurgerProgram::unary_sub(BurgerProgram::one.operator()<Int>()), BurgerProgram::zero.operator()<Int>(), v);
            BurgerProgram::Array d1a = BurgerProgram::mul(BurgerProgram::float_elem(BurgerProgram::unary_sub(c0)), shift_v);
            BurgerProgram::Array d2a = BurgerProgram::binary_sub(BurgerProgram::mul(BurgerProgram::float_elem(c1), shift_v), BurgerProgram::mul(BurgerProgram::float_elem(c2), u0));
            shift_v = BurgerProgram::rotate(BurgerProgram::one.operator()<Int>(), BurgerProgram::zero.operator()<Int>(), v);
            d1a = BurgerProgram::binary_add(d1a, BurgerProgram::mul(BurgerProgram::float_elem(c0), shift_v));
            d2a = BurgerProgram::binary_add(d2a, BurgerProgram::mul(BurgerProgram::float_elem(c1), shift_v));
            shift_v = BurgerProgram::rotate(BurgerProgram::unary_sub(BurgerProgram::one.operator()<Int>()), BurgerProgram::one.operator()<Int>(), v);
            BurgerProgram::Array d1b = BurgerProgram::mul(BurgerProgram::float_elem(BurgerProgram::unary_sub(c0)), shift_v);
            BurgerProgram::Array d2b = BurgerProgram::binary_sub(BurgerProgram::mul(BurgerProgram::float_elem(c1), shift_v), BurgerProgram::mul(BurgerProgram::float_elem(c2), u0));
            shift_v = BurgerProgram::rotate(BurgerProgram::one.operator()<Int>(), BurgerProgram::one.operator()<Int>(), v);
            d1b = BurgerProgram::binary_add(d1b, BurgerProgram::mul(BurgerProgram::float_elem(c0), shift_v));
            d2b = BurgerProgram::binary_add(d2b, BurgerProgram::mul(BurgerProgram::float_elem(c1), shift_v));
            shift_v = BurgerProgram::rotate(BurgerProgram::unary_sub(BurgerProgram::one.operator()<Int>()), BurgerProgram::two(), v);
            BurgerProgram::Array d1c = BurgerProgram::mul(BurgerProgram::float_elem(BurgerProgram::unary_sub(c0)), shift_v);
            BurgerProgram::Array d2c = BurgerProgram::binary_sub(BurgerProgram::mul(BurgerProgram::float_elem(c1), shift_v), BurgerProgram::mul(BurgerProgram::float_elem(c2), u0));
            shift_v = BurgerProgram::rotate(BurgerProgram::one.operator()<Int>(), BurgerProgram::two(), v);
            d1c = BurgerProgram::binary_add(d1c, BurgerProgram::mul(BurgerProgram::float_elem(c0), shift_v));
            d2c = BurgerProgram::binary_add(d2c, BurgerProgram::mul(BurgerProgram::float_elem(c1), shift_v));
            d1a = BurgerProgram::binary_add(BurgerProgram::binary_add(BurgerProgram::mul(u0, d1a), BurgerProgram::mul(u1, d1b)), BurgerProgram::mul(u2, d1c));
            d2a = BurgerProgram::binary_add(BurgerProgram::binary_add(d2a, d2b), d2c);
            u = BurgerProgram::binary_add(u, BurgerProgram::mul(BurgerProgram::float_elem(c4), BurgerProgram::binary_sub(BurgerProgram::mul(BurgerProgram::float_elem(c3), d2a), d1a)));
        };
    };

    static BurgerProgram::_snippet snippet;
    struct _take {
        inline BurgerProgram::Array operator()(const BurgerProgram::Int& t, const BurgerProgram::Array& a) {
            BurgerProgram::Shape drop_sh_0 = BurgerProgram::drop_shape_elem(a, BurgerProgram::zero.operator()<Int>());
            BurgerProgram::Array res = BurgerProgram::create_array(BurgerProgram::cat_shape(BurgerProgram::create_shape1(BurgerProgram::abs(t)), drop_sh_0));
            BurgerProgram::IndexContainer ixc = BurgerProgram::create_partial_indices(res, BurgerProgram::one.operator()<Int>());
            BurgerProgram::Int c = BurgerProgram::zero.operator()<Int>();
            BurgerProgram::take_repeat(a, t, ixc, res, c);
            return res;
        };
    };

    static BurgerProgram::_take take;
    struct _take_body {
        inline void operator()(const BurgerProgram::Array& a, const BurgerProgram::Int& t, const BurgerProgram::IndexContainer& ixc, BurgerProgram::Array& res, BurgerProgram::Int& c) {
            BurgerProgram::Index ix = BurgerProgram::get_index_ixc(ixc, c);
            if (BurgerProgram::le(BurgerProgram::zero.operator()<Int>(), t))
            {
                BurgerProgram::set(res, ix, BurgerProgram::get(a, ix));
            }
            else
            {
                BurgerProgram::Int s0 = BurgerProgram::get_shape_elem(a, BurgerProgram::zero.operator()<Int>());
                BurgerProgram::Int i0 = BurgerProgram::get_index_elem(ix, BurgerProgram::zero.operator()<Int>());
                BurgerProgram::Index new_ix = BurgerProgram::create_index1(BurgerProgram::binary_add(BurgerProgram::binary_sub(s0, BurgerProgram::abs(t)), i0));
                BurgerProgram::set(res, ix, BurgerProgram::get(a, new_ix));
            }
            c = BurgerProgram::binary_add(c, BurgerProgram::one.operator()<Int>());
        };
    };

    static BurgerProgram::_take_body take_body;
    struct _take_cond {
        inline bool operator()(const BurgerProgram::Array& a, const BurgerProgram::Int& t, const BurgerProgram::IndexContainer& ixc, const BurgerProgram::Array& res, const BurgerProgram::Int& c) {
            return BurgerProgram::lt(c, BurgerProgram::abs(t));
        };
    };

private:
    static while_loop3_2<BurgerProgram::Array, BurgerProgram::Int, BurgerProgram::IndexContainer, BurgerProgram::Array, BurgerProgram::Int, BurgerProgram::_take_body, BurgerProgram::_take_cond> __while_loop3_26;
public:
    static BurgerProgram::_take_cond take_cond;
    struct _take_repeat {
        inline void operator()(const BurgerProgram::Array& context1, const BurgerProgram::Int& context2, const BurgerProgram::IndexContainer& context3, BurgerProgram::Array& state1, BurgerProgram::Int& state2) {
            return __while_loop3_26.repeat(context1, context2, context3, state1, state2);
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
        inline BurgerProgram::Int operator()(const BurgerProgram::IndexContainer& ixc) {
            return __array.total(ixc);
        };
        inline BurgerProgram::Int operator()(const BurgerProgram::Shape& s) {
            return __array.total(s);
        };
        inline BurgerProgram::Int operator()(const BurgerProgram::Array& a) {
            return __array.total(a);
        };
    };

    static BurgerProgram::_total total;
    struct _transpose {
        inline BurgerProgram::PaddedArray operator()(const BurgerProgram::PaddedArray& a) {
            BurgerProgram::Array reshaped_array = BurgerProgram::create_array(BurgerProgram::padded_shape(a));
            BurgerProgram::PaddedArray transposed_array = BurgerProgram::create_padded_array(BurgerProgram::reverse_shape(BurgerProgram::shape(a)), BurgerProgram::reverse_shape(BurgerProgram::padded_shape(a)), reshaped_array);
            BurgerProgram::IndexContainer ix_space = BurgerProgram::create_total_indices(transposed_array);
            BurgerProgram::Int counter = BurgerProgram::zero.operator()<Int>();
            BurgerProgram::padded_transpose_repeat(a, ix_space, transposed_array, counter);
            return transposed_array;
        };
        inline BurgerProgram::Array operator()(const BurgerProgram::Array& a) {
            BurgerProgram::Array transposed_array = BurgerProgram::create_array(BurgerProgram::reverse_shape(BurgerProgram::shape(a)));
            BurgerProgram::IndexContainer ix_space = BurgerProgram::create_total_indices(transposed_array);
            BurgerProgram::Int counter = BurgerProgram::zero.operator()<Int>();
            BurgerProgram::transpose_repeat(a, ix_space, transposed_array, counter);
            return transposed_array;
        };
    };

    static BurgerProgram::_transpose transpose;
    struct _transpose_body {
        inline void operator()(const BurgerProgram::Array& a, const BurgerProgram::IndexContainer& ixc, BurgerProgram::Array& res, BurgerProgram::Int& c) {
            BurgerProgram::Index current_ix = BurgerProgram::get_index_ixc(ixc, c);
            BurgerProgram::Float64 current_element = BurgerProgram::unwrap_scalar(BurgerProgram::get(a, BurgerProgram::reverse_index(current_ix)));
            BurgerProgram::set(res, current_ix, current_element);
            c = BurgerProgram::binary_add(c, BurgerProgram::one.operator()<Int>());
        };
    };

    static BurgerProgram::_transpose_body transpose_body;
    struct _transpose_repeat {
        inline void operator()(const BurgerProgram::Array& context1, const BurgerProgram::IndexContainer& context2, BurgerProgram::Array& state1, BurgerProgram::Int& state2) {
            return __while_loop2_20.repeat(context1, context2, state1, state2);
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
        inline BurgerProgram::Float operator()(const BurgerProgram::Float& a) {
            return __array.unary_sub(a);
        };
        inline BurgerProgram::Int operator()(const BurgerProgram::Int& a) {
            return __array.unary_sub(a);
        };
        inline BurgerProgram::Float64 operator()(const BurgerProgram::Float64& a) {
            return __float64_utils.unary_sub(a);
        };
        inline BurgerProgram::Array operator()(const BurgerProgram::Array& a) {
            BurgerProgram::IndexContainer ix_space = BurgerProgram::create_total_indices(a);
            BurgerProgram::Array a_upd = a;
            BurgerProgram::Int counter = BurgerProgram::zero.operator()<Int>();
            BurgerProgram::unary_sub_repeat(ix_space, a_upd, counter);
            return a_upd;
        };
    };

    static BurgerProgram::_unary_sub unary_sub;
    struct _unary_sub_body {
        inline void operator()(const BurgerProgram::IndexContainer& ix_space, BurgerProgram::Array& a, BurgerProgram::Int& c) {
            BurgerProgram::Index ix = BurgerProgram::get_index_ixc(ix_space, c);
            BurgerProgram::Float64 new_value = BurgerProgram::unary_sub(BurgerProgram::unwrap_scalar(BurgerProgram::get(a, ix)));
            BurgerProgram::set(a, ix, new_value);
            c = BurgerProgram::binary_add(c, BurgerProgram::one.operator()<Int>());
        };
    };

    static BurgerProgram::_unary_sub_body unary_sub_body;
    struct _unary_sub_cond {
        inline bool operator()(const BurgerProgram::IndexContainer& ix_space, const BurgerProgram::Array& a, const BurgerProgram::Int& c) {
            return BurgerProgram::lt(c, BurgerProgram::total(ix_space));
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
        inline BurgerProgram::Float64 operator()(const BurgerProgram::Array& a) {
            return __array.unwrap_scalar(a);
        };
    };

    static BurgerProgram::_unwrap_scalar unwrap_scalar;
    struct _upper_bound {
        inline bool operator()(const BurgerProgram::Array& a, const BurgerProgram::IndexContainer& i, const BurgerProgram::Array& res, const BurgerProgram::Int& c) {
            return BurgerProgram::lt(c, BurgerProgram::total(i));
        };
    };

private:
    static while_loop2_2<BurgerProgram::Array, BurgerProgram::IndexContainer, BurgerProgram::Array, BurgerProgram::Int, BurgerProgram::_transpose_body, BurgerProgram::_upper_bound> __while_loop2_20;
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
    struct _zero {
        template <typename T>
        inline T operator()() {
            T o;
            Float64Arrays::zero0(o);
            return o;
        };
    };

    static Float64Arrays::_zero zero;
    struct _one {
        template <typename T>
        inline T operator()() {
            T o;
            Float64Arrays::one0(o);
            return o;
        };
    };

    static Float64Arrays::_one one;
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
private:
    static inline void one0(Float64Arrays::Int& o) {
        o = __array.one();
    };
    static inline void zero0(Float64Arrays::Int& o) {
        o = __array.zero();
    };
public:
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
            c = Float64Arrays::binary_add(c, Float64Arrays::one.operator()<Int>());
        };
    };

    static Float64Arrays::_padded_transpose_body padded_transpose_body;
    struct _padded_transpose_repeat {
        inline void operator()(const Float64Arrays::PaddedArray& context1, const Float64Arrays::IndexContainer& context2, Float64Arrays::PaddedArray& state1, Float64Arrays::Int& state2) {
            return __while_loop2_25.repeat(context1, context2, state1, state2);
        };
    };

    static Float64Arrays::_padded_transpose_repeat padded_transpose_repeat;
    struct _padded_upper_bound {
        inline bool operator()(const Float64Arrays::PaddedArray& a, const Float64Arrays::IndexContainer& i, const Float64Arrays::PaddedArray& res, const Float64Arrays::Int& c) {
            return Float64Arrays::lt(c, Float64Arrays::total(i));
        };
    };

private:
    static while_loop2_2<Float64Arrays::PaddedArray, Float64Arrays::IndexContainer, Float64Arrays::PaddedArray, Float64Arrays::Int, Float64Arrays::_padded_transpose_body, Float64Arrays::_padded_upper_bound> __while_loop2_25;
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
    struct _print_element {
        inline void operator()(const Float64Arrays::Float64& e) {
            return __array.print_element(e);
        };
    };

    static Float64Arrays::_print_element print_element;
private:
    static inline void one0(Float64Arrays::Float64& o) {
        o = __float64_utils.one();
    };
    static inline void zero0(Float64Arrays::Float64& o) {
        o = __float64_utils.zero();
    };
public:
    typedef array<Float64Arrays::Float64>::Float Float;
    struct _abs {
        inline Float64Arrays::Int operator()(const Float64Arrays::Int& a) {
            return __array.abs(a);
        };
        inline Float64Arrays::Float operator()(const Float64Arrays::Float& a) {
            return __array.abs(a);
        };
        inline Float64Arrays::Float64 operator()(const Float64Arrays::Float64& a) {
            return __float64_utils.abs(a);
        };
    };

    static Float64Arrays::_abs abs;
    struct _elem_float {
        inline Float64Arrays::Float operator()(const Float64Arrays::Float64& e) {
            return __array.elem_float(e);
        };
    };

    static Float64Arrays::_elem_float elem_float;
    struct _float_elem {
        inline Float64Arrays::Float64 operator()(const Float64Arrays::Float& f) {
            return __array.float_elem(f);
        };
    };

    static Float64Arrays::_float_elem float_elem;
    struct _le {
        inline bool operator()(const Float64Arrays::Int& a, const Float64Arrays::Int& b) {
            return __array.le(a, b);
        };
        inline bool operator()(const Float64Arrays::Float& a, const Float64Arrays::Float& b) {
            return __array.le(a, b);
        };
        inline bool operator()(const Float64Arrays::Float64& a, const Float64Arrays::Float64& b) {
            return __float64_utils.le(a, b);
        };
    };

    static Float64Arrays::_le le;
    struct _lt {
        inline bool operator()(const Float64Arrays::Float64& a, const Float64Arrays::Float64& b) {
            return __float64_utils.lt(a, b);
        };
        inline bool operator()(const Float64Arrays::Int& a, const Float64Arrays::Int& b) {
            return __array.lt(a, b);
        };
        inline bool operator()(const Float64Arrays::Float& a, const Float64Arrays::Float& b) {
            return __array.lt(a, b);
        };
    };

    static Float64Arrays::_lt lt;
    struct _oneF {
        inline Float64Arrays::Float operator()() {
            return __array.oneF();
        };
    };

    static Float64Arrays::_oneF oneF;
    struct _print_float {
        inline void operator()(const Float64Arrays::Float& f) {
            return __array.print_float(f);
        };
    };

    static Float64Arrays::_print_float print_float;
    struct _zeroF {
        inline Float64Arrays::Float operator()() {
            return __array.zeroF();
        };
    };

    static Float64Arrays::_zeroF zeroF;
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
            Float64Arrays::Array res = Float64Arrays::create_array(Float64Arrays::shape(a));
            Float64Arrays::Int counter = Float64Arrays::zero.operator()<Int>();
            Float64Arrays::bmb_plus_rep(a, b, ix_space, res, counter);
            return res;
        };
        inline Float64Arrays::Array operator()(const Float64Arrays::Float64& e, const Float64Arrays::Array& a) {
            Float64Arrays::IndexContainer ix_space = Float64Arrays::create_total_indices(a);
            Float64Arrays::Array upd_a = a;
            Float64Arrays::Int counter = Float64Arrays::zero.operator()<Int>();
            Float64Arrays::lm_plus_rep(e, ix_space, upd_a, counter);
            return upd_a;
        };
        inline Float64Arrays::Float64 operator()(const Float64Arrays::Float64& a, const Float64Arrays::Float64& b) {
            return __float64_utils.binary_add(a, b);
        };
        inline Float64Arrays::Int operator()(const Float64Arrays::Int& a, const Float64Arrays::Int& b) {
            return __array.binary_add(a, b);
        };
        inline Float64Arrays::Float operator()(const Float64Arrays::Float& a, const Float64Arrays::Float& b) {
            return __array.binary_add(a, b);
        };
    };

    static Float64Arrays::_binary_add binary_add;
    struct _binary_sub {
        inline Float64Arrays::Array operator()(const Float64Arrays::Array& a, const Float64Arrays::Array& b) {
            Float64Arrays::IndexContainer ix_space = Float64Arrays::create_total_indices(a);
            Float64Arrays::Array res = Float64Arrays::create_array(Float64Arrays::shape(a));
            Float64Arrays::Int counter = Float64Arrays::zero.operator()<Int>();
            Float64Arrays::bmb_sub_rep(a, b, ix_space, res, counter);
            return res;
        };
        inline Float64Arrays::Array operator()(const Float64Arrays::Float64& e, const Float64Arrays::Array& a) {
            Float64Arrays::IndexContainer ix_space = Float64Arrays::create_total_indices(a);
            Float64Arrays::Array upd_a = a;
            Float64Arrays::Int counter = Float64Arrays::zero.operator()<Int>();
            Float64Arrays::lm_sub_rep(e, ix_space, upd_a, counter);
            return upd_a;
        };
        inline Float64Arrays::Float64 operator()(const Float64Arrays::Float64& a, const Float64Arrays::Float64& b) {
            return __float64_utils.binary_sub(a, b);
        };
        inline Float64Arrays::Int operator()(const Float64Arrays::Int& a, const Float64Arrays::Int& b) {
            return __array.binary_sub(a, b);
        };
        inline Float64Arrays::Float operator()(const Float64Arrays::Float& a, const Float64Arrays::Float& b) {
            return __array.binary_sub(a, b);
        };
    };

    static Float64Arrays::_binary_sub binary_sub;
    struct _bmb_div {
        inline void operator()(const Float64Arrays::Array& a, const Float64Arrays::Array& b, const Float64Arrays::IndexContainer& ix_space, Float64Arrays::Array& res, Float64Arrays::Int& c) {
            Float64Arrays::Index ix = Float64Arrays::get_index_ixc(ix_space, c);
            Float64Arrays::Float64 new_value = Float64Arrays::div(Float64Arrays::unwrap_scalar(Float64Arrays::get(a, ix)), Float64Arrays::unwrap_scalar(Float64Arrays::get(b, ix)));
            Float64Arrays::set(res, ix, new_value);
            c = Float64Arrays::binary_add(c, Float64Arrays::one.operator()<Int>());
        };
    };

    static Float64Arrays::_bmb_div bmb_div;
    struct _bmb_div_rep {
        inline void operator()(const Float64Arrays::Array& context1, const Float64Arrays::Array& context2, const Float64Arrays::IndexContainer& context3, Float64Arrays::Array& state1, Float64Arrays::Int& state2) {
            return __while_loop3_2.repeat(context1, context2, context3, state1, state2);
        };
    };

    static Float64Arrays::_bmb_div_rep bmb_div_rep;
    struct _bmb_mul {
        inline void operator()(const Float64Arrays::Array& a, const Float64Arrays::Array& b, const Float64Arrays::IndexContainer& ix_space, Float64Arrays::Array& res, Float64Arrays::Int& c) {
            Float64Arrays::Index ix = Float64Arrays::get_index_ixc(ix_space, c);
            Float64Arrays::Float64 new_value = Float64Arrays::mul(Float64Arrays::unwrap_scalar(Float64Arrays::get(a, ix)), Float64Arrays::unwrap_scalar(Float64Arrays::get(b, ix)));
            Float64Arrays::set(res, ix, new_value);
            c = Float64Arrays::binary_add(c, Float64Arrays::one.operator()<Int>());
        };
    };

    static Float64Arrays::_bmb_mul bmb_mul;
    struct _bmb_mul_rep {
        inline void operator()(const Float64Arrays::Array& context1, const Float64Arrays::Array& context2, const Float64Arrays::IndexContainer& context3, Float64Arrays::Array& state1, Float64Arrays::Int& state2) {
            return __while_loop3_20.repeat(context1, context2, context3, state1, state2);
        };
    };

    static Float64Arrays::_bmb_mul_rep bmb_mul_rep;
    struct _bmb_plus {
        inline void operator()(const Float64Arrays::Array& a, const Float64Arrays::Array& b, const Float64Arrays::IndexContainer& ix_space, Float64Arrays::Array& res, Float64Arrays::Int& c) {
            Float64Arrays::Index ix = Float64Arrays::get_index_ixc(ix_space, c);
            Float64Arrays::Float64 new_value = Float64Arrays::binary_add(Float64Arrays::unwrap_scalar(Float64Arrays::get(a, ix)), Float64Arrays::unwrap_scalar(Float64Arrays::get(b, ix)));
            Float64Arrays::set(res, ix, new_value);
            c = Float64Arrays::binary_add(c, Float64Arrays::one.operator()<Int>());
        };
    };

    static Float64Arrays::_bmb_plus bmb_plus;
    struct _bmb_plus_rep {
        inline void operator()(const Float64Arrays::Array& context1, const Float64Arrays::Array& context2, const Float64Arrays::IndexContainer& context3, Float64Arrays::Array& state1, Float64Arrays::Int& state2) {
            return __while_loop3_21.repeat(context1, context2, context3, state1, state2);
        };
    };

    static Float64Arrays::_bmb_plus_rep bmb_plus_rep;
    struct _bmb_sub {
        inline void operator()(const Float64Arrays::Array& a, const Float64Arrays::Array& b, const Float64Arrays::IndexContainer& ix_space, Float64Arrays::Array& res, Float64Arrays::Int& c) {
            Float64Arrays::Index ix = Float64Arrays::get_index_ixc(ix_space, c);
            Float64Arrays::Float64 new_value = Float64Arrays::binary_sub(Float64Arrays::unwrap_scalar(Float64Arrays::get(a, ix)), Float64Arrays::unwrap_scalar(Float64Arrays::get(b, ix)));
            Float64Arrays::set(res, ix, new_value);
            c = Float64Arrays::binary_add(c, Float64Arrays::one.operator()<Int>());
        };
    };

    static Float64Arrays::_bmb_sub bmb_sub;
    struct _bmb_sub_rep {
        inline void operator()(const Float64Arrays::Array& context1, const Float64Arrays::Array& context2, const Float64Arrays::IndexContainer& context3, Float64Arrays::Array& state1, Float64Arrays::Int& state2) {
            return __while_loop3_22.repeat(context1, context2, context3, state1, state2);
        };
    };

    static Float64Arrays::_bmb_sub_rep bmb_sub_rep;
    struct _cat {
        inline Float64Arrays::Array operator()(const Float64Arrays::Array& a, const Float64Arrays::Array& b) {
            Float64Arrays::Shape drop_s0 = Float64Arrays::drop_shape_elem(a, Float64Arrays::zero.operator()<Int>());
            Float64Arrays::Shape s0a_s0b = Float64Arrays::create_shape1(Float64Arrays::binary_add(Float64Arrays::get_shape_elem(a, Float64Arrays::zero.operator()<Int>()), Float64Arrays::get_shape_elem(b, Float64Arrays::zero.operator()<Int>())));
            Float64Arrays::Shape res_shape = Float64Arrays::cat_shape(s0a_s0b, drop_s0);
            Float64Arrays::Array res = Float64Arrays::create_array(res_shape);
            Float64Arrays::IndexContainer ixc = Float64Arrays::create_partial_indices(res, Float64Arrays::one.operator()<Int>());
            Float64Arrays::Int c = Float64Arrays::zero.operator()<Int>();
            Float64Arrays::cat_repeat(a, b, ixc, res, c);
            return res;
        };
    };

    static Float64Arrays::_cat cat;
    struct _cat_body {
        inline void operator()(const Float64Arrays::Array& a, const Float64Arrays::Array& b, const Float64Arrays::IndexContainer& ixc, Float64Arrays::Array& res, Float64Arrays::Int& c) {
            Float64Arrays::Index ix = Float64Arrays::get_index_ixc(ixc, c);
            Float64Arrays::Int s0 = Float64Arrays::get_shape_elem(a, Float64Arrays::zero.operator()<Int>());
            Float64Arrays::Int i0 = Float64Arrays::get_index_elem(ix, Float64Arrays::zero.operator()<Int>());
            if (Float64Arrays::lt(i0, s0))
            {
                Float64Arrays::set(res, ix, Float64Arrays::get(a, ix));
            }
            else
            {
                Float64Arrays::Index new_ix = Float64Arrays::create_index1(Float64Arrays::binary_sub(i0, s0));
                Float64Arrays::set(res, ix, Float64Arrays::get(b, new_ix));
            }
            c = Float64Arrays::binary_add(c, Float64Arrays::one.operator()<Int>());
        };
    };

    static Float64Arrays::_cat_body cat_body;
    struct _cat_cond {
        inline bool operator()(const Float64Arrays::Array& a, const Float64Arrays::Array& b, const Float64Arrays::IndexContainer& ixc, const Float64Arrays::Array& res, const Float64Arrays::Int& c) {
            return Float64Arrays::lt(c, Float64Arrays::total(ixc));
        };
    };

private:
    static while_loop3_2<Float64Arrays::Array, Float64Arrays::Array, Float64Arrays::IndexContainer, Float64Arrays::Array, Float64Arrays::Int, Float64Arrays::_cat_body, Float64Arrays::_cat_cond> __while_loop3_23;
public:
    static Float64Arrays::_cat_cond cat_cond;
    struct _cat_repeat {
        inline void operator()(const Float64Arrays::Array& context1, const Float64Arrays::Array& context2, const Float64Arrays::IndexContainer& context3, Float64Arrays::Array& state1, Float64Arrays::Int& state2) {
            return __while_loop3_23.repeat(context1, context2, context3, state1, state2);
        };
    };

    static Float64Arrays::_cat_repeat cat_repeat;
    struct _circular_padl {
        inline Float64Arrays::PaddedArray operator()(const Float64Arrays::PaddedArray& a, const Float64Arrays::Int& ix) {
            Float64Arrays::Array padding = Float64Arrays::get(a, Float64Arrays::create_index1(ix));
            Float64Arrays::Shape reshape_shape = Float64Arrays::cat_shape(Float64Arrays::create_shape1(Float64Arrays::one.operator()<Int>()), Float64Arrays::shape(padding));
            Float64Arrays::Array reshaped_padding = Float64Arrays::reshape(padding, reshape_shape);
            Float64Arrays::Array catenated_array = Float64Arrays::cat(reshaped_padding, Float64Arrays::padded_to_unpadded(a));
            Float64Arrays::Shape unpadded_shape = Float64Arrays::shape(a);
            Float64Arrays::Shape padded_shape = Float64Arrays::shape(catenated_array);
            Float64Arrays::PaddedArray res = Float64Arrays::create_padded_array(unpadded_shape, padded_shape, catenated_array);
            return res;
        };
        inline Float64Arrays::PaddedArray operator()(const Float64Arrays::Array& a, const Float64Arrays::Int& ix) {
            Float64Arrays::Array padding = Float64Arrays::get(a, Float64Arrays::create_index1(ix));
            Float64Arrays::Shape reshape_shape = Float64Arrays::cat_shape(Float64Arrays::create_shape1(Float64Arrays::one.operator()<Int>()), Float64Arrays::shape(padding));
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
            Float64Arrays::Shape reshape_shape = Float64Arrays::cat_shape(Float64Arrays::create_shape1(Float64Arrays::one.operator()<Int>()), Float64Arrays::shape(padding));
            Float64Arrays::Array reshaped_padding = Float64Arrays::reshape(padding, reshape_shape);
            Float64Arrays::Array catenated_array = Float64Arrays::cat(Float64Arrays::padded_to_unpadded(a), reshaped_padding);
            Float64Arrays::Shape padded_shape = Float64Arrays::shape(catenated_array);
            Float64Arrays::PaddedArray res = Float64Arrays::create_padded_array(unpadded_shape, padded_shape, catenated_array);
            return res;
        };
        inline Float64Arrays::PaddedArray operator()(const Float64Arrays::Array& a, const Float64Arrays::Int& ix) {
            Float64Arrays::Array padding = Float64Arrays::get(a, Float64Arrays::create_index1(ix));
            Float64Arrays::Shape reshape_shape = Float64Arrays::cat_shape(Float64Arrays::create_shape1(Float64Arrays::one.operator()<Int>()), Float64Arrays::shape(padding));
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
            Float64Arrays::Array res = Float64Arrays::create_array(Float64Arrays::shape(a));
            Float64Arrays::Int counter = Float64Arrays::zero.operator()<Int>();
            Float64Arrays::bmb_div_rep(a, b, ix_space, res, counter);
            return res;
        };
        inline Float64Arrays::Array operator()(const Float64Arrays::Float64& e, const Float64Arrays::Array& a) {
            Float64Arrays::IndexContainer ix_space = Float64Arrays::create_total_indices(a);
            Float64Arrays::Array upd_a = a;
            Float64Arrays::Int counter = Float64Arrays::zero.operator()<Int>();
            Float64Arrays::lm_div_rep(e, ix_space, upd_a, counter);
            return upd_a;
        };
        inline Float64Arrays::Float64 operator()(const Float64Arrays::Float64& a, const Float64Arrays::Float64& b) {
            return __float64_utils.div(a, b);
        };
        inline Float64Arrays::Int operator()(const Float64Arrays::Int& a, const Float64Arrays::Int& b) {
            return __array.div(a, b);
        };
        inline Float64Arrays::Float operator()(const Float64Arrays::Float& a, const Float64Arrays::Float& b) {
            return __array.div(a, b);
        };
    };

    static Float64Arrays::_div div;
    struct _drop {
        inline Float64Arrays::Array operator()(const Float64Arrays::Int& t, const Float64Arrays::Array& a) {
            Float64Arrays::Int s0 = Float64Arrays::get_shape_elem(a, Float64Arrays::zero.operator()<Int>());
            Float64Arrays::Shape drop_sh_0 = Float64Arrays::drop_shape_elem(a, Float64Arrays::zero.operator()<Int>());
            Float64Arrays::Shape res_shape = Float64Arrays::cat_shape(Float64Arrays::create_shape1(Float64Arrays::binary_sub(s0, Float64Arrays::abs(t))), drop_sh_0);
            Float64Arrays::Array res = Float64Arrays::create_array(res_shape);
            Float64Arrays::IndexContainer ixc = Float64Arrays::create_partial_indices(res, Float64Arrays::one.operator()<Int>());
            Float64Arrays::Int c = Float64Arrays::zero.operator()<Int>();
            Float64Arrays::drop_repeat(a, t, ixc, res, c);
            return res;
        };
    };

    static Float64Arrays::_drop drop;
    struct _drop_body {
        inline void operator()(const Float64Arrays::Array& a, const Float64Arrays::Int& t, const Float64Arrays::IndexContainer& ixc, Float64Arrays::Array& res, Float64Arrays::Int& c) {
            Float64Arrays::Index ix = Float64Arrays::get_index_ixc(ixc, c);
            if (Float64Arrays::le(Float64Arrays::zero.operator()<Int>(), t))
            {
                Float64Arrays::Int i0 = Float64Arrays::get_index_elem(ix, Float64Arrays::zero.operator()<Int>());
                Float64Arrays::Index new_ix = Float64Arrays::create_index1(Float64Arrays::binary_add(i0, t));
                Float64Arrays::set(res, ix, Float64Arrays::get(a, new_ix));
            }
            else
            {
                Float64Arrays::set(res, ix, Float64Arrays::get(a, ix));
            }
            c = Float64Arrays::binary_add(c, Float64Arrays::one.operator()<Int>());
        };
    };

    static Float64Arrays::_drop_body drop_body;
    struct _drop_cond {
        inline bool operator()(const Float64Arrays::Array& a, const Float64Arrays::Int& t, const Float64Arrays::IndexContainer& ixc, const Float64Arrays::Array& res, const Float64Arrays::Int& c) {
            return Float64Arrays::lt(c, Float64Arrays::total(ixc));
        };
    };

private:
    static while_loop3_2<Float64Arrays::Array, Float64Arrays::Int, Float64Arrays::IndexContainer, Float64Arrays::Array, Float64Arrays::Int, Float64Arrays::_drop_body, Float64Arrays::_drop_cond> __while_loop3_25;
public:
    static Float64Arrays::_drop_cond drop_cond;
    struct _drop_repeat {
        inline void operator()(const Float64Arrays::Array& context1, const Float64Arrays::Int& context2, const Float64Arrays::IndexContainer& context3, Float64Arrays::Array& state1, Float64Arrays::Int& state2) {
            return __while_loop3_25.repeat(context1, context2, context3, state1, state2);
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
    struct _leftmap_cond {
        inline bool operator()(const Float64Arrays::Float64& e, const Float64Arrays::IndexContainer& ix_space, const Float64Arrays::Array& a, const Float64Arrays::Int& c) {
            return Float64Arrays::lt(c, Float64Arrays::total(ix_space));
        };
    };

    static Float64Arrays::_leftmap_cond leftmap_cond;
    struct _lm_div_rep {
        inline void operator()(const Float64Arrays::Float64& context1, const Float64Arrays::IndexContainer& context2, Float64Arrays::Array& state1, Float64Arrays::Int& state2) {
            return __while_loop2_21.repeat(context1, context2, state1, state2);
        };
    };

    static Float64Arrays::_lm_div_rep lm_div_rep;
    struct _lm_mul_rep {
        inline void operator()(const Float64Arrays::Float64& context1, const Float64Arrays::IndexContainer& context2, Float64Arrays::Array& state1, Float64Arrays::Int& state2) {
            return __while_loop2_22.repeat(context1, context2, state1, state2);
        };
    };

    static Float64Arrays::_lm_mul_rep lm_mul_rep;
    struct _lm_plus_rep {
        inline void operator()(const Float64Arrays::Float64& context1, const Float64Arrays::IndexContainer& context2, Float64Arrays::Array& state1, Float64Arrays::Int& state2) {
            return __while_loop2_23.repeat(context1, context2, state1, state2);
        };
    };

    static Float64Arrays::_lm_plus_rep lm_plus_rep;
    struct _lm_sub_rep {
        inline void operator()(const Float64Arrays::Float64& context1, const Float64Arrays::IndexContainer& context2, Float64Arrays::Array& state1, Float64Arrays::Int& state2) {
            return __while_loop2_24.repeat(context1, context2, state1, state2);
        };
    };

    static Float64Arrays::_lm_sub_rep lm_sub_rep;
    struct _lmb_div {
        inline void operator()(const Float64Arrays::Float64& e, const Float64Arrays::IndexContainer& ix_space, Float64Arrays::Array& a, Float64Arrays::Int& c) {
            Float64Arrays::Index ix = Float64Arrays::get_index_ixc(ix_space, c);
            Float64Arrays::Float64 new_value = Float64Arrays::div(e, Float64Arrays::unwrap_scalar(Float64Arrays::get(a, ix)));
            Float64Arrays::set(a, ix, new_value);
            c = Float64Arrays::binary_add(c, Float64Arrays::one.operator()<Int>());
        };
    };

private:
    static while_loop2_2<Float64Arrays::Float64, Float64Arrays::IndexContainer, Float64Arrays::Array, Float64Arrays::Int, Float64Arrays::_lmb_div, Float64Arrays::_leftmap_cond> __while_loop2_21;
public:
    static Float64Arrays::_lmb_div lmb_div;
    struct _lmb_mul {
        inline void operator()(const Float64Arrays::Float64& e, const Float64Arrays::IndexContainer& ix_space, Float64Arrays::Array& a, Float64Arrays::Int& c) {
            Float64Arrays::Index ix = Float64Arrays::get_index_ixc(ix_space, c);
            Float64Arrays::Float64 new_value = Float64Arrays::mul(e, Float64Arrays::unwrap_scalar(Float64Arrays::get(a, ix)));
            Float64Arrays::set(a, ix, new_value);
            c = Float64Arrays::binary_add(c, Float64Arrays::one.operator()<Int>());
        };
    };

private:
    static while_loop2_2<Float64Arrays::Float64, Float64Arrays::IndexContainer, Float64Arrays::Array, Float64Arrays::Int, Float64Arrays::_lmb_mul, Float64Arrays::_leftmap_cond> __while_loop2_22;
public:
    static Float64Arrays::_lmb_mul lmb_mul;
    struct _lmb_plus {
        inline void operator()(const Float64Arrays::Float64& e, const Float64Arrays::IndexContainer& ix_space, Float64Arrays::Array& a, Float64Arrays::Int& c) {
            Float64Arrays::Index ix = Float64Arrays::get_index_ixc(ix_space, c);
            Float64Arrays::Float64 new_value = Float64Arrays::binary_add(e, Float64Arrays::unwrap_scalar(Float64Arrays::get(a, ix)));
            Float64Arrays::set(a, ix, new_value);
            c = Float64Arrays::binary_add(c, Float64Arrays::one.operator()<Int>());
        };
    };

private:
    static while_loop2_2<Float64Arrays::Float64, Float64Arrays::IndexContainer, Float64Arrays::Array, Float64Arrays::Int, Float64Arrays::_lmb_plus, Float64Arrays::_leftmap_cond> __while_loop2_23;
public:
    static Float64Arrays::_lmb_plus lmb_plus;
    struct _lmb_sub {
        inline void operator()(const Float64Arrays::Float64& e, const Float64Arrays::IndexContainer& ix_space, Float64Arrays::Array& a, Float64Arrays::Int& c) {
            Float64Arrays::Index ix = Float64Arrays::get_index_ixc(ix_space, c);
            Float64Arrays::Float64 new_value = Float64Arrays::binary_sub(e, Float64Arrays::unwrap_scalar(Float64Arrays::get(a, ix)));
            Float64Arrays::set(a, ix, new_value);
            c = Float64Arrays::binary_add(c, Float64Arrays::one.operator()<Int>());
        };
    };

private:
    static while_loop2_2<Float64Arrays::Float64, Float64Arrays::IndexContainer, Float64Arrays::Array, Float64Arrays::Int, Float64Arrays::_lmb_sub, Float64Arrays::_leftmap_cond> __while_loop2_24;
public:
    static Float64Arrays::_lmb_sub lmb_sub;
    struct _mapped_ops_cond {
        inline bool operator()(const Float64Arrays::Array& a, const Float64Arrays::Array& b, const Float64Arrays::IndexContainer& ix_space, const Float64Arrays::Array& res, const Float64Arrays::Int& c) {
            return Float64Arrays::lt(c, Float64Arrays::total(ix_space));
        };
    };

private:
    static while_loop3_2<Float64Arrays::Array, Float64Arrays::Array, Float64Arrays::IndexContainer, Float64Arrays::Array, Float64Arrays::Int, Float64Arrays::_bmb_div, Float64Arrays::_mapped_ops_cond> __while_loop3_2;
    static while_loop3_2<Float64Arrays::Array, Float64Arrays::Array, Float64Arrays::IndexContainer, Float64Arrays::Array, Float64Arrays::Int, Float64Arrays::_bmb_mul, Float64Arrays::_mapped_ops_cond> __while_loop3_20;
    static while_loop3_2<Float64Arrays::Array, Float64Arrays::Array, Float64Arrays::IndexContainer, Float64Arrays::Array, Float64Arrays::Int, Float64Arrays::_bmb_plus, Float64Arrays::_mapped_ops_cond> __while_loop3_21;
    static while_loop3_2<Float64Arrays::Array, Float64Arrays::Array, Float64Arrays::IndexContainer, Float64Arrays::Array, Float64Arrays::Int, Float64Arrays::_bmb_sub, Float64Arrays::_mapped_ops_cond> __while_loop3_22;
public:
    static Float64Arrays::_mapped_ops_cond mapped_ops_cond;
    struct _mul {
        inline Float64Arrays::Array operator()(const Float64Arrays::Array& a, const Float64Arrays::Array& b) {
            Float64Arrays::IndexContainer ix_space = Float64Arrays::create_total_indices(a);
            Float64Arrays::Array res = Float64Arrays::create_array(Float64Arrays::shape(a));
            Float64Arrays::Int counter = Float64Arrays::zero.operator()<Int>();
            Float64Arrays::bmb_mul_rep(a, b, ix_space, res, counter);
            return res;
        };
        inline Float64Arrays::Array operator()(const Float64Arrays::Float64& e, const Float64Arrays::Array& a) {
            Float64Arrays::IndexContainer ix_space = Float64Arrays::create_total_indices(a);
            Float64Arrays::Array upd_a = a;
            Float64Arrays::Int counter = Float64Arrays::zero.operator()<Int>();
            Float64Arrays::lm_mul_rep(e, ix_space, upd_a, counter);
            return upd_a;
        };
        inline Float64Arrays::Float64 operator()(const Float64Arrays::Float64& a, const Float64Arrays::Float64& b) {
            return __float64_utils.mul(a, b);
        };
        inline Float64Arrays::Int operator()(const Float64Arrays::Int& a, const Float64Arrays::Int& b) {
            return __array.mul(a, b);
        };
        inline Float64Arrays::Float operator()(const Float64Arrays::Float& a, const Float64Arrays::Float& b) {
            return __array.mul(a, b);
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
            Float64Arrays::Int counter = Float64Arrays::zero.operator()<Int>();
            Float64Arrays::reshape_repeat(input_array, new_array, counter);
            return new_array;
        };
    };

    static Float64Arrays::_reshape reshape;
    struct _reshape_body {
        inline void operator()(const Float64Arrays::Array& old_array, Float64Arrays::Array& new_array, Float64Arrays::Int& counter) {
            Float64Arrays::set(new_array, counter, Float64Arrays::unwrap_scalar(Float64Arrays::get(old_array, counter)));
            counter = Float64Arrays::binary_add(counter, Float64Arrays::one.operator()<Int>());
        };
    };

    static Float64Arrays::_reshape_body reshape_body;
    struct _reshape_cond {
        inline bool operator()(const Float64Arrays::Array& old_array, const Float64Arrays::Array& new_array, const Float64Arrays::Int& counter) {
            return Float64Arrays::lt(counter, Float64Arrays::total(new_array));
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
            Float64Arrays::Int counter = Float64Arrays::zero.operator()<Int>();
            Float64Arrays::reverse_repeat(a, valid_indices, res_array, counter);
            return res_array;
        };
    };

    static Float64Arrays::_reverse reverse;
    struct _reverse_body {
        inline void operator()(const Float64Arrays::Array& input, const Float64Arrays::IndexContainer& indices, Float64Arrays::Array& res, Float64Arrays::Int& c) {
            Float64Arrays::Index ix = Float64Arrays::get_index_ixc(indices, c);
            Float64Arrays::Float64 elem = Float64Arrays::unwrap_scalar(Float64Arrays::get(input, ix));
            Float64Arrays::Int sh_0 = Float64Arrays::get_shape_elem(input, Float64Arrays::zero.operator()<Int>());
            Float64Arrays::Int ix_0 = Float64Arrays::get_index_elem(ix, Float64Arrays::zero.operator()<Int>());
            Float64Arrays::Int new_ix_0 = Float64Arrays::binary_sub(sh_0, Float64Arrays::binary_add(ix_0, Float64Arrays::one.operator()<Int>()));
            Float64Arrays::Index new_ix = Float64Arrays::cat_index(Float64Arrays::create_index1(new_ix_0), Float64Arrays::drop_index_elem(ix, Float64Arrays::zero.operator()<Int>()));
            Float64Arrays::set(res, new_ix, elem);
            c = Float64Arrays::binary_add(c, Float64Arrays::one.operator()<Int>());
        };
    };

    static Float64Arrays::_reverse_body reverse_body;
    struct _reverse_cond {
        inline bool operator()(const Float64Arrays::Array& input, const Float64Arrays::IndexContainer& indices, const Float64Arrays::Array& res, const Float64Arrays::Int& c) {
            return Float64Arrays::lt(c, Float64Arrays::total(indices));
        };
    };

private:
    static while_loop2_2<Float64Arrays::Array, Float64Arrays::IndexContainer, Float64Arrays::Array, Float64Arrays::Int, Float64Arrays::_reverse_body, Float64Arrays::_reverse_cond> __while_loop2_2;
public:
    static Float64Arrays::_reverse_cond reverse_cond;
    struct _reverse_repeat {
        inline void operator()(const Float64Arrays::Array& context1, const Float64Arrays::IndexContainer& context2, Float64Arrays::Array& state1, Float64Arrays::Int& state2) {
            return __while_loop2_2.repeat(context1, context2, state1, state2);
        };
    };

    static Float64Arrays::_reverse_repeat reverse_repeat;
    struct _rotate {
        inline Float64Arrays::Array operator()(const Float64Arrays::Int& sigma, const Float64Arrays::Int& j, const Float64Arrays::Array& a) {
            Float64Arrays::IndexContainer ix_space = Float64Arrays::create_partial_indices(a, j);
            Float64Arrays::Array res = Float64Arrays::create_array(Float64Arrays::shape(a));
            Float64Arrays::Int c = Float64Arrays::zero.operator()<Int>();
            Float64Arrays::rotate_repeat(a, ix_space, sigma, res, c);
            return res;
        };
    };

    static Float64Arrays::_rotate rotate;
    struct _rotate_body {
        inline void operator()(const Float64Arrays::Array& a, const Float64Arrays::IndexContainer& ixc, const Float64Arrays::Int& sigma, Float64Arrays::Array& res, Float64Arrays::Int& c) {
            Float64Arrays::Index ix = Float64Arrays::get_index_ixc(ixc, c);
            if (Float64Arrays::le(Float64Arrays::zero.operator()<Int>(), sigma))
            {
                Float64Arrays::Array e1 = Float64Arrays::take(Float64Arrays::unary_sub(sigma), Float64Arrays::get(a, ix));
                Float64Arrays::Array e2 = Float64Arrays::drop(Float64Arrays::unary_sub(sigma), Float64Arrays::get(a, ix));
                Float64Arrays::set(res, ix, Float64Arrays::cat(e1, e2));
            }
            else
            {
                Float64Arrays::Array e1 = Float64Arrays::drop(sigma, Float64Arrays::get(a, ix));
                Float64Arrays::Array e2 = Float64Arrays::take(sigma, Float64Arrays::get(a, ix));
                Float64Arrays::set(res, ix, Float64Arrays::cat(e1, e2));
            }
            c = Float64Arrays::binary_add(c, Float64Arrays::one.operator()<Int>());
        };
    };

    static Float64Arrays::_rotate_body rotate_body;
    struct _rotate_cond {
        inline bool operator()(const Float64Arrays::Array& a, const Float64Arrays::IndexContainer& ixc, const Float64Arrays::Int& sigma, const Float64Arrays::Array& res, const Float64Arrays::Int& c) {
            return Float64Arrays::lt(c, Float64Arrays::total(ixc));
        };
    };

private:
    static while_loop3_2<Float64Arrays::Array, Float64Arrays::IndexContainer, Float64Arrays::Int, Float64Arrays::Array, Float64Arrays::Int, Float64Arrays::_rotate_body, Float64Arrays::_rotate_cond> __while_loop3_24;
public:
    static Float64Arrays::_rotate_cond rotate_cond;
    struct _rotate_repeat {
        inline void operator()(const Float64Arrays::Array& context1, const Float64Arrays::IndexContainer& context2, const Float64Arrays::Int& context3, Float64Arrays::Array& state1, Float64Arrays::Int& state2) {
            return __while_loop3_24.repeat(context1, context2, context3, state1, state2);
        };
    };

    static Float64Arrays::_rotate_repeat rotate_repeat;
    struct _scalarLeftMap {
        inline void operator()(const Float64Arrays::Float64& e, const Float64Arrays::Array& a, const Float64Arrays::Index& ix) {
            assert((Float64Arrays::unwrap_scalar(Float64Arrays::get(Float64Arrays::binary_add(e, a), ix))) == (Float64Arrays::binary_add(e, Float64Arrays::unwrap_scalar(Float64Arrays::get(a, ix)))));
            assert((Float64Arrays::unwrap_scalar(Float64Arrays::get(Float64Arrays::binary_sub(e, a), ix))) == (Float64Arrays::binary_sub(e, Float64Arrays::unwrap_scalar(Float64Arrays::get(a, ix)))));
            assert((Float64Arrays::unwrap_scalar(Float64Arrays::get(Float64Arrays::mul(e, a), ix))) == (Float64Arrays::mul(e, Float64Arrays::unwrap_scalar(Float64Arrays::get(a, ix)))));
            assert((Float64Arrays::unwrap_scalar(Float64Arrays::get(Float64Arrays::div(e, a), ix))) == (Float64Arrays::div(e, Float64Arrays::unwrap_scalar(Float64Arrays::get(a, ix)))));
        };
    };

    static Float64Arrays::_scalarLeftMap scalarLeftMap;
    struct _set {
        inline void operator()(Float64Arrays::Array& a, const Float64Arrays::Int& ix, const Float64Arrays::Float64& e) {
            return __array.set(a, ix, e);
        };
        inline void operator()(Float64Arrays::Array& a, const Float64Arrays::Index& ix, const Float64Arrays::Float64& e) {
            return __array.set(a, ix, e);
        };
        inline void operator()(Float64Arrays::Array& a, const Float64Arrays::Index& ix, const Float64Arrays::Array& e) {
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
        inline Float64Arrays::Array operator()(const Float64Arrays::Int& t, const Float64Arrays::Array& a) {
            Float64Arrays::Shape drop_sh_0 = Float64Arrays::drop_shape_elem(a, Float64Arrays::zero.operator()<Int>());
            Float64Arrays::Array res = Float64Arrays::create_array(Float64Arrays::cat_shape(Float64Arrays::create_shape1(Float64Arrays::abs(t)), drop_sh_0));
            Float64Arrays::IndexContainer ixc = Float64Arrays::create_partial_indices(res, Float64Arrays::one.operator()<Int>());
            Float64Arrays::Int c = Float64Arrays::zero.operator()<Int>();
            Float64Arrays::take_repeat(a, t, ixc, res, c);
            return res;
        };
    };

    static Float64Arrays::_take take;
    struct _take_body {
        inline void operator()(const Float64Arrays::Array& a, const Float64Arrays::Int& t, const Float64Arrays::IndexContainer& ixc, Float64Arrays::Array& res, Float64Arrays::Int& c) {
            Float64Arrays::Index ix = Float64Arrays::get_index_ixc(ixc, c);
            if (Float64Arrays::le(Float64Arrays::zero.operator()<Int>(), t))
            {
                Float64Arrays::set(res, ix, Float64Arrays::get(a, ix));
            }
            else
            {
                Float64Arrays::Int s0 = Float64Arrays::get_shape_elem(a, Float64Arrays::zero.operator()<Int>());
                Float64Arrays::Int i0 = Float64Arrays::get_index_elem(ix, Float64Arrays::zero.operator()<Int>());
                Float64Arrays::Index new_ix = Float64Arrays::create_index1(Float64Arrays::binary_add(Float64Arrays::binary_sub(s0, Float64Arrays::abs(t)), i0));
                Float64Arrays::set(res, ix, Float64Arrays::get(a, new_ix));
            }
            c = Float64Arrays::binary_add(c, Float64Arrays::one.operator()<Int>());
        };
    };

    static Float64Arrays::_take_body take_body;
    struct _take_cond {
        inline bool operator()(const Float64Arrays::Array& a, const Float64Arrays::Int& t, const Float64Arrays::IndexContainer& ixc, const Float64Arrays::Array& res, const Float64Arrays::Int& c) {
            return Float64Arrays::lt(c, Float64Arrays::abs(t));
        };
    };

private:
    static while_loop3_2<Float64Arrays::Array, Float64Arrays::Int, Float64Arrays::IndexContainer, Float64Arrays::Array, Float64Arrays::Int, Float64Arrays::_take_body, Float64Arrays::_take_cond> __while_loop3_26;
public:
    static Float64Arrays::_take_cond take_cond;
    struct _take_repeat {
        inline void operator()(const Float64Arrays::Array& context1, const Float64Arrays::Int& context2, const Float64Arrays::IndexContainer& context3, Float64Arrays::Array& state1, Float64Arrays::Int& state2) {
            return __while_loop3_26.repeat(context1, context2, context3, state1, state2);
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
            Float64Arrays::Int counter = Float64Arrays::zero.operator()<Int>();
            Float64Arrays::transpose_repeat(a, ix_space, transposed_array, counter);
            return transposed_array;
        };
        inline Float64Arrays::PaddedArray operator()(const Float64Arrays::PaddedArray& a) {
            Float64Arrays::Array reshaped_array = Float64Arrays::create_array(Float64Arrays::padded_shape(a));
            Float64Arrays::PaddedArray transposed_array = Float64Arrays::create_padded_array(Float64Arrays::reverse_shape(Float64Arrays::shape(a)), Float64Arrays::reverse_shape(Float64Arrays::padded_shape(a)), reshaped_array);
            Float64Arrays::IndexContainer ix_space = Float64Arrays::create_total_indices(transposed_array);
            Float64Arrays::Int counter = Float64Arrays::zero.operator()<Int>();
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
            c = Float64Arrays::binary_add(c, Float64Arrays::one.operator()<Int>());
        };
    };

    static Float64Arrays::_transpose_body transpose_body;
    struct _transpose_repeat {
        inline void operator()(const Float64Arrays::Array& context1, const Float64Arrays::IndexContainer& context2, Float64Arrays::Array& state1, Float64Arrays::Int& state2) {
            return __while_loop2_20.repeat(context1, context2, state1, state2);
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
            Float64Arrays::Int counter = Float64Arrays::zero.operator()<Int>();
            Float64Arrays::unary_sub_repeat(ix_space, a_upd, counter);
            return a_upd;
        };
        inline Float64Arrays::Float64 operator()(const Float64Arrays::Float64& a) {
            return __float64_utils.unary_sub(a);
        };
        inline Float64Arrays::Int operator()(const Float64Arrays::Int& a) {
            return __array.unary_sub(a);
        };
        inline Float64Arrays::Float operator()(const Float64Arrays::Float& a) {
            return __array.unary_sub(a);
        };
    };

    static Float64Arrays::_unary_sub unary_sub;
    struct _unary_sub_body {
        inline void operator()(const Float64Arrays::IndexContainer& ix_space, Float64Arrays::Array& a, Float64Arrays::Int& c) {
            Float64Arrays::Index ix = Float64Arrays::get_index_ixc(ix_space, c);
            Float64Arrays::Float64 new_value = Float64Arrays::unary_sub(Float64Arrays::unwrap_scalar(Float64Arrays::get(a, ix)));
            Float64Arrays::set(a, ix, new_value);
            c = Float64Arrays::binary_add(c, Float64Arrays::one.operator()<Int>());
        };
    };

    static Float64Arrays::_unary_sub_body unary_sub_body;
    struct _unary_sub_cond {
        inline bool operator()(const Float64Arrays::IndexContainer& ix_space, const Float64Arrays::Array& a, const Float64Arrays::Int& c) {
            return Float64Arrays::lt(c, Float64Arrays::total(ix_space));
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
            return Float64Arrays::lt(c, Float64Arrays::total(i));
        };
    };

private:
    static while_loop2_2<Float64Arrays::Array, Float64Arrays::IndexContainer, Float64Arrays::Array, Float64Arrays::Int, Float64Arrays::_transpose_body, Float64Arrays::_upper_bound> __while_loop2_20;
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
struct Int64Arrays {
    struct _zero {
        template <typename T>
        inline T operator()() {
            T o;
            Int64Arrays::zero0(o);
            return o;
        };
    };

    static Int64Arrays::_zero zero;
    struct _one {
        template <typename T>
        inline T operator()() {
            T o;
            Int64Arrays::one0(o);
            return o;
        };
    };

    static Int64Arrays::_one one;
private:
    static int64_utils __int64_utils;
public:
    typedef int64_utils::Int64 Int64;
    typedef array<Int64Arrays::Int64>::PaddedArray PaddedArray;
    struct _print_parray {
        inline void operator()(const Int64Arrays::PaddedArray& a) {
            return __array.print_parray(a);
        };
    };

    static Int64Arrays::_print_parray print_parray;
    typedef array<Int64Arrays::Int64>::Shape Shape;
    struct _cat_shape {
        inline Int64Arrays::Shape operator()(const Int64Arrays::Shape& a, const Int64Arrays::Shape& b) {
            return __array.cat_shape(a, b);
        };
    };

    static Int64Arrays::_cat_shape cat_shape;
    struct _padded_shape {
        inline Int64Arrays::Shape operator()(const Int64Arrays::PaddedArray& a) {
            return __array.padded_shape(a);
        };
    };

    static Int64Arrays::_padded_shape padded_shape;
    struct _print_shape {
        inline void operator()(const Int64Arrays::Shape& sh) {
            return __array.print_shape(sh);
        };
    };

    static Int64Arrays::_print_shape print_shape;
    struct _reverse_shape {
        inline Int64Arrays::Shape operator()(const Int64Arrays::Shape& s) {
            return __array.reverse_shape(s);
        };
    };

    static Int64Arrays::_reverse_shape reverse_shape;
private:
    static array<Int64Arrays::Int64> __array;
public:
    struct _eq {
        inline bool operator()(const Int64Arrays::Int64& a, const Int64Arrays::Int64& b) {
            return __int64_utils.eq(a, b);
        };
    };

    static Int64Arrays::_eq eq;
    struct _print_element {
        inline void operator()(const Int64Arrays::Int64& e) {
            return __array.print_element(e);
        };
    };

    static Int64Arrays::_print_element print_element;
private:
    static inline void one0(Int64Arrays::Int64& o) {
        o = __int64_utils.one();
    };
    static inline void zero0(Int64Arrays::Int64& o) {
        o = __int64_utils.zero();
    };
public:
    typedef array<Int64Arrays::Int64>::Int Int;
    struct _create_shape1 {
        inline Int64Arrays::Shape operator()(const Int64Arrays::Int& a) {
            return __array.create_shape1(a);
        };
    };

    static Int64Arrays::_create_shape1 create_shape1;
    struct _create_shape2 {
        inline Int64Arrays::Shape operator()(const Int64Arrays::Int& a, const Int64Arrays::Int& b) {
            return __array.create_shape2(a, b);
        };
    };

    static Int64Arrays::_create_shape2 create_shape2;
    struct _create_shape3 {
        inline Int64Arrays::Shape operator()(const Int64Arrays::Int& a, const Int64Arrays::Int& b, const Int64Arrays::Int& c) {
            return __array.create_shape3(a, b, c);
        };
    };

    static Int64Arrays::_create_shape3 create_shape3;
    struct _elem_int {
        inline Int64Arrays::Int operator()(const Int64Arrays::Int64& a) {
            return __array.elem_int(a);
        };
    };

    static Int64Arrays::_elem_int elem_int;
    struct _int_elem {
        inline Int64Arrays::Int64 operator()(const Int64Arrays::Int& a) {
            return __array.int_elem(a);
        };
    };

    static Int64Arrays::_int_elem int_elem;
    struct _padded_dim {
        inline Int64Arrays::Int operator()(const Int64Arrays::PaddedArray& a) {
            return __array.padded_dim(a);
        };
    };

    static Int64Arrays::_padded_dim padded_dim;
    struct _padded_drop_shape_elem {
        inline Int64Arrays::Shape operator()(const Int64Arrays::PaddedArray& a, const Int64Arrays::Int& i) {
            return __array.padded_drop_shape_elem(a, i);
        };
    };

    static Int64Arrays::_padded_drop_shape_elem padded_drop_shape_elem;
    struct _padded_get_shape_elem {
        inline Int64Arrays::Int operator()(const Int64Arrays::PaddedArray& a, const Int64Arrays::Int& i) {
            return __array.padded_get_shape_elem(a, i);
        };
    };

    static Int64Arrays::_padded_get_shape_elem padded_get_shape_elem;
    struct _padded_total {
        inline Int64Arrays::Int operator()(const Int64Arrays::PaddedArray& a) {
            return __array.padded_total(a);
        };
    };

    static Int64Arrays::_padded_total padded_total;
    struct _print_int {
        inline void operator()(const Int64Arrays::Int& u) {
            return __array.print_int(u);
        };
    };

    static Int64Arrays::_print_int print_int;
private:
    static inline void one0(Int64Arrays::Int& o) {
        o = __array.one();
    };
    static inline void zero0(Int64Arrays::Int& o) {
        o = __array.zero();
    };
public:
    typedef array<Int64Arrays::Int64>::IndexContainer IndexContainer;
    struct _padded_transpose_body {
        inline void operator()(const Int64Arrays::PaddedArray& a, const Int64Arrays::IndexContainer& ixc, Int64Arrays::PaddedArray& res, Int64Arrays::Int& c) {
            Int64Arrays::Index current_ix = Int64Arrays::get_index_ixc(ixc, c);
            Int64Arrays::Int64 current_element = Int64Arrays::unwrap_scalar(Int64Arrays::get(a, Int64Arrays::reverse_index(current_ix)));
            Int64Arrays::set(res, current_ix, current_element);
            c = Int64Arrays::binary_add(c, Int64Arrays::one.operator()<Int>());
        };
    };

    static Int64Arrays::_padded_transpose_body padded_transpose_body;
    struct _padded_transpose_repeat {
        inline void operator()(const Int64Arrays::PaddedArray& context1, const Int64Arrays::IndexContainer& context2, Int64Arrays::PaddedArray& state1, Int64Arrays::Int& state2) {
            return __while_loop2_25.repeat(context1, context2, state1, state2);
        };
    };

    static Int64Arrays::_padded_transpose_repeat padded_transpose_repeat;
    struct _padded_upper_bound {
        inline bool operator()(const Int64Arrays::PaddedArray& a, const Int64Arrays::IndexContainer& i, const Int64Arrays::PaddedArray& res, const Int64Arrays::Int& c) {
            return Int64Arrays::lt(c, Int64Arrays::total(i));
        };
    };

private:
    static while_loop2_2<Int64Arrays::PaddedArray, Int64Arrays::IndexContainer, Int64Arrays::PaddedArray, Int64Arrays::Int, Int64Arrays::_padded_transpose_body, Int64Arrays::_padded_upper_bound> __while_loop2_25;
public:
    static Int64Arrays::_padded_upper_bound padded_upper_bound;
    struct _print_index_container {
        inline void operator()(const Int64Arrays::IndexContainer& i) {
            return __array.print_index_container(i);
        };
    };

    static Int64Arrays::_print_index_container print_index_container;
    typedef array<Int64Arrays::Int64>::Index Index;
    struct _cat_index {
        inline Int64Arrays::Index operator()(const Int64Arrays::Index& i, const Int64Arrays::Index& j) {
            return __array.cat_index(i, j);
        };
    };

    static Int64Arrays::_cat_index cat_index;
    struct _create_index1 {
        inline Int64Arrays::Index operator()(const Int64Arrays::Int& a) {
            return __array.create_index1(a);
        };
    };

    static Int64Arrays::_create_index1 create_index1;
    struct _create_index2 {
        inline Int64Arrays::Index operator()(const Int64Arrays::Int& a, const Int64Arrays::Int& b) {
            return __array.create_index2(a, b);
        };
    };

    static Int64Arrays::_create_index2 create_index2;
    struct _create_index3 {
        inline Int64Arrays::Index operator()(const Int64Arrays::Int& a, const Int64Arrays::Int& b, const Int64Arrays::Int& c) {
            return __array.create_index3(a, b, c);
        };
    };

    static Int64Arrays::_create_index3 create_index3;
    struct _drop_index_elem {
        inline Int64Arrays::Index operator()(const Int64Arrays::Index& ix, const Int64Arrays::Int& i) {
            return __array.drop_index_elem(ix, i);
        };
    };

    static Int64Arrays::_drop_index_elem drop_index_elem;
    struct _get_index_elem {
        inline Int64Arrays::Int operator()(const Int64Arrays::Index& ix, const Int64Arrays::Int& i) {
            return __array.get_index_elem(ix, i);
        };
    };

    static Int64Arrays::_get_index_elem get_index_elem;
    struct _get_index_ixc {
        inline Int64Arrays::Index operator()(const Int64Arrays::IndexContainer& ixc, const Int64Arrays::Int& ix) {
            return __array.get_index_ixc(ixc, ix);
        };
    };

    static Int64Arrays::_get_index_ixc get_index_ixc;
    struct _print_index {
        inline void operator()(const Int64Arrays::Index& i) {
            return __array.print_index(i);
        };
    };

    static Int64Arrays::_print_index print_index;
    struct _reverse_index {
        inline Int64Arrays::Index operator()(const Int64Arrays::Index& ix) {
            return __array.reverse_index(ix);
        };
    };

    static Int64Arrays::_reverse_index reverse_index;
    struct _test_index {
        inline Int64Arrays::Index operator()() {
            return __array.test_index();
        };
    };

    static Int64Arrays::_test_index test_index;
    typedef array<Int64Arrays::Int64>::Float Float;
    struct _abs {
        inline Int64Arrays::Int operator()(const Int64Arrays::Int& a) {
            return __array.abs(a);
        };
        inline Int64Arrays::Float operator()(const Int64Arrays::Float& a) {
            return __array.abs(a);
        };
        inline Int64Arrays::Int64 operator()(const Int64Arrays::Int64& a) {
            return __int64_utils.abs(a);
        };
    };

    static Int64Arrays::_abs abs;
    struct _elem_float {
        inline Int64Arrays::Float operator()(const Int64Arrays::Int64& e) {
            return __array.elem_float(e);
        };
    };

    static Int64Arrays::_elem_float elem_float;
    struct _float_elem {
        inline Int64Arrays::Int64 operator()(const Int64Arrays::Float& f) {
            return __array.float_elem(f);
        };
    };

    static Int64Arrays::_float_elem float_elem;
    struct _le {
        inline bool operator()(const Int64Arrays::Int& a, const Int64Arrays::Int& b) {
            return __array.le(a, b);
        };
        inline bool operator()(const Int64Arrays::Float& a, const Int64Arrays::Float& b) {
            return __array.le(a, b);
        };
        inline bool operator()(const Int64Arrays::Int64& a, const Int64Arrays::Int64& b) {
            return __int64_utils.le(a, b);
        };
    };

    static Int64Arrays::_le le;
    struct _lt {
        inline bool operator()(const Int64Arrays::Int64& a, const Int64Arrays::Int64& b) {
            return __int64_utils.lt(a, b);
        };
        inline bool operator()(const Int64Arrays::Int& a, const Int64Arrays::Int& b) {
            return __array.lt(a, b);
        };
        inline bool operator()(const Int64Arrays::Float& a, const Int64Arrays::Float& b) {
            return __array.lt(a, b);
        };
    };

    static Int64Arrays::_lt lt;
    struct _oneF {
        inline Int64Arrays::Float operator()() {
            return __array.oneF();
        };
    };

    static Int64Arrays::_oneF oneF;
    struct _print_float {
        inline void operator()(const Int64Arrays::Float& f) {
            return __array.print_float(f);
        };
    };

    static Int64Arrays::_print_float print_float;
    struct _zeroF {
        inline Int64Arrays::Float operator()() {
            return __array.zeroF();
        };
    };

    static Int64Arrays::_zeroF zeroF;
    typedef array<Int64Arrays::Int64>::Array Array;
    struct _binaryMap {
        inline void operator()(const Int64Arrays::Array& a, const Int64Arrays::Array& b, const Int64Arrays::Index& ix) {
            assert((Int64Arrays::unwrap_scalar(Int64Arrays::get(Int64Arrays::binary_add(a, b), ix))) == (Int64Arrays::binary_add(Int64Arrays::unwrap_scalar(Int64Arrays::get(a, ix)), Int64Arrays::unwrap_scalar(Int64Arrays::get(b, ix)))));
            assert((Int64Arrays::unwrap_scalar(Int64Arrays::get(Int64Arrays::binary_sub(a, b), ix))) == (Int64Arrays::binary_sub(Int64Arrays::unwrap_scalar(Int64Arrays::get(a, ix)), Int64Arrays::unwrap_scalar(Int64Arrays::get(b, ix)))));
            assert((Int64Arrays::unwrap_scalar(Int64Arrays::get(Int64Arrays::mul(a, b), ix))) == (Int64Arrays::mul(Int64Arrays::unwrap_scalar(Int64Arrays::get(a, ix)), Int64Arrays::unwrap_scalar(Int64Arrays::get(b, ix)))));
            assert((Int64Arrays::unwrap_scalar(Int64Arrays::get(Int64Arrays::div(a, b), ix))) == (Int64Arrays::div(Int64Arrays::unwrap_scalar(Int64Arrays::get(a, ix)), Int64Arrays::unwrap_scalar(Int64Arrays::get(b, ix)))));
        };
    };

    static Int64Arrays::_binaryMap binaryMap;
    struct _binary_add {
        inline Int64Arrays::Array operator()(const Int64Arrays::Array& a, const Int64Arrays::Array& b) {
            Int64Arrays::IndexContainer ix_space = Int64Arrays::create_total_indices(a);
            Int64Arrays::Array res = Int64Arrays::create_array(Int64Arrays::shape(a));
            Int64Arrays::Int counter = Int64Arrays::zero.operator()<Int>();
            Int64Arrays::bmb_plus_rep(a, b, ix_space, res, counter);
            return res;
        };
        inline Int64Arrays::Array operator()(const Int64Arrays::Int64& e, const Int64Arrays::Array& a) {
            Int64Arrays::IndexContainer ix_space = Int64Arrays::create_total_indices(a);
            Int64Arrays::Array upd_a = a;
            Int64Arrays::Int counter = Int64Arrays::zero.operator()<Int>();
            Int64Arrays::lm_plus_rep(e, ix_space, upd_a, counter);
            return upd_a;
        };
        inline Int64Arrays::Int64 operator()(const Int64Arrays::Int64& a, const Int64Arrays::Int64& b) {
            return __int64_utils.binary_add(a, b);
        };
        inline Int64Arrays::Int operator()(const Int64Arrays::Int& a, const Int64Arrays::Int& b) {
            return __array.binary_add(a, b);
        };
        inline Int64Arrays::Float operator()(const Int64Arrays::Float& a, const Int64Arrays::Float& b) {
            return __array.binary_add(a, b);
        };
    };

    static Int64Arrays::_binary_add binary_add;
    struct _binary_sub {
        inline Int64Arrays::Array operator()(const Int64Arrays::Array& a, const Int64Arrays::Array& b) {
            Int64Arrays::IndexContainer ix_space = Int64Arrays::create_total_indices(a);
            Int64Arrays::Array res = Int64Arrays::create_array(Int64Arrays::shape(a));
            Int64Arrays::Int counter = Int64Arrays::zero.operator()<Int>();
            Int64Arrays::bmb_sub_rep(a, b, ix_space, res, counter);
            return res;
        };
        inline Int64Arrays::Array operator()(const Int64Arrays::Int64& e, const Int64Arrays::Array& a) {
            Int64Arrays::IndexContainer ix_space = Int64Arrays::create_total_indices(a);
            Int64Arrays::Array upd_a = a;
            Int64Arrays::Int counter = Int64Arrays::zero.operator()<Int>();
            Int64Arrays::lm_sub_rep(e, ix_space, upd_a, counter);
            return upd_a;
        };
        inline Int64Arrays::Int64 operator()(const Int64Arrays::Int64& a, const Int64Arrays::Int64& b) {
            return __int64_utils.binary_sub(a, b);
        };
        inline Int64Arrays::Int operator()(const Int64Arrays::Int& a, const Int64Arrays::Int& b) {
            return __array.binary_sub(a, b);
        };
        inline Int64Arrays::Float operator()(const Int64Arrays::Float& a, const Int64Arrays::Float& b) {
            return __array.binary_sub(a, b);
        };
    };

    static Int64Arrays::_binary_sub binary_sub;
    struct _bmb_div {
        inline void operator()(const Int64Arrays::Array& a, const Int64Arrays::Array& b, const Int64Arrays::IndexContainer& ix_space, Int64Arrays::Array& res, Int64Arrays::Int& c) {
            Int64Arrays::Index ix = Int64Arrays::get_index_ixc(ix_space, c);
            Int64Arrays::Int64 new_value = Int64Arrays::div(Int64Arrays::unwrap_scalar(Int64Arrays::get(a, ix)), Int64Arrays::unwrap_scalar(Int64Arrays::get(b, ix)));
            Int64Arrays::set(res, ix, new_value);
            c = Int64Arrays::binary_add(c, Int64Arrays::one.operator()<Int>());
        };
    };

    static Int64Arrays::_bmb_div bmb_div;
    struct _bmb_div_rep {
        inline void operator()(const Int64Arrays::Array& context1, const Int64Arrays::Array& context2, const Int64Arrays::IndexContainer& context3, Int64Arrays::Array& state1, Int64Arrays::Int& state2) {
            return __while_loop3_2.repeat(context1, context2, context3, state1, state2);
        };
    };

    static Int64Arrays::_bmb_div_rep bmb_div_rep;
    struct _bmb_mul {
        inline void operator()(const Int64Arrays::Array& a, const Int64Arrays::Array& b, const Int64Arrays::IndexContainer& ix_space, Int64Arrays::Array& res, Int64Arrays::Int& c) {
            Int64Arrays::Index ix = Int64Arrays::get_index_ixc(ix_space, c);
            Int64Arrays::Int64 new_value = Int64Arrays::mul(Int64Arrays::unwrap_scalar(Int64Arrays::get(a, ix)), Int64Arrays::unwrap_scalar(Int64Arrays::get(b, ix)));
            Int64Arrays::set(res, ix, new_value);
            c = Int64Arrays::binary_add(c, Int64Arrays::one.operator()<Int>());
        };
    };

    static Int64Arrays::_bmb_mul bmb_mul;
    struct _bmb_mul_rep {
        inline void operator()(const Int64Arrays::Array& context1, const Int64Arrays::Array& context2, const Int64Arrays::IndexContainer& context3, Int64Arrays::Array& state1, Int64Arrays::Int& state2) {
            return __while_loop3_20.repeat(context1, context2, context3, state1, state2);
        };
    };

    static Int64Arrays::_bmb_mul_rep bmb_mul_rep;
    struct _bmb_plus {
        inline void operator()(const Int64Arrays::Array& a, const Int64Arrays::Array& b, const Int64Arrays::IndexContainer& ix_space, Int64Arrays::Array& res, Int64Arrays::Int& c) {
            Int64Arrays::Index ix = Int64Arrays::get_index_ixc(ix_space, c);
            Int64Arrays::Int64 new_value = Int64Arrays::binary_add(Int64Arrays::unwrap_scalar(Int64Arrays::get(a, ix)), Int64Arrays::unwrap_scalar(Int64Arrays::get(b, ix)));
            Int64Arrays::set(res, ix, new_value);
            c = Int64Arrays::binary_add(c, Int64Arrays::one.operator()<Int>());
        };
    };

    static Int64Arrays::_bmb_plus bmb_plus;
    struct _bmb_plus_rep {
        inline void operator()(const Int64Arrays::Array& context1, const Int64Arrays::Array& context2, const Int64Arrays::IndexContainer& context3, Int64Arrays::Array& state1, Int64Arrays::Int& state2) {
            return __while_loop3_21.repeat(context1, context2, context3, state1, state2);
        };
    };

    static Int64Arrays::_bmb_plus_rep bmb_plus_rep;
    struct _bmb_sub {
        inline void operator()(const Int64Arrays::Array& a, const Int64Arrays::Array& b, const Int64Arrays::IndexContainer& ix_space, Int64Arrays::Array& res, Int64Arrays::Int& c) {
            Int64Arrays::Index ix = Int64Arrays::get_index_ixc(ix_space, c);
            Int64Arrays::Int64 new_value = Int64Arrays::binary_sub(Int64Arrays::unwrap_scalar(Int64Arrays::get(a, ix)), Int64Arrays::unwrap_scalar(Int64Arrays::get(b, ix)));
            Int64Arrays::set(res, ix, new_value);
            c = Int64Arrays::binary_add(c, Int64Arrays::one.operator()<Int>());
        };
    };

    static Int64Arrays::_bmb_sub bmb_sub;
    struct _bmb_sub_rep {
        inline void operator()(const Int64Arrays::Array& context1, const Int64Arrays::Array& context2, const Int64Arrays::IndexContainer& context3, Int64Arrays::Array& state1, Int64Arrays::Int& state2) {
            return __while_loop3_22.repeat(context1, context2, context3, state1, state2);
        };
    };

    static Int64Arrays::_bmb_sub_rep bmb_sub_rep;
    struct _cat {
        inline Int64Arrays::Array operator()(const Int64Arrays::Array& a, const Int64Arrays::Array& b) {
            Int64Arrays::Shape drop_s0 = Int64Arrays::drop_shape_elem(a, Int64Arrays::zero.operator()<Int>());
            Int64Arrays::Shape s0a_s0b = Int64Arrays::create_shape1(Int64Arrays::binary_add(Int64Arrays::get_shape_elem(a, Int64Arrays::zero.operator()<Int>()), Int64Arrays::get_shape_elem(b, Int64Arrays::zero.operator()<Int>())));
            Int64Arrays::Shape res_shape = Int64Arrays::cat_shape(s0a_s0b, drop_s0);
            Int64Arrays::Array res = Int64Arrays::create_array(res_shape);
            Int64Arrays::IndexContainer ixc = Int64Arrays::create_partial_indices(res, Int64Arrays::one.operator()<Int>());
            Int64Arrays::Int c = Int64Arrays::zero.operator()<Int>();
            Int64Arrays::cat_repeat(a, b, ixc, res, c);
            return res;
        };
    };

    static Int64Arrays::_cat cat;
    struct _cat_body {
        inline void operator()(const Int64Arrays::Array& a, const Int64Arrays::Array& b, const Int64Arrays::IndexContainer& ixc, Int64Arrays::Array& res, Int64Arrays::Int& c) {
            Int64Arrays::Index ix = Int64Arrays::get_index_ixc(ixc, c);
            Int64Arrays::Int s0 = Int64Arrays::get_shape_elem(a, Int64Arrays::zero.operator()<Int>());
            Int64Arrays::Int i0 = Int64Arrays::get_index_elem(ix, Int64Arrays::zero.operator()<Int>());
            if (Int64Arrays::lt(i0, s0))
            {
                Int64Arrays::set(res, ix, Int64Arrays::get(a, ix));
            }
            else
            {
                Int64Arrays::Index new_ix = Int64Arrays::create_index1(Int64Arrays::binary_sub(i0, s0));
                Int64Arrays::set(res, ix, Int64Arrays::get(b, new_ix));
            }
            c = Int64Arrays::binary_add(c, Int64Arrays::one.operator()<Int>());
        };
    };

    static Int64Arrays::_cat_body cat_body;
    struct _cat_cond {
        inline bool operator()(const Int64Arrays::Array& a, const Int64Arrays::Array& b, const Int64Arrays::IndexContainer& ixc, const Int64Arrays::Array& res, const Int64Arrays::Int& c) {
            return Int64Arrays::lt(c, Int64Arrays::total(ixc));
        };
    };

private:
    static while_loop3_2<Int64Arrays::Array, Int64Arrays::Array, Int64Arrays::IndexContainer, Int64Arrays::Array, Int64Arrays::Int, Int64Arrays::_cat_body, Int64Arrays::_cat_cond> __while_loop3_23;
public:
    static Int64Arrays::_cat_cond cat_cond;
    struct _cat_repeat {
        inline void operator()(const Int64Arrays::Array& context1, const Int64Arrays::Array& context2, const Int64Arrays::IndexContainer& context3, Int64Arrays::Array& state1, Int64Arrays::Int& state2) {
            return __while_loop3_23.repeat(context1, context2, context3, state1, state2);
        };
    };

    static Int64Arrays::_cat_repeat cat_repeat;
    struct _circular_padl {
        inline Int64Arrays::PaddedArray operator()(const Int64Arrays::PaddedArray& a, const Int64Arrays::Int& ix) {
            Int64Arrays::Array padding = Int64Arrays::get(a, Int64Arrays::create_index1(ix));
            Int64Arrays::Shape reshape_shape = Int64Arrays::cat_shape(Int64Arrays::create_shape1(Int64Arrays::one.operator()<Int>()), Int64Arrays::shape(padding));
            Int64Arrays::Array reshaped_padding = Int64Arrays::reshape(padding, reshape_shape);
            Int64Arrays::Array catenated_array = Int64Arrays::cat(reshaped_padding, Int64Arrays::padded_to_unpadded(a));
            Int64Arrays::Shape unpadded_shape = Int64Arrays::shape(a);
            Int64Arrays::Shape padded_shape = Int64Arrays::shape(catenated_array);
            Int64Arrays::PaddedArray res = Int64Arrays::create_padded_array(unpadded_shape, padded_shape, catenated_array);
            return res;
        };
        inline Int64Arrays::PaddedArray operator()(const Int64Arrays::Array& a, const Int64Arrays::Int& ix) {
            Int64Arrays::Array padding = Int64Arrays::get(a, Int64Arrays::create_index1(ix));
            Int64Arrays::Shape reshape_shape = Int64Arrays::cat_shape(Int64Arrays::create_shape1(Int64Arrays::one.operator()<Int>()), Int64Arrays::shape(padding));
            Int64Arrays::Array reshaped_padding = Int64Arrays::reshape(padding, reshape_shape);
            Int64Arrays::Array catenated_array = Int64Arrays::cat(reshaped_padding, a);
            Int64Arrays::Shape unpadded_shape = Int64Arrays::shape(a);
            Int64Arrays::Shape padded_shape = Int64Arrays::shape(catenated_array);
            Int64Arrays::PaddedArray res = Int64Arrays::create_padded_array(unpadded_shape, padded_shape, catenated_array);
            return res;
        };
    };

    static Int64Arrays::_circular_padl circular_padl;
    struct _circular_padr {
        inline Int64Arrays::PaddedArray operator()(const Int64Arrays::PaddedArray& a, const Int64Arrays::Int& ix) {
            Int64Arrays::Shape unpadded_shape = Int64Arrays::shape(a);
            Int64Arrays::Array padding = Int64Arrays::get(a, Int64Arrays::create_index1(ix));
            Int64Arrays::Shape reshape_shape = Int64Arrays::cat_shape(Int64Arrays::create_shape1(Int64Arrays::one.operator()<Int>()), Int64Arrays::shape(padding));
            Int64Arrays::Array reshaped_padding = Int64Arrays::reshape(padding, reshape_shape);
            Int64Arrays::Array catenated_array = Int64Arrays::cat(Int64Arrays::padded_to_unpadded(a), reshaped_padding);
            Int64Arrays::Shape padded_shape = Int64Arrays::shape(catenated_array);
            Int64Arrays::PaddedArray res = Int64Arrays::create_padded_array(unpadded_shape, padded_shape, catenated_array);
            return res;
        };
        inline Int64Arrays::PaddedArray operator()(const Int64Arrays::Array& a, const Int64Arrays::Int& ix) {
            Int64Arrays::Array padding = Int64Arrays::get(a, Int64Arrays::create_index1(ix));
            Int64Arrays::Shape reshape_shape = Int64Arrays::cat_shape(Int64Arrays::create_shape1(Int64Arrays::one.operator()<Int>()), Int64Arrays::shape(padding));
            Int64Arrays::Array reshaped_padding = Int64Arrays::reshape(padding, reshape_shape);
            Int64Arrays::Array catenated_array = Int64Arrays::cat(a, reshaped_padding);
            Int64Arrays::Shape unpadded_shape = Int64Arrays::shape(a);
            Int64Arrays::Shape padded_shape = Int64Arrays::shape(catenated_array);
            Int64Arrays::PaddedArray res = Int64Arrays::create_padded_array(unpadded_shape, padded_shape, catenated_array);
            return res;
        };
    };

    static Int64Arrays::_circular_padr circular_padr;
    struct _create_array {
        inline Int64Arrays::Array operator()(const Int64Arrays::Shape& sh) {
            return __array.create_array(sh);
        };
    };

    static Int64Arrays::_create_array create_array;
    struct _create_padded_array {
        inline Int64Arrays::PaddedArray operator()(const Int64Arrays::Shape& unpadded_shape, const Int64Arrays::Shape& padded_shape, const Int64Arrays::Array& padded_array) {
            return __array.create_padded_array(unpadded_shape, padded_shape, padded_array);
        };
    };

    static Int64Arrays::_create_padded_array create_padded_array;
    struct _create_partial_indices {
        inline Int64Arrays::IndexContainer operator()(const Int64Arrays::Array& a, const Int64Arrays::Int& i) {
            return __array.create_partial_indices(a, i);
        };
    };

    static Int64Arrays::_create_partial_indices create_partial_indices;
    struct _create_total_indices {
        inline Int64Arrays::IndexContainer operator()(const Int64Arrays::Array& a) {
            return __array.create_total_indices(a);
        };
        inline Int64Arrays::IndexContainer operator()(const Int64Arrays::PaddedArray& a) {
            return __array.create_total_indices(a);
        };
    };

    static Int64Arrays::_create_total_indices create_total_indices;
    struct _dim {
        inline Int64Arrays::Int operator()(const Int64Arrays::Array& a) {
            return __array.dim(a);
        };
    };

    static Int64Arrays::_dim dim;
    struct _div {
        inline Int64Arrays::Array operator()(const Int64Arrays::Array& a, const Int64Arrays::Array& b) {
            Int64Arrays::IndexContainer ix_space = Int64Arrays::create_total_indices(a);
            Int64Arrays::Array res = Int64Arrays::create_array(Int64Arrays::shape(a));
            Int64Arrays::Int counter = Int64Arrays::zero.operator()<Int>();
            Int64Arrays::bmb_div_rep(a, b, ix_space, res, counter);
            return res;
        };
        inline Int64Arrays::Array operator()(const Int64Arrays::Int64& e, const Int64Arrays::Array& a) {
            Int64Arrays::IndexContainer ix_space = Int64Arrays::create_total_indices(a);
            Int64Arrays::Array upd_a = a;
            Int64Arrays::Int counter = Int64Arrays::zero.operator()<Int>();
            Int64Arrays::lm_div_rep(e, ix_space, upd_a, counter);
            return upd_a;
        };
        inline Int64Arrays::Int64 operator()(const Int64Arrays::Int64& a, const Int64Arrays::Int64& b) {
            return __int64_utils.div(a, b);
        };
        inline Int64Arrays::Int operator()(const Int64Arrays::Int& a, const Int64Arrays::Int& b) {
            return __array.div(a, b);
        };
        inline Int64Arrays::Float operator()(const Int64Arrays::Float& a, const Int64Arrays::Float& b) {
            return __array.div(a, b);
        };
    };

    static Int64Arrays::_div div;
    struct _drop {
        inline Int64Arrays::Array operator()(const Int64Arrays::Int& t, const Int64Arrays::Array& a) {
            Int64Arrays::Int s0 = Int64Arrays::get_shape_elem(a, Int64Arrays::zero.operator()<Int>());
            Int64Arrays::Shape drop_sh_0 = Int64Arrays::drop_shape_elem(a, Int64Arrays::zero.operator()<Int>());
            Int64Arrays::Shape res_shape = Int64Arrays::cat_shape(Int64Arrays::create_shape1(Int64Arrays::binary_sub(s0, Int64Arrays::abs(t))), drop_sh_0);
            Int64Arrays::Array res = Int64Arrays::create_array(res_shape);
            Int64Arrays::IndexContainer ixc = Int64Arrays::create_partial_indices(res, Int64Arrays::one.operator()<Int>());
            Int64Arrays::Int c = Int64Arrays::zero.operator()<Int>();
            Int64Arrays::drop_repeat(a, t, ixc, res, c);
            return res;
        };
    };

    static Int64Arrays::_drop drop;
    struct _drop_body {
        inline void operator()(const Int64Arrays::Array& a, const Int64Arrays::Int& t, const Int64Arrays::IndexContainer& ixc, Int64Arrays::Array& res, Int64Arrays::Int& c) {
            Int64Arrays::Index ix = Int64Arrays::get_index_ixc(ixc, c);
            if (Int64Arrays::le(Int64Arrays::zero.operator()<Int>(), t))
            {
                Int64Arrays::Int i0 = Int64Arrays::get_index_elem(ix, Int64Arrays::zero.operator()<Int>());
                Int64Arrays::Index new_ix = Int64Arrays::create_index1(Int64Arrays::binary_add(i0, t));
                Int64Arrays::set(res, ix, Int64Arrays::get(a, new_ix));
            }
            else
            {
                Int64Arrays::set(res, ix, Int64Arrays::get(a, ix));
            }
            c = Int64Arrays::binary_add(c, Int64Arrays::one.operator()<Int>());
        };
    };

    static Int64Arrays::_drop_body drop_body;
    struct _drop_cond {
        inline bool operator()(const Int64Arrays::Array& a, const Int64Arrays::Int& t, const Int64Arrays::IndexContainer& ixc, const Int64Arrays::Array& res, const Int64Arrays::Int& c) {
            return Int64Arrays::lt(c, Int64Arrays::total(ixc));
        };
    };

private:
    static while_loop3_2<Int64Arrays::Array, Int64Arrays::Int, Int64Arrays::IndexContainer, Int64Arrays::Array, Int64Arrays::Int, Int64Arrays::_drop_body, Int64Arrays::_drop_cond> __while_loop3_25;
public:
    static Int64Arrays::_drop_cond drop_cond;
    struct _drop_repeat {
        inline void operator()(const Int64Arrays::Array& context1, const Int64Arrays::Int& context2, const Int64Arrays::IndexContainer& context3, Int64Arrays::Array& state1, Int64Arrays::Int& state2) {
            return __while_loop3_25.repeat(context1, context2, context3, state1, state2);
        };
    };

    static Int64Arrays::_drop_repeat drop_repeat;
    struct _drop_shape_elem {
        inline Int64Arrays::Shape operator()(const Int64Arrays::Array& a, const Int64Arrays::Int& i) {
            return __array.drop_shape_elem(a, i);
        };
    };

    static Int64Arrays::_drop_shape_elem drop_shape_elem;
    struct _get {
        inline Int64Arrays::Array operator()(const Int64Arrays::Array& a, const Int64Arrays::Int& ix) {
            return __array.get(a, ix);
        };
        inline Int64Arrays::Array operator()(const Int64Arrays::Array& a, const Int64Arrays::Index& ix) {
            return __array.get(a, ix);
        };
        inline Int64Arrays::Array operator()(const Int64Arrays::PaddedArray& a, const Int64Arrays::Index& ix) {
            return __array.get(a, ix);
        };
    };

    static Int64Arrays::_get get;
    struct _get_shape_elem {
        inline Int64Arrays::Int operator()(const Int64Arrays::Array& a, const Int64Arrays::Int& i) {
            return __array.get_shape_elem(a, i);
        };
    };

    static Int64Arrays::_get_shape_elem get_shape_elem;
    struct _leftmap_cond {
        inline bool operator()(const Int64Arrays::Int64& e, const Int64Arrays::IndexContainer& ix_space, const Int64Arrays::Array& a, const Int64Arrays::Int& c) {
            return Int64Arrays::lt(c, Int64Arrays::total(ix_space));
        };
    };

    static Int64Arrays::_leftmap_cond leftmap_cond;
    struct _lm_div_rep {
        inline void operator()(const Int64Arrays::Int64& context1, const Int64Arrays::IndexContainer& context2, Int64Arrays::Array& state1, Int64Arrays::Int& state2) {
            return __while_loop2_21.repeat(context1, context2, state1, state2);
        };
    };

    static Int64Arrays::_lm_div_rep lm_div_rep;
    struct _lm_mul_rep {
        inline void operator()(const Int64Arrays::Int64& context1, const Int64Arrays::IndexContainer& context2, Int64Arrays::Array& state1, Int64Arrays::Int& state2) {
            return __while_loop2_22.repeat(context1, context2, state1, state2);
        };
    };

    static Int64Arrays::_lm_mul_rep lm_mul_rep;
    struct _lm_plus_rep {
        inline void operator()(const Int64Arrays::Int64& context1, const Int64Arrays::IndexContainer& context2, Int64Arrays::Array& state1, Int64Arrays::Int& state2) {
            return __while_loop2_23.repeat(context1, context2, state1, state2);
        };
    };

    static Int64Arrays::_lm_plus_rep lm_plus_rep;
    struct _lm_sub_rep {
        inline void operator()(const Int64Arrays::Int64& context1, const Int64Arrays::IndexContainer& context2, Int64Arrays::Array& state1, Int64Arrays::Int& state2) {
            return __while_loop2_24.repeat(context1, context2, state1, state2);
        };
    };

    static Int64Arrays::_lm_sub_rep lm_sub_rep;
    struct _lmb_div {
        inline void operator()(const Int64Arrays::Int64& e, const Int64Arrays::IndexContainer& ix_space, Int64Arrays::Array& a, Int64Arrays::Int& c) {
            Int64Arrays::Index ix = Int64Arrays::get_index_ixc(ix_space, c);
            Int64Arrays::Int64 new_value = Int64Arrays::div(e, Int64Arrays::unwrap_scalar(Int64Arrays::get(a, ix)));
            Int64Arrays::set(a, ix, new_value);
            c = Int64Arrays::binary_add(c, Int64Arrays::one.operator()<Int>());
        };
    };

private:
    static while_loop2_2<Int64Arrays::Int64, Int64Arrays::IndexContainer, Int64Arrays::Array, Int64Arrays::Int, Int64Arrays::_lmb_div, Int64Arrays::_leftmap_cond> __while_loop2_21;
public:
    static Int64Arrays::_lmb_div lmb_div;
    struct _lmb_mul {
        inline void operator()(const Int64Arrays::Int64& e, const Int64Arrays::IndexContainer& ix_space, Int64Arrays::Array& a, Int64Arrays::Int& c) {
            Int64Arrays::Index ix = Int64Arrays::get_index_ixc(ix_space, c);
            Int64Arrays::Int64 new_value = Int64Arrays::mul(e, Int64Arrays::unwrap_scalar(Int64Arrays::get(a, ix)));
            Int64Arrays::set(a, ix, new_value);
            c = Int64Arrays::binary_add(c, Int64Arrays::one.operator()<Int>());
        };
    };

private:
    static while_loop2_2<Int64Arrays::Int64, Int64Arrays::IndexContainer, Int64Arrays::Array, Int64Arrays::Int, Int64Arrays::_lmb_mul, Int64Arrays::_leftmap_cond> __while_loop2_22;
public:
    static Int64Arrays::_lmb_mul lmb_mul;
    struct _lmb_plus {
        inline void operator()(const Int64Arrays::Int64& e, const Int64Arrays::IndexContainer& ix_space, Int64Arrays::Array& a, Int64Arrays::Int& c) {
            Int64Arrays::Index ix = Int64Arrays::get_index_ixc(ix_space, c);
            Int64Arrays::Int64 new_value = Int64Arrays::binary_add(e, Int64Arrays::unwrap_scalar(Int64Arrays::get(a, ix)));
            Int64Arrays::set(a, ix, new_value);
            c = Int64Arrays::binary_add(c, Int64Arrays::one.operator()<Int>());
        };
    };

private:
    static while_loop2_2<Int64Arrays::Int64, Int64Arrays::IndexContainer, Int64Arrays::Array, Int64Arrays::Int, Int64Arrays::_lmb_plus, Int64Arrays::_leftmap_cond> __while_loop2_23;
public:
    static Int64Arrays::_lmb_plus lmb_plus;
    struct _lmb_sub {
        inline void operator()(const Int64Arrays::Int64& e, const Int64Arrays::IndexContainer& ix_space, Int64Arrays::Array& a, Int64Arrays::Int& c) {
            Int64Arrays::Index ix = Int64Arrays::get_index_ixc(ix_space, c);
            Int64Arrays::Int64 new_value = Int64Arrays::binary_sub(e, Int64Arrays::unwrap_scalar(Int64Arrays::get(a, ix)));
            Int64Arrays::set(a, ix, new_value);
            c = Int64Arrays::binary_add(c, Int64Arrays::one.operator()<Int>());
        };
    };

private:
    static while_loop2_2<Int64Arrays::Int64, Int64Arrays::IndexContainer, Int64Arrays::Array, Int64Arrays::Int, Int64Arrays::_lmb_sub, Int64Arrays::_leftmap_cond> __while_loop2_24;
public:
    static Int64Arrays::_lmb_sub lmb_sub;
    struct _mapped_ops_cond {
        inline bool operator()(const Int64Arrays::Array& a, const Int64Arrays::Array& b, const Int64Arrays::IndexContainer& ix_space, const Int64Arrays::Array& res, const Int64Arrays::Int& c) {
            return Int64Arrays::lt(c, Int64Arrays::total(ix_space));
        };
    };

private:
    static while_loop3_2<Int64Arrays::Array, Int64Arrays::Array, Int64Arrays::IndexContainer, Int64Arrays::Array, Int64Arrays::Int, Int64Arrays::_bmb_div, Int64Arrays::_mapped_ops_cond> __while_loop3_2;
    static while_loop3_2<Int64Arrays::Array, Int64Arrays::Array, Int64Arrays::IndexContainer, Int64Arrays::Array, Int64Arrays::Int, Int64Arrays::_bmb_mul, Int64Arrays::_mapped_ops_cond> __while_loop3_20;
    static while_loop3_2<Int64Arrays::Array, Int64Arrays::Array, Int64Arrays::IndexContainer, Int64Arrays::Array, Int64Arrays::Int, Int64Arrays::_bmb_plus, Int64Arrays::_mapped_ops_cond> __while_loop3_21;
    static while_loop3_2<Int64Arrays::Array, Int64Arrays::Array, Int64Arrays::IndexContainer, Int64Arrays::Array, Int64Arrays::Int, Int64Arrays::_bmb_sub, Int64Arrays::_mapped_ops_cond> __while_loop3_22;
public:
    static Int64Arrays::_mapped_ops_cond mapped_ops_cond;
    struct _mul {
        inline Int64Arrays::Array operator()(const Int64Arrays::Array& a, const Int64Arrays::Array& b) {
            Int64Arrays::IndexContainer ix_space = Int64Arrays::create_total_indices(a);
            Int64Arrays::Array res = Int64Arrays::create_array(Int64Arrays::shape(a));
            Int64Arrays::Int counter = Int64Arrays::zero.operator()<Int>();
            Int64Arrays::bmb_mul_rep(a, b, ix_space, res, counter);
            return res;
        };
        inline Int64Arrays::Array operator()(const Int64Arrays::Int64& e, const Int64Arrays::Array& a) {
            Int64Arrays::IndexContainer ix_space = Int64Arrays::create_total_indices(a);
            Int64Arrays::Array upd_a = a;
            Int64Arrays::Int counter = Int64Arrays::zero.operator()<Int>();
            Int64Arrays::lm_mul_rep(e, ix_space, upd_a, counter);
            return upd_a;
        };
        inline Int64Arrays::Int64 operator()(const Int64Arrays::Int64& a, const Int64Arrays::Int64& b) {
            return __int64_utils.mul(a, b);
        };
        inline Int64Arrays::Int operator()(const Int64Arrays::Int& a, const Int64Arrays::Int& b) {
            return __array.mul(a, b);
        };
        inline Int64Arrays::Float operator()(const Int64Arrays::Float& a, const Int64Arrays::Float& b) {
            return __array.mul(a, b);
        };
    };

    static Int64Arrays::_mul mul;
    struct _padded_to_unpadded {
        inline Int64Arrays::Array operator()(const Int64Arrays::PaddedArray& a) {
            return __array.padded_to_unpadded(a);
        };
    };

    static Int64Arrays::_padded_to_unpadded padded_to_unpadded;
    struct _print_array {
        inline void operator()(const Int64Arrays::Array& a) {
            return __array.print_array(a);
        };
    };

    static Int64Arrays::_print_array print_array;
    struct _reshape {
        inline Int64Arrays::Array operator()(const Int64Arrays::Array& input_array, const Int64Arrays::Shape& s) {
            Int64Arrays::Array new_array = Int64Arrays::create_array(s);
            Int64Arrays::Int counter = Int64Arrays::zero.operator()<Int>();
            Int64Arrays::reshape_repeat(input_array, new_array, counter);
            return new_array;
        };
    };

    static Int64Arrays::_reshape reshape;
    struct _reshape_body {
        inline void operator()(const Int64Arrays::Array& old_array, Int64Arrays::Array& new_array, Int64Arrays::Int& counter) {
            Int64Arrays::set(new_array, counter, Int64Arrays::unwrap_scalar(Int64Arrays::get(old_array, counter)));
            counter = Int64Arrays::binary_add(counter, Int64Arrays::one.operator()<Int>());
        };
    };

    static Int64Arrays::_reshape_body reshape_body;
    struct _reshape_cond {
        inline bool operator()(const Int64Arrays::Array& old_array, const Int64Arrays::Array& new_array, const Int64Arrays::Int& counter) {
            return Int64Arrays::lt(counter, Int64Arrays::total(new_array));
        };
    };

private:
    static while_loop1_2<Int64Arrays::Array, Int64Arrays::Array, Int64Arrays::Int, Int64Arrays::_reshape_body, Int64Arrays::_reshape_cond> __while_loop1_2;
public:
    static Int64Arrays::_reshape_cond reshape_cond;
    struct _reshape_repeat {
        inline void operator()(const Int64Arrays::Array& context1, Int64Arrays::Array& state1, Int64Arrays::Int& state2) {
            return __while_loop1_2.repeat(context1, state1, state2);
        };
    };

    static Int64Arrays::_reshape_repeat reshape_repeat;
    struct _reverse {
        inline Int64Arrays::Array operator()(const Int64Arrays::Array& a) {
            Int64Arrays::Array res_array = Int64Arrays::create_array(Int64Arrays::shape(a));
            Int64Arrays::IndexContainer valid_indices = Int64Arrays::create_total_indices(res_array);
            Int64Arrays::Int counter = Int64Arrays::zero.operator()<Int>();
            Int64Arrays::reverse_repeat(a, valid_indices, res_array, counter);
            return res_array;
        };
    };

    static Int64Arrays::_reverse reverse;
    struct _reverse_body {
        inline void operator()(const Int64Arrays::Array& input, const Int64Arrays::IndexContainer& indices, Int64Arrays::Array& res, Int64Arrays::Int& c) {
            Int64Arrays::Index ix = Int64Arrays::get_index_ixc(indices, c);
            Int64Arrays::Int64 elem = Int64Arrays::unwrap_scalar(Int64Arrays::get(input, ix));
            Int64Arrays::Int sh_0 = Int64Arrays::get_shape_elem(input, Int64Arrays::zero.operator()<Int>());
            Int64Arrays::Int ix_0 = Int64Arrays::get_index_elem(ix, Int64Arrays::zero.operator()<Int>());
            Int64Arrays::Int new_ix_0 = Int64Arrays::binary_sub(sh_0, Int64Arrays::binary_add(ix_0, Int64Arrays::one.operator()<Int>()));
            Int64Arrays::Index new_ix = Int64Arrays::cat_index(Int64Arrays::create_index1(new_ix_0), Int64Arrays::drop_index_elem(ix, Int64Arrays::zero.operator()<Int>()));
            Int64Arrays::set(res, new_ix, elem);
            c = Int64Arrays::binary_add(c, Int64Arrays::one.operator()<Int>());
        };
    };

    static Int64Arrays::_reverse_body reverse_body;
    struct _reverse_cond {
        inline bool operator()(const Int64Arrays::Array& input, const Int64Arrays::IndexContainer& indices, const Int64Arrays::Array& res, const Int64Arrays::Int& c) {
            return Int64Arrays::lt(c, Int64Arrays::total(indices));
        };
    };

private:
    static while_loop2_2<Int64Arrays::Array, Int64Arrays::IndexContainer, Int64Arrays::Array, Int64Arrays::Int, Int64Arrays::_reverse_body, Int64Arrays::_reverse_cond> __while_loop2_2;
public:
    static Int64Arrays::_reverse_cond reverse_cond;
    struct _reverse_repeat {
        inline void operator()(const Int64Arrays::Array& context1, const Int64Arrays::IndexContainer& context2, Int64Arrays::Array& state1, Int64Arrays::Int& state2) {
            return __while_loop2_2.repeat(context1, context2, state1, state2);
        };
    };

    static Int64Arrays::_reverse_repeat reverse_repeat;
    struct _rotate {
        inline Int64Arrays::Array operator()(const Int64Arrays::Int& sigma, const Int64Arrays::Int& j, const Int64Arrays::Array& a) {
            Int64Arrays::IndexContainer ix_space = Int64Arrays::create_partial_indices(a, j);
            Int64Arrays::Array res = Int64Arrays::create_array(Int64Arrays::shape(a));
            Int64Arrays::Int c = Int64Arrays::zero.operator()<Int>();
            Int64Arrays::rotate_repeat(a, ix_space, sigma, res, c);
            return res;
        };
    };

    static Int64Arrays::_rotate rotate;
    struct _rotate_body {
        inline void operator()(const Int64Arrays::Array& a, const Int64Arrays::IndexContainer& ixc, const Int64Arrays::Int& sigma, Int64Arrays::Array& res, Int64Arrays::Int& c) {
            Int64Arrays::Index ix = Int64Arrays::get_index_ixc(ixc, c);
            if (Int64Arrays::le(Int64Arrays::zero.operator()<Int>(), sigma))
            {
                Int64Arrays::Array e1 = Int64Arrays::take(Int64Arrays::unary_sub(sigma), Int64Arrays::get(a, ix));
                Int64Arrays::Array e2 = Int64Arrays::drop(Int64Arrays::unary_sub(sigma), Int64Arrays::get(a, ix));
                Int64Arrays::set(res, ix, Int64Arrays::cat(e1, e2));
            }
            else
            {
                Int64Arrays::Array e1 = Int64Arrays::drop(sigma, Int64Arrays::get(a, ix));
                Int64Arrays::Array e2 = Int64Arrays::take(sigma, Int64Arrays::get(a, ix));
                Int64Arrays::set(res, ix, Int64Arrays::cat(e1, e2));
            }
            c = Int64Arrays::binary_add(c, Int64Arrays::one.operator()<Int>());
        };
    };

    static Int64Arrays::_rotate_body rotate_body;
    struct _rotate_cond {
        inline bool operator()(const Int64Arrays::Array& a, const Int64Arrays::IndexContainer& ixc, const Int64Arrays::Int& sigma, const Int64Arrays::Array& res, const Int64Arrays::Int& c) {
            return Int64Arrays::lt(c, Int64Arrays::total(ixc));
        };
    };

private:
    static while_loop3_2<Int64Arrays::Array, Int64Arrays::IndexContainer, Int64Arrays::Int, Int64Arrays::Array, Int64Arrays::Int, Int64Arrays::_rotate_body, Int64Arrays::_rotate_cond> __while_loop3_24;
public:
    static Int64Arrays::_rotate_cond rotate_cond;
    struct _rotate_repeat {
        inline void operator()(const Int64Arrays::Array& context1, const Int64Arrays::IndexContainer& context2, const Int64Arrays::Int& context3, Int64Arrays::Array& state1, Int64Arrays::Int& state2) {
            return __while_loop3_24.repeat(context1, context2, context3, state1, state2);
        };
    };

    static Int64Arrays::_rotate_repeat rotate_repeat;
    struct _scalarLeftMap {
        inline void operator()(const Int64Arrays::Int64& e, const Int64Arrays::Array& a, const Int64Arrays::Index& ix) {
            assert((Int64Arrays::unwrap_scalar(Int64Arrays::get(Int64Arrays::binary_add(e, a), ix))) == (Int64Arrays::binary_add(e, Int64Arrays::unwrap_scalar(Int64Arrays::get(a, ix)))));
            assert((Int64Arrays::unwrap_scalar(Int64Arrays::get(Int64Arrays::binary_sub(e, a), ix))) == (Int64Arrays::binary_sub(e, Int64Arrays::unwrap_scalar(Int64Arrays::get(a, ix)))));
            assert((Int64Arrays::unwrap_scalar(Int64Arrays::get(Int64Arrays::mul(e, a), ix))) == (Int64Arrays::mul(e, Int64Arrays::unwrap_scalar(Int64Arrays::get(a, ix)))));
            assert((Int64Arrays::unwrap_scalar(Int64Arrays::get(Int64Arrays::div(e, a), ix))) == (Int64Arrays::div(e, Int64Arrays::unwrap_scalar(Int64Arrays::get(a, ix)))));
        };
    };

    static Int64Arrays::_scalarLeftMap scalarLeftMap;
    struct _set {
        inline void operator()(Int64Arrays::Array& a, const Int64Arrays::Int& ix, const Int64Arrays::Int64& e) {
            return __array.set(a, ix, e);
        };
        inline void operator()(Int64Arrays::Array& a, const Int64Arrays::Index& ix, const Int64Arrays::Int64& e) {
            return __array.set(a, ix, e);
        };
        inline void operator()(Int64Arrays::Array& a, const Int64Arrays::Index& ix, const Int64Arrays::Array& e) {
            return __array.set(a, ix, e);
        };
        inline void operator()(Int64Arrays::PaddedArray& a, const Int64Arrays::Index& ix, const Int64Arrays::Int64& e) {
            return __array.set(a, ix, e);
        };
    };

    static Int64Arrays::_set set;
    struct _shape {
        inline Int64Arrays::Shape operator()(const Int64Arrays::Array& a) {
            return __array.shape(a);
        };
        inline Int64Arrays::Shape operator()(const Int64Arrays::PaddedArray& a) {
            return __array.shape(a);
        };
    };

    static Int64Arrays::_shape shape;
    struct _take {
        inline Int64Arrays::Array operator()(const Int64Arrays::Int& t, const Int64Arrays::Array& a) {
            Int64Arrays::Shape drop_sh_0 = Int64Arrays::drop_shape_elem(a, Int64Arrays::zero.operator()<Int>());
            Int64Arrays::Array res = Int64Arrays::create_array(Int64Arrays::cat_shape(Int64Arrays::create_shape1(Int64Arrays::abs(t)), drop_sh_0));
            Int64Arrays::IndexContainer ixc = Int64Arrays::create_partial_indices(res, Int64Arrays::one.operator()<Int>());
            Int64Arrays::Int c = Int64Arrays::zero.operator()<Int>();
            Int64Arrays::take_repeat(a, t, ixc, res, c);
            return res;
        };
    };

    static Int64Arrays::_take take;
    struct _take_body {
        inline void operator()(const Int64Arrays::Array& a, const Int64Arrays::Int& t, const Int64Arrays::IndexContainer& ixc, Int64Arrays::Array& res, Int64Arrays::Int& c) {
            Int64Arrays::Index ix = Int64Arrays::get_index_ixc(ixc, c);
            if (Int64Arrays::le(Int64Arrays::zero.operator()<Int>(), t))
            {
                Int64Arrays::set(res, ix, Int64Arrays::get(a, ix));
            }
            else
            {
                Int64Arrays::Int s0 = Int64Arrays::get_shape_elem(a, Int64Arrays::zero.operator()<Int>());
                Int64Arrays::Int i0 = Int64Arrays::get_index_elem(ix, Int64Arrays::zero.operator()<Int>());
                Int64Arrays::Index new_ix = Int64Arrays::create_index1(Int64Arrays::binary_add(Int64Arrays::binary_sub(s0, Int64Arrays::abs(t)), i0));
                Int64Arrays::set(res, ix, Int64Arrays::get(a, new_ix));
            }
            c = Int64Arrays::binary_add(c, Int64Arrays::one.operator()<Int>());
        };
    };

    static Int64Arrays::_take_body take_body;
    struct _take_cond {
        inline bool operator()(const Int64Arrays::Array& a, const Int64Arrays::Int& t, const Int64Arrays::IndexContainer& ixc, const Int64Arrays::Array& res, const Int64Arrays::Int& c) {
            return Int64Arrays::lt(c, Int64Arrays::abs(t));
        };
    };

private:
    static while_loop3_2<Int64Arrays::Array, Int64Arrays::Int, Int64Arrays::IndexContainer, Int64Arrays::Array, Int64Arrays::Int, Int64Arrays::_take_body, Int64Arrays::_take_cond> __while_loop3_26;
public:
    static Int64Arrays::_take_cond take_cond;
    struct _take_repeat {
        inline void operator()(const Int64Arrays::Array& context1, const Int64Arrays::Int& context2, const Int64Arrays::IndexContainer& context3, Int64Arrays::Array& state1, Int64Arrays::Int& state2) {
            return __while_loop3_26.repeat(context1, context2, context3, state1, state2);
        };
    };

    static Int64Arrays::_take_repeat take_repeat;
    struct _test_array3_2_2 {
        inline Int64Arrays::Array operator()() {
            return __array.test_array3_2_2();
        };
    };

    static Int64Arrays::_test_array3_2_2 test_array3_2_2;
    struct _test_array3_2_2F {
        inline Int64Arrays::Array operator()() {
            return __array.test_array3_2_2F();
        };
    };

    static Int64Arrays::_test_array3_2_2F test_array3_2_2F;
    struct _test_array3_3 {
        inline Int64Arrays::Array operator()() {
            return __array.test_array3_3();
        };
    };

    static Int64Arrays::_test_array3_3 test_array3_3;
    struct _test_vector2 {
        inline Int64Arrays::Array operator()() {
            return __array.test_vector2();
        };
    };

    static Int64Arrays::_test_vector2 test_vector2;
    struct _test_vector3 {
        inline Int64Arrays::Array operator()() {
            return __array.test_vector3();
        };
    };

    static Int64Arrays::_test_vector3 test_vector3;
    struct _test_vector5 {
        inline Int64Arrays::Array operator()() {
            return __array.test_vector5();
        };
    };

    static Int64Arrays::_test_vector5 test_vector5;
    struct _total {
        inline Int64Arrays::Int operator()(const Int64Arrays::Array& a) {
            return __array.total(a);
        };
        inline Int64Arrays::Int operator()(const Int64Arrays::Shape& s) {
            return __array.total(s);
        };
        inline Int64Arrays::Int operator()(const Int64Arrays::IndexContainer& ixc) {
            return __array.total(ixc);
        };
    };

    static Int64Arrays::_total total;
    struct _transpose {
        inline Int64Arrays::Array operator()(const Int64Arrays::Array& a) {
            Int64Arrays::Array transposed_array = Int64Arrays::create_array(Int64Arrays::reverse_shape(Int64Arrays::shape(a)));
            Int64Arrays::IndexContainer ix_space = Int64Arrays::create_total_indices(transposed_array);
            Int64Arrays::Int counter = Int64Arrays::zero.operator()<Int>();
            Int64Arrays::transpose_repeat(a, ix_space, transposed_array, counter);
            return transposed_array;
        };
        inline Int64Arrays::PaddedArray operator()(const Int64Arrays::PaddedArray& a) {
            Int64Arrays::Array reshaped_array = Int64Arrays::create_array(Int64Arrays::padded_shape(a));
            Int64Arrays::PaddedArray transposed_array = Int64Arrays::create_padded_array(Int64Arrays::reverse_shape(Int64Arrays::shape(a)), Int64Arrays::reverse_shape(Int64Arrays::padded_shape(a)), reshaped_array);
            Int64Arrays::IndexContainer ix_space = Int64Arrays::create_total_indices(transposed_array);
            Int64Arrays::Int counter = Int64Arrays::zero.operator()<Int>();
            Int64Arrays::padded_transpose_repeat(a, ix_space, transposed_array, counter);
            return transposed_array;
        };
    };

    static Int64Arrays::_transpose transpose;
    struct _transpose_body {
        inline void operator()(const Int64Arrays::Array& a, const Int64Arrays::IndexContainer& ixc, Int64Arrays::Array& res, Int64Arrays::Int& c) {
            Int64Arrays::Index current_ix = Int64Arrays::get_index_ixc(ixc, c);
            Int64Arrays::Int64 current_element = Int64Arrays::unwrap_scalar(Int64Arrays::get(a, Int64Arrays::reverse_index(current_ix)));
            Int64Arrays::set(res, current_ix, current_element);
            c = Int64Arrays::binary_add(c, Int64Arrays::one.operator()<Int>());
        };
    };

    static Int64Arrays::_transpose_body transpose_body;
    struct _transpose_repeat {
        inline void operator()(const Int64Arrays::Array& context1, const Int64Arrays::IndexContainer& context2, Int64Arrays::Array& state1, Int64Arrays::Int& state2) {
            return __while_loop2_20.repeat(context1, context2, state1, state2);
        };
    };

    static Int64Arrays::_transpose_repeat transpose_repeat;
    struct _unaryMap {
        inline void operator()(const Int64Arrays::Array& a, const Int64Arrays::Index& ix) {
            assert((Int64Arrays::unwrap_scalar(Int64Arrays::get(Int64Arrays::unary_sub(a), ix))) == (Int64Arrays::unary_sub(Int64Arrays::unwrap_scalar(Int64Arrays::get(a, ix)))));
        };
    };

    static Int64Arrays::_unaryMap unaryMap;
    struct _unary_sub {
        inline Int64Arrays::Array operator()(const Int64Arrays::Array& a) {
            Int64Arrays::IndexContainer ix_space = Int64Arrays::create_total_indices(a);
            Int64Arrays::Array a_upd = a;
            Int64Arrays::Int counter = Int64Arrays::zero.operator()<Int>();
            Int64Arrays::unary_sub_repeat(ix_space, a_upd, counter);
            return a_upd;
        };
        inline Int64Arrays::Int64 operator()(const Int64Arrays::Int64& a) {
            return __int64_utils.unary_sub(a);
        };
        inline Int64Arrays::Int operator()(const Int64Arrays::Int& a) {
            return __array.unary_sub(a);
        };
        inline Int64Arrays::Float operator()(const Int64Arrays::Float& a) {
            return __array.unary_sub(a);
        };
    };

    static Int64Arrays::_unary_sub unary_sub;
    struct _unary_sub_body {
        inline void operator()(const Int64Arrays::IndexContainer& ix_space, Int64Arrays::Array& a, Int64Arrays::Int& c) {
            Int64Arrays::Index ix = Int64Arrays::get_index_ixc(ix_space, c);
            Int64Arrays::Int64 new_value = Int64Arrays::unary_sub(Int64Arrays::unwrap_scalar(Int64Arrays::get(a, ix)));
            Int64Arrays::set(a, ix, new_value);
            c = Int64Arrays::binary_add(c, Int64Arrays::one.operator()<Int>());
        };
    };

    static Int64Arrays::_unary_sub_body unary_sub_body;
    struct _unary_sub_cond {
        inline bool operator()(const Int64Arrays::IndexContainer& ix_space, const Int64Arrays::Array& a, const Int64Arrays::Int& c) {
            return Int64Arrays::lt(c, Int64Arrays::total(ix_space));
        };
    };

private:
    static while_loop1_2<Int64Arrays::IndexContainer, Int64Arrays::Array, Int64Arrays::Int, Int64Arrays::_unary_sub_body, Int64Arrays::_unary_sub_cond> __while_loop1_20;
public:
    static Int64Arrays::_unary_sub_cond unary_sub_cond;
    struct _unary_sub_repeat {
        inline void operator()(const Int64Arrays::IndexContainer& context1, Int64Arrays::Array& state1, Int64Arrays::Int& state2) {
            return __while_loop1_20.repeat(context1, state1, state2);
        };
    };

    static Int64Arrays::_unary_sub_repeat unary_sub_repeat;
    struct _unwrap_scalar {
        inline Int64Arrays::Int64 operator()(const Int64Arrays::Array& a) {
            return __array.unwrap_scalar(a);
        };
    };

    static Int64Arrays::_unwrap_scalar unwrap_scalar;
    struct _upper_bound {
        inline bool operator()(const Int64Arrays::Array& a, const Int64Arrays::IndexContainer& i, const Int64Arrays::Array& res, const Int64Arrays::Int& c) {
            return Int64Arrays::lt(c, Int64Arrays::total(i));
        };
    };

private:
    static while_loop2_2<Int64Arrays::Array, Int64Arrays::IndexContainer, Int64Arrays::Array, Int64Arrays::Int, Int64Arrays::_transpose_body, Int64Arrays::_upper_bound> __while_loop2_20;
public:
    static Int64Arrays::_upper_bound upper_bound;
};
} // examples
} // moa
} // mg_src
} // moa_cpp