#pragma once

#include "base.hpp"
#include <cassert>


namespace examples {
namespace pde {
namespace mg_src {
namespace pde_cpp {
struct BasePDEProgram {
    struct _two {
        template <typename T>
        inline T operator()() {
            T o;
            BasePDEProgram::two0(o);
            return o;
        };
    };

    static BasePDEProgram::_two two;
    struct _one {
        template <typename T>
        inline T operator()() {
            T o;
            BasePDEProgram::one0(o);
            return o;
        };
    };

    static BasePDEProgram::_one one;
private:
    static array_ops __array_ops;
public:
    typedef array_ops::Offset Offset;
private:
    static inline void one0(BasePDEProgram::Offset& o) {
        o = __array_ops.one_offset();
    };
public:
    typedef array_ops::Nat Nat;
    typedef array_ops::Index Index;
    typedef array_ops::Float Float;
    struct _three {
        inline BasePDEProgram::Float operator()() {
            return __array_ops.three_float();
        };
    };

    static BasePDEProgram::_three three;
    struct _unary_sub {
        inline BasePDEProgram::Float operator()(const BasePDEProgram::Float& f) {
            return __array_ops.unary_sub(f);
        };
        inline BasePDEProgram::Offset operator()(const BasePDEProgram::Offset& o) {
            return __array_ops.unary_sub(o);
        };
    };

    static BasePDEProgram::_unary_sub unary_sub;
private:
    static inline void one0(BasePDEProgram::Float& o) {
        o = __array_ops.one_float();
    };
    static inline void two0(BasePDEProgram::Float& o) {
        o = __array_ops.two_float();
    };
public:
    typedef array_ops::Axis Axis;
    struct _rotate_ix {
        inline BasePDEProgram::Index operator()(const BasePDEProgram::Index& ix, const BasePDEProgram::Axis& axis, const BasePDEProgram::Offset& o) {
            return __array_ops.rotate_ix(ix, axis, o);
        };
    };

    static BasePDEProgram::_rotate_ix rotate_ix;
    struct _zero {
        inline BasePDEProgram::Axis operator()() {
            return __array_ops.zero_axis();
        };
    };

    static BasePDEProgram::_zero zero;
private:
    static inline void one0(BasePDEProgram::Axis& o) {
        o = __array_ops.one_axis();
    };
    static inline void two0(BasePDEProgram::Axis& o) {
        o = __array_ops.two_axis();
    };
public:
    typedef array_ops::Array Array;
    struct _all_substeps {
        inline void operator()(BasePDEProgram::Array& u0, BasePDEProgram::Array& u1, BasePDEProgram::Array& u2, const BasePDEProgram::Float& c0, const BasePDEProgram::Float& c1, const BasePDEProgram::Float& c2, const BasePDEProgram::Float& c3, const BasePDEProgram::Float& c4) {
            BasePDEProgram::Array v0 = u0;
            BasePDEProgram::Array v1 = u1;
            BasePDEProgram::Array v2 = u2;
            v0 = BasePDEProgram::forall_ix_snippet(v0, u0, u0, u1, u2, c0, c1, c2, c3, c4);
            v1 = BasePDEProgram::forall_ix_snippet(v1, u1, u0, u1, u2, c0, c1, c2, c3, c4);
            v2 = BasePDEProgram::forall_ix_snippet(v2, u2, u0, u1, u2, c0, c1, c2, c3, c4);
            u0 = BasePDEProgram::forall_ix_snippet(u0, v0, u0, u1, u2, c0, c1, c2, c3, c4);
            u1 = BasePDEProgram::forall_ix_snippet(u1, v1, u0, u1, u2, c0, c1, c2, c3, c4);
            u2 = BasePDEProgram::forall_ix_snippet(u2, v2, u0, u1, u2, c0, c1, c2, c3, c4);
        };
    };

    static BasePDEProgram::_all_substeps all_substeps;
    struct _binary_add {
        inline BasePDEProgram::Float operator()(const BasePDEProgram::Float& lhs, const BasePDEProgram::Float& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        inline BasePDEProgram::Array operator()(const BasePDEProgram::Float& lhs, const BasePDEProgram::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        inline BasePDEProgram::Array operator()(const BasePDEProgram::Array& lhs, const BasePDEProgram::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
    };

    static BasePDEProgram::_binary_add binary_add;
    struct _binary_sub {
        inline BasePDEProgram::Float operator()(const BasePDEProgram::Float& lhs, const BasePDEProgram::Float& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        inline BasePDEProgram::Array operator()(const BasePDEProgram::Float& lhs, const BasePDEProgram::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        inline BasePDEProgram::Array operator()(const BasePDEProgram::Array& lhs, const BasePDEProgram::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
    };

    static BasePDEProgram::_binary_sub binary_sub;
    struct _div {
        inline BasePDEProgram::Float operator()(const BasePDEProgram::Float& num, const BasePDEProgram::Float& den) {
            return __array_ops.div(num, den);
        };
        inline BasePDEProgram::Array operator()(const BasePDEProgram::Float& num, const BasePDEProgram::Array& den) {
            return __array_ops.div(num, den);
        };
    };

    static BasePDEProgram::_div div;
    struct _forall_ix_snippet {
        inline BasePDEProgram::Array operator()(const BasePDEProgram::Array& u, const BasePDEProgram::Array& v, const BasePDEProgram::Array& u0, const BasePDEProgram::Array& u1, const BasePDEProgram::Array& u2, const BasePDEProgram::Float& c0, const BasePDEProgram::Float& c1, const BasePDEProgram::Float& c2, const BasePDEProgram::Float& c3, const BasePDEProgram::Float& c4) {
            return __forall_ops.forall_ix_snippet(u, v, u0, u1, u2, c0, c1, c2, c3, c4);
        };
    };

    static BasePDEProgram::_forall_ix_snippet forall_ix_snippet;
    struct _mul {
        inline BasePDEProgram::Float operator()(const BasePDEProgram::Float& lhs, const BasePDEProgram::Float& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        inline BasePDEProgram::Array operator()(const BasePDEProgram::Float& lhs, const BasePDEProgram::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        inline BasePDEProgram::Array operator()(const BasePDEProgram::Array& lhs, const BasePDEProgram::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
    };

    static BasePDEProgram::_mul mul;
    struct _psi {
        inline BasePDEProgram::Float operator()(const BasePDEProgram::Index& ix, const BasePDEProgram::Array& array) {
            return __array_ops.psi(ix, array);
        };
    };

    static BasePDEProgram::_psi psi;
    struct _rotate {
        inline BasePDEProgram::Array operator()(const BasePDEProgram::Array& a, const BasePDEProgram::Axis& axis, const BasePDEProgram::Offset& o) {
            return __array_ops.rotate(a, axis, o);
        };
    };

    static BasePDEProgram::_rotate rotate;
    struct _snippet_ix {
        inline BasePDEProgram::Float operator()(const BasePDEProgram::Array& u, const BasePDEProgram::Array& v, const BasePDEProgram::Array& u0, const BasePDEProgram::Array& u1, const BasePDEProgram::Array& u2, const BasePDEProgram::Float& c0, const BasePDEProgram::Float& c1, const BasePDEProgram::Float& c2, const BasePDEProgram::Float& c3, const BasePDEProgram::Float& c4, const BasePDEProgram::Index& ix) {
            BasePDEProgram::Axis zero = BasePDEProgram::zero();
            BasePDEProgram::Offset one = BasePDEProgram::one.operator()<Offset>();
            BasePDEProgram::Axis two = BasePDEProgram::two.operator()<Axis>();
            return BasePDEProgram::psi(ix, BasePDEProgram::binary_add(u, BasePDEProgram::mul(c4, BasePDEProgram::binary_sub(BasePDEProgram::mul(c3, BasePDEProgram::binary_sub(BasePDEProgram::mul(c1, BasePDEProgram::binary_add(BasePDEProgram::binary_add(BasePDEProgram::binary_add(BasePDEProgram::binary_add(BasePDEProgram::binary_add(BasePDEProgram::rotate(v, zero, BasePDEProgram::unary_sub(one)), BasePDEProgram::rotate(v, zero, one)), BasePDEProgram::rotate(v, BasePDEProgram::one.operator()<Axis>(), BasePDEProgram::unary_sub(one))), BasePDEProgram::rotate(v, BasePDEProgram::one.operator()<Axis>(), one)), BasePDEProgram::rotate(v, two, BasePDEProgram::unary_sub(one))), BasePDEProgram::rotate(v, two, one))), BasePDEProgram::mul(BasePDEProgram::mul(BasePDEProgram::three(), c2), u0))), BasePDEProgram::mul(c0, BasePDEProgram::binary_add(BasePDEProgram::binary_add(BasePDEProgram::mul(BasePDEProgram::binary_sub(BasePDEProgram::rotate(v, zero, one), BasePDEProgram::rotate(v, zero, BasePDEProgram::unary_sub(one))), u0), BasePDEProgram::mul(BasePDEProgram::binary_sub(BasePDEProgram::rotate(v, BasePDEProgram::one.operator()<Axis>(), one), BasePDEProgram::rotate(v, BasePDEProgram::one.operator()<Axis>(), BasePDEProgram::unary_sub(one))), u1)), BasePDEProgram::mul(BasePDEProgram::binary_sub(BasePDEProgram::rotate(v, two, one), BasePDEProgram::rotate(v, two, BasePDEProgram::unary_sub(one))), u2)))))));
        };
    };

    typedef forall_ops<BasePDEProgram::Array, BasePDEProgram::Axis, BasePDEProgram::Float, BasePDEProgram::Index, BasePDEProgram::Nat, BasePDEProgram::Offset, BasePDEProgram::_snippet_ix>::ScalarIndex ScalarIndex;
    struct _make_ix {
        inline BasePDEProgram::Index operator()(const BasePDEProgram::ScalarIndex& a, const BasePDEProgram::ScalarIndex& b, const BasePDEProgram::ScalarIndex& c) {
            return __forall_ops.make_ix(a, b, c);
        };
    };

    static BasePDEProgram::_make_ix make_ix;
private:
    static forall_ops<BasePDEProgram::Array, BasePDEProgram::Axis, BasePDEProgram::Float, BasePDEProgram::Index, BasePDEProgram::Nat, BasePDEProgram::Offset, BasePDEProgram::_snippet_ix> __forall_ops;
public:
    static BasePDEProgram::_snippet_ix snippet_ix;
    struct _step {
        inline void operator()(BasePDEProgram::Array& u0, BasePDEProgram::Array& u1, BasePDEProgram::Array& u2, const BasePDEProgram::Float& nu, const BasePDEProgram::Float& dx, const BasePDEProgram::Float& dt) {
            BasePDEProgram::Float one = BasePDEProgram::one.operator()<Float>();
            BasePDEProgram::Float _2 = BasePDEProgram::two.operator()<Float>();
            BasePDEProgram::Float c0 = BasePDEProgram::div(BasePDEProgram::div(one, _2), dx);
            BasePDEProgram::Float c1 = BasePDEProgram::div(BasePDEProgram::div(one, dx), dx);
            BasePDEProgram::Float c2 = BasePDEProgram::div(BasePDEProgram::div(_2, dx), dx);
            BasePDEProgram::Float c3 = nu;
            BasePDEProgram::Float c4 = BasePDEProgram::div(dt, _2);
            BasePDEProgram::all_substeps(u0, u1, u2, c0, c1, c2, c3, c4);
        };
    };

    static BasePDEProgram::_step step;
};
} // examples
} // pde
} // mg_src
} // pde_cpp

namespace examples {
namespace pde {
namespace mg_src {
namespace pde_cpp {
struct PDEProgram {
    struct _two {
        template <typename T>
        inline T operator()() {
            T o;
            PDEProgram::two0(o);
            return o;
        };
    };

    static PDEProgram::_two two;
    struct _one {
        template <typename T>
        inline T operator()() {
            T o;
            PDEProgram::one0(o);
            return o;
        };
    };

    static PDEProgram::_one one;
private:
    static array_ops __array_ops;
public:
    typedef array_ops::Offset Offset;
private:
    static inline void one0(PDEProgram::Offset& o) {
        o = __array_ops.one_offset();
    };
public:
    typedef array_ops::Nat Nat;
    typedef array_ops::Index Index;
    typedef array_ops::Float Float;
    struct _three {
        inline PDEProgram::Float operator()() {
            return __array_ops.three_float();
        };
    };

    static PDEProgram::_three three;
    struct _unary_sub {
        inline PDEProgram::Offset operator()(const PDEProgram::Offset& o) {
            return __array_ops.unary_sub(o);
        };
        inline PDEProgram::Float operator()(const PDEProgram::Float& f) {
            return __array_ops.unary_sub(f);
        };
    };

    static PDEProgram::_unary_sub unary_sub;
private:
    static inline void one0(PDEProgram::Float& o) {
        o = __array_ops.one_float();
    };
    static inline void two0(PDEProgram::Float& o) {
        o = __array_ops.two_float();
    };
public:
    typedef array_ops::Axis Axis;
    struct _rotate_ix {
        inline PDEProgram::Index operator()(const PDEProgram::Index& ix, const PDEProgram::Axis& axis, const PDEProgram::Offset& o) {
            return __array_ops.rotate_ix(ix, axis, o);
        };
    };

    static PDEProgram::_rotate_ix rotate_ix;
    struct _rotate_ix_padded {
        inline PDEProgram::Index operator()(const PDEProgram::Index& ix, const PDEProgram::Axis& axis, const PDEProgram::Offset& offset) {
            return __forall_ops.rotate_ix_padded(ix, axis, offset);
        };
    };

    static PDEProgram::_rotate_ix_padded rotate_ix_padded;
    struct _zero {
        inline PDEProgram::Axis operator()() {
            return __array_ops.zero_axis();
        };
    };

    static PDEProgram::_zero zero;
private:
    static inline void one0(PDEProgram::Axis& o) {
        o = __array_ops.one_axis();
    };
    static inline void two0(PDEProgram::Axis& o) {
        o = __array_ops.two_axis();
    };
public:
    typedef array_ops::Array Array;
    struct _all_substeps {
        inline void operator()(PDEProgram::Array& u0, PDEProgram::Array& u1, PDEProgram::Array& u2, const PDEProgram::Float& c0, const PDEProgram::Float& c1, const PDEProgram::Float& c2, const PDEProgram::Float& c3, const PDEProgram::Float& c4) {
            PDEProgram::Array v0 = u0;
            PDEProgram::Array v1 = u1;
            PDEProgram::Array v2 = u2;
            v0 = [&]() {
                PDEProgram::Array result = PDEProgram::forall_ix_snippet_specialized_psi_padded(v0, u0, u0, u1, u2, c0, c1, c2, c3, c4);
                PDEProgram::refill_all_padding(result);
                return result;
            }();
            v1 = [&]() {
                PDEProgram::Array result = PDEProgram::forall_ix_snippet_specialized_psi_padded(v1, u1, u0, u1, u2, c0, c1, c2, c3, c4);
                PDEProgram::refill_all_padding(result);
                return result;
            }();
            v2 = [&]() {
                PDEProgram::Array result = PDEProgram::forall_ix_snippet_specialized_psi_padded(v2, u2, u0, u1, u2, c0, c1, c2, c3, c4);
                PDEProgram::refill_all_padding(result);
                return result;
            }();
            u0 = [&]() {
                PDEProgram::Array result = PDEProgram::forall_ix_snippet_specialized_psi_padded(u0, v0, u0, u1, u2, c0, c1, c2, c3, c4);
                PDEProgram::refill_all_padding(result);
                return result;
            }();
            u1 = [&]() {
                PDEProgram::Array result = PDEProgram::forall_ix_snippet_specialized_psi_padded(u1, v1, u0, u1, u2, c0, c1, c2, c3, c4);
                PDEProgram::refill_all_padding(result);
                return result;
            }();
            u2 = [&]() {
                PDEProgram::Array result = PDEProgram::forall_ix_snippet_specialized_psi_padded(u2, v2, u0, u1, u2, c0, c1, c2, c3, c4);
                PDEProgram::refill_all_padding(result);
                return result;
            }();
        };
    };

    static PDEProgram::_all_substeps all_substeps;
    struct _binary_sub {
        inline PDEProgram::Float operator()(const PDEProgram::Float& lhs, const PDEProgram::Float& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        inline PDEProgram::Array operator()(const PDEProgram::Float& lhs, const PDEProgram::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        inline PDEProgram::Array operator()(const PDEProgram::Array& lhs, const PDEProgram::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
    };

    static PDEProgram::_binary_sub binary_sub;
    struct _div {
        inline PDEProgram::Array operator()(const PDEProgram::Float& num, const PDEProgram::Array& den) {
            return __array_ops.div(num, den);
        };
        inline PDEProgram::Float operator()(const PDEProgram::Float& num, const PDEProgram::Float& den) {
            return __array_ops.div(num, den);
        };
    };

    static PDEProgram::_div div;
    struct _forall_ix_snippet {
        inline PDEProgram::Array operator()(const PDEProgram::Array& u, const PDEProgram::Array& v, const PDEProgram::Array& u0, const PDEProgram::Array& u1, const PDEProgram::Array& u2, const PDEProgram::Float& c0, const PDEProgram::Float& c1, const PDEProgram::Float& c2, const PDEProgram::Float& c3, const PDEProgram::Float& c4) {
            return __forall_ops.forall_ix_snippet(u, v, u0, u1, u2, c0, c1, c2, c3, c4);
        };
    };

    static PDEProgram::_forall_ix_snippet forall_ix_snippet;
    struct _forall_ix_snippet_padded {
        inline PDEProgram::Array operator()(const PDEProgram::Array& u, const PDEProgram::Array& v, const PDEProgram::Array& u0, const PDEProgram::Array& u1, const PDEProgram::Array& u2, const PDEProgram::Float& c0, const PDEProgram::Float& c1, const PDEProgram::Float& c2, const PDEProgram::Float& c3, const PDEProgram::Float& c4) {
            return __forall_ops.forall_ix_snippet_padded(u, v, u0, u1, u2, c0, c1, c2, c3, c4);
        };
    };

    static PDEProgram::_forall_ix_snippet_padded forall_ix_snippet_padded;
    struct _forall_ix_snippet_specialized_psi_padded {
        inline PDEProgram::Array operator()(const PDEProgram::Array& u, const PDEProgram::Array& v, const PDEProgram::Array& u0, const PDEProgram::Array& u1, const PDEProgram::Array& u2, const PDEProgram::Float& c0, const PDEProgram::Float& c1, const PDEProgram::Float& c2, const PDEProgram::Float& c3, const PDEProgram::Float& c4) {
            return __specialize_psi_ops_2.forall_ix_snippet_specialized_psi_padded(u, v, u0, u1, u2, c0, c1, c2, c3, c4);
        };
    };

    static PDEProgram::_forall_ix_snippet_specialized_psi_padded forall_ix_snippet_specialized_psi_padded;
    struct _mul {
        inline PDEProgram::Float operator()(const PDEProgram::Float& lhs, const PDEProgram::Float& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        inline PDEProgram::Array operator()(const PDEProgram::Float& lhs, const PDEProgram::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        inline PDEProgram::Array operator()(const PDEProgram::Array& lhs, const PDEProgram::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
    };

    static PDEProgram::_mul mul;
    struct _refill_all_padding {
        inline void operator()(PDEProgram::Array& a) {
            return __forall_ops.refill_all_padding(a);
        };
    };

    static PDEProgram::_refill_all_padding refill_all_padding;
    struct _rotate {
        inline PDEProgram::Array operator()(const PDEProgram::Array& a, const PDEProgram::Axis& axis, const PDEProgram::Offset& o) {
            return __array_ops.rotate(a, axis, o);
        };
    };

    static PDEProgram::_rotate rotate;
    struct _snippet_ix {
        inline PDEProgram::Float operator()(const PDEProgram::Array& u, const PDEProgram::Array& v, const PDEProgram::Array& u0, const PDEProgram::Array& u1, const PDEProgram::Array& u2, const PDEProgram::Float& c0, const PDEProgram::Float& c1, const PDEProgram::Float& c2, const PDEProgram::Float& c3, const PDEProgram::Float& c4, const PDEProgram::Index& ix) {
            PDEProgram::Axis zero = PDEProgram::zero();
            PDEProgram::Offset one = PDEProgram::one.operator()<Offset>();
            PDEProgram::Axis two = PDEProgram::two.operator()<Axis>();
            return PDEProgram::binary_add(PDEProgram::psi(PDEProgram::ix_0(ix), PDEProgram::ix_1(ix), PDEProgram::ix_2(ix), u), PDEProgram::mul(c4, PDEProgram::binary_sub(PDEProgram::mul(c3, PDEProgram::binary_sub(PDEProgram::mul(c1, PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::psi(PDEProgram::ix_0(PDEProgram::rotate_ix_padded(ix, zero, PDEProgram::unary_sub(one))), PDEProgram::ix_1(PDEProgram::rotate_ix_padded(ix, zero, PDEProgram::unary_sub(one))), PDEProgram::ix_2(PDEProgram::rotate_ix_padded(ix, zero, PDEProgram::unary_sub(one))), v), PDEProgram::psi(PDEProgram::ix_0(PDEProgram::rotate_ix_padded(ix, zero, one)), PDEProgram::ix_1(PDEProgram::rotate_ix_padded(ix, zero, one)), PDEProgram::ix_2(PDEProgram::rotate_ix_padded(ix, zero, one)), v)), PDEProgram::psi(PDEProgram::ix_0(PDEProgram::rotate_ix_padded(ix, PDEProgram::one.operator()<Axis>(), PDEProgram::unary_sub(one))), PDEProgram::ix_1(PDEProgram::rotate_ix_padded(ix, PDEProgram::one.operator()<Axis>(), PDEProgram::unary_sub(one))), PDEProgram::ix_2(PDEProgram::rotate_ix_padded(ix, PDEProgram::one.operator()<Axis>(), PDEProgram::unary_sub(one))), v)), PDEProgram::psi(PDEProgram::ix_0(PDEProgram::rotate_ix_padded(ix, PDEProgram::one.operator()<Axis>(), one)), PDEProgram::ix_1(PDEProgram::rotate_ix_padded(ix, PDEProgram::one.operator()<Axis>(), one)), PDEProgram::ix_2(PDEProgram::rotate_ix_padded(ix, PDEProgram::one.operator()<Axis>(), one)), v)), PDEProgram::psi(PDEProgram::ix_0(PDEProgram::rotate_ix_padded(ix, two, PDEProgram::unary_sub(one))), PDEProgram::ix_1(PDEProgram::rotate_ix_padded(ix, two, PDEProgram::unary_sub(one))), PDEProgram::ix_2(PDEProgram::rotate_ix_padded(ix, two, PDEProgram::unary_sub(one))), v)), PDEProgram::psi(PDEProgram::ix_0(PDEProgram::rotate_ix_padded(ix, two, one)), PDEProgram::ix_1(PDEProgram::rotate_ix_padded(ix, two, one)), PDEProgram::ix_2(PDEProgram::rotate_ix_padded(ix, two, one)), v))), PDEProgram::mul(PDEProgram::mul(PDEProgram::three(), c2), PDEProgram::psi(PDEProgram::ix_0(ix), PDEProgram::ix_1(ix), PDEProgram::ix_2(ix), u0)))), PDEProgram::mul(c0, PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::psi(PDEProgram::ix_0(PDEProgram::rotate_ix_padded(ix, zero, one)), PDEProgram::ix_1(PDEProgram::rotate_ix_padded(ix, zero, one)), PDEProgram::ix_2(PDEProgram::rotate_ix_padded(ix, zero, one)), v), PDEProgram::psi(PDEProgram::ix_0(PDEProgram::rotate_ix_padded(ix, zero, PDEProgram::unary_sub(one))), PDEProgram::ix_1(PDEProgram::rotate_ix_padded(ix, zero, PDEProgram::unary_sub(one))), PDEProgram::ix_2(PDEProgram::rotate_ix_padded(ix, zero, PDEProgram::unary_sub(one))), v)), PDEProgram::psi(PDEProgram::ix_0(ix), PDEProgram::ix_1(ix), PDEProgram::ix_2(ix), u0)), PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::psi(PDEProgram::ix_0(PDEProgram::rotate_ix_padded(ix, PDEProgram::one.operator()<Axis>(), one)), PDEProgram::ix_1(PDEProgram::rotate_ix_padded(ix, PDEProgram::one.operator()<Axis>(), one)), PDEProgram::ix_2(PDEProgram::rotate_ix_padded(ix, PDEProgram::one.operator()<Axis>(), one)), v), PDEProgram::psi(PDEProgram::ix_0(PDEProgram::rotate_ix_padded(ix, PDEProgram::one.operator()<Axis>(), PDEProgram::unary_sub(one))), PDEProgram::ix_1(PDEProgram::rotate_ix_padded(ix, PDEProgram::one.operator()<Axis>(), PDEProgram::unary_sub(one))), PDEProgram::ix_2(PDEProgram::rotate_ix_padded(ix, PDEProgram::one.operator()<Axis>(), PDEProgram::unary_sub(one))), v)), PDEProgram::psi(PDEProgram::ix_0(ix), PDEProgram::ix_1(ix), PDEProgram::ix_2(ix), u1))), PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::psi(PDEProgram::ix_0(PDEProgram::rotate_ix_padded(ix, two, one)), PDEProgram::ix_1(PDEProgram::rotate_ix_padded(ix, two, one)), PDEProgram::ix_2(PDEProgram::rotate_ix_padded(ix, two, one)), v), PDEProgram::psi(PDEProgram::ix_0(PDEProgram::rotate_ix_padded(ix, two, PDEProgram::unary_sub(one))), PDEProgram::ix_1(PDEProgram::rotate_ix_padded(ix, two, PDEProgram::unary_sub(one))), PDEProgram::ix_2(PDEProgram::rotate_ix_padded(ix, two, PDEProgram::unary_sub(one))), v)), PDEProgram::psi(PDEProgram::ix_0(ix), PDEProgram::ix_1(ix), PDEProgram::ix_2(ix), u2)))))));
        };
    };

    typedef forall_ops<PDEProgram::Array, PDEProgram::Axis, PDEProgram::Float, PDEProgram::Index, PDEProgram::Nat, PDEProgram::Offset, PDEProgram::_snippet_ix>::ScalarIndex ScalarIndex;
    struct _binary_add {
        inline PDEProgram::ScalarIndex operator()(const PDEProgram::ScalarIndex& six, const PDEProgram::Offset& o) {
            return __specialize_psi_ops_2.binary_add(six, o);
        };
        inline PDEProgram::Float operator()(const PDEProgram::Float& lhs, const PDEProgram::Float& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        inline PDEProgram::Array operator()(const PDEProgram::Float& lhs, const PDEProgram::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        inline PDEProgram::Array operator()(const PDEProgram::Array& lhs, const PDEProgram::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
    };

    static PDEProgram::_binary_add binary_add;
    struct _ix_0 {
        inline PDEProgram::ScalarIndex operator()(const PDEProgram::Index& ix) {
            return __specialize_psi_ops_2.ix_0(ix);
        };
    };

    static PDEProgram::_ix_0 ix_0;
    struct _ix_1 {
        inline PDEProgram::ScalarIndex operator()(const PDEProgram::Index& ix) {
            return __specialize_psi_ops_2.ix_1(ix);
        };
    };

    static PDEProgram::_ix_1 ix_1;
    struct _ix_2 {
        inline PDEProgram::ScalarIndex operator()(const PDEProgram::Index& ix) {
            return __specialize_psi_ops_2.ix_2(ix);
        };
    };

    static PDEProgram::_ix_2 ix_2;
    struct _make_ix {
        inline PDEProgram::Index operator()(const PDEProgram::ScalarIndex& a, const PDEProgram::ScalarIndex& b, const PDEProgram::ScalarIndex& c) {
            return __forall_ops.make_ix(a, b, c);
        };
    };

    static PDEProgram::_make_ix make_ix;
    struct _psi {
        inline PDEProgram::Float operator()(const PDEProgram::ScalarIndex& i, const PDEProgram::ScalarIndex& j, const PDEProgram::ScalarIndex& k, const PDEProgram::Array& a) {
            return __specialize_psi_ops_2.psi(i, j, k, a);
        };
        inline PDEProgram::Float operator()(const PDEProgram::Index& ix, const PDEProgram::Array& array) {
            return __array_ops.psi(ix, array);
        };
    };

    static PDEProgram::_psi psi;
private:
    static forall_ops<PDEProgram::Array, PDEProgram::Axis, PDEProgram::Float, PDEProgram::Index, PDEProgram::Nat, PDEProgram::Offset, PDEProgram::_snippet_ix> __forall_ops;
public:
    static PDEProgram::_snippet_ix snippet_ix;
    struct _snippet_ix_specialized {
        inline PDEProgram::Float operator()(const PDEProgram::Array& u, const PDEProgram::Array& v, const PDEProgram::Array& u0, const PDEProgram::Array& u1, const PDEProgram::Array& u2, const PDEProgram::Float& c0, const PDEProgram::Float& c1, const PDEProgram::Float& c2, const PDEProgram::Float& c3, const PDEProgram::Float& c4, const PDEProgram::ScalarIndex& i, const PDEProgram::ScalarIndex& j, const PDEProgram::ScalarIndex& k) {
            PDEProgram::Axis zero = PDEProgram::zero();
            PDEProgram::Offset one = PDEProgram::one.operator()<Offset>();
            PDEProgram::Axis two = PDEProgram::two.operator()<Axis>();
            return PDEProgram::binary_add(PDEProgram::psi(i, j, k, u), PDEProgram::mul(c4, PDEProgram::binary_sub(PDEProgram::mul(c3, PDEProgram::binary_sub(PDEProgram::mul(c1, PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::psi(PDEProgram::mod(PDEProgram::binary_add(i, PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), PDEProgram::shape_0()), j, k, v), PDEProgram::psi(PDEProgram::mod(PDEProgram::binary_add(i, PDEProgram::one.operator()<Offset>()), PDEProgram::shape_0()), j, k, v)), PDEProgram::psi(i, PDEProgram::mod(PDEProgram::binary_add(j, PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), PDEProgram::shape_1()), k, v)), PDEProgram::psi(i, PDEProgram::mod(PDEProgram::binary_add(j, PDEProgram::one.operator()<Offset>()), PDEProgram::shape_1()), k, v)), PDEProgram::psi(i, j, PDEProgram::mod(PDEProgram::binary_add(k, PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), PDEProgram::shape_2()), v)), PDEProgram::psi(i, j, PDEProgram::mod(PDEProgram::binary_add(k, PDEProgram::one.operator()<Offset>()), PDEProgram::shape_2()), v))), PDEProgram::mul(PDEProgram::mul(PDEProgram::three(), c2), PDEProgram::psi(i, j, k, u0)))), PDEProgram::mul(c0, PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::psi(PDEProgram::mod(PDEProgram::binary_add(i, PDEProgram::one.operator()<Offset>()), PDEProgram::shape_0()), j, k, v), PDEProgram::psi(PDEProgram::mod(PDEProgram::binary_add(i, PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), PDEProgram::shape_0()), j, k, v)), PDEProgram::psi(i, j, k, u0)), PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::psi(i, PDEProgram::mod(PDEProgram::binary_add(j, PDEProgram::one.operator()<Offset>()), PDEProgram::shape_1()), k, v), PDEProgram::psi(i, PDEProgram::mod(PDEProgram::binary_add(j, PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), PDEProgram::shape_1()), k, v)), PDEProgram::psi(i, j, k, u1))), PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::psi(i, j, PDEProgram::mod(PDEProgram::binary_add(k, PDEProgram::one.operator()<Offset>()), PDEProgram::shape_2()), v), PDEProgram::psi(i, j, PDEProgram::mod(PDEProgram::binary_add(k, PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), PDEProgram::shape_2()), v)), PDEProgram::psi(i, j, k, u2)))))));
        };
    };

    typedef specialize_psi_ops_2<PDEProgram::Array, PDEProgram::Float, PDEProgram::Index, PDEProgram::Offset, PDEProgram::ScalarIndex, PDEProgram::_snippet_ix_specialized>::AxisLength AxisLength;
    struct _mod {
        inline PDEProgram::ScalarIndex operator()(const PDEProgram::ScalarIndex& six, const PDEProgram::AxisLength& sc) {
            return __specialize_psi_ops_2.mod(six, sc);
        };
    };

    static PDEProgram::_mod mod;
    struct _shape_0 {
        inline PDEProgram::AxisLength operator()() {
            return __specialize_psi_ops_2.shape_0();
        };
    };

    static PDEProgram::_shape_0 shape_0;
    struct _shape_1 {
        inline PDEProgram::AxisLength operator()() {
            return __specialize_psi_ops_2.shape_1();
        };
    };

    static PDEProgram::_shape_1 shape_1;
    struct _shape_2 {
        inline PDEProgram::AxisLength operator()() {
            return __specialize_psi_ops_2.shape_2();
        };
    };

    static PDEProgram::_shape_2 shape_2;
private:
    static specialize_psi_ops_2<PDEProgram::Array, PDEProgram::Float, PDEProgram::Index, PDEProgram::Offset, PDEProgram::ScalarIndex, PDEProgram::_snippet_ix_specialized> __specialize_psi_ops_2;
public:
    static PDEProgram::_snippet_ix_specialized snippet_ix_specialized;
    struct _step {
        inline void operator()(PDEProgram::Array& u0, PDEProgram::Array& u1, PDEProgram::Array& u2, const PDEProgram::Float& nu, const PDEProgram::Float& dx, const PDEProgram::Float& dt) {
            PDEProgram::Float one = PDEProgram::one.operator()<Float>();
            PDEProgram::Float _2 = PDEProgram::two.operator()<Float>();
            PDEProgram::Float c0 = PDEProgram::div(PDEProgram::div(one, _2), dx);
            PDEProgram::Float c1 = PDEProgram::div(PDEProgram::div(one, dx), dx);
            PDEProgram::Float c2 = PDEProgram::div(PDEProgram::div(_2, dx), dx);
            PDEProgram::Float c3 = nu;
            PDEProgram::Float c4 = PDEProgram::div(dt, _2);
            PDEProgram::all_substeps(u0, u1, u2, c0, c1, c2, c3, c4);
        };
    };

    static PDEProgram::_step step;
};
} // examples
} // pde
} // mg_src
} // pde_cpp