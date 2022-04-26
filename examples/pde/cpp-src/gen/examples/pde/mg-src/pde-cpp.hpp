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
            v0 = BasePDEProgram::snippet(v0, u0, u0, u1, u2, c0, c1, c2, c3, c4);
            v1 = BasePDEProgram::snippet(v1, u1, u0, u1, u2, c0, c1, c2, c3, c4);
            v2 = BasePDEProgram::snippet(v2, u2, u0, u1, u2, c0, c1, c2, c3, c4);
            u0 = BasePDEProgram::snippet(u0, v0, u0, u1, u2, c0, c1, c2, c3, c4);
            u1 = BasePDEProgram::snippet(u1, v1, u0, u1, u2, c0, c1, c2, c3, c4);
            u2 = BasePDEProgram::snippet(u2, v2, u0, u1, u2, c0, c1, c2, c3, c4);
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
    struct _snippet {
        inline BasePDEProgram::Array operator()(const BasePDEProgram::Array& u, const BasePDEProgram::Array& v, const BasePDEProgram::Array& u0, const BasePDEProgram::Array& u1, const BasePDEProgram::Array& u2, const BasePDEProgram::Float& c0, const BasePDEProgram::Float& c1, const BasePDEProgram::Float& c2, const BasePDEProgram::Float& c3, const BasePDEProgram::Float& c4) {
            return BasePDEProgram::binary_add(u, BasePDEProgram::mul(c4, BasePDEProgram::binary_sub(BasePDEProgram::mul(c3, BasePDEProgram::binary_sub(BasePDEProgram::mul(c1, BasePDEProgram::binary_add(BasePDEProgram::binary_add(BasePDEProgram::binary_add(BasePDEProgram::binary_add(BasePDEProgram::binary_add(BasePDEProgram::rotate(v, BasePDEProgram::zero(), BasePDEProgram::unary_sub(BasePDEProgram::one.operator()<Offset>())), BasePDEProgram::rotate(v, BasePDEProgram::zero(), BasePDEProgram::one.operator()<Offset>())), BasePDEProgram::rotate(v, BasePDEProgram::one.operator()<Axis>(), BasePDEProgram::unary_sub(BasePDEProgram::one.operator()<Offset>()))), BasePDEProgram::rotate(v, BasePDEProgram::one.operator()<Axis>(), BasePDEProgram::one.operator()<Offset>())), BasePDEProgram::rotate(v, BasePDEProgram::two.operator()<Axis>(), BasePDEProgram::unary_sub(BasePDEProgram::one.operator()<Offset>()))), BasePDEProgram::rotate(v, BasePDEProgram::two.operator()<Axis>(), BasePDEProgram::one.operator()<Offset>()))), BasePDEProgram::mul(BasePDEProgram::mul(BasePDEProgram::three(), c2), u0))), BasePDEProgram::mul(c0, BasePDEProgram::binary_add(BasePDEProgram::binary_add(BasePDEProgram::mul(BasePDEProgram::binary_sub(BasePDEProgram::rotate(v, BasePDEProgram::zero(), BasePDEProgram::one.operator()<Offset>()), BasePDEProgram::rotate(v, BasePDEProgram::zero(), BasePDEProgram::unary_sub(BasePDEProgram::one.operator()<Offset>()))), u0), BasePDEProgram::mul(BasePDEProgram::binary_sub(BasePDEProgram::rotate(v, BasePDEProgram::one.operator()<Axis>(), BasePDEProgram::one.operator()<Offset>()), BasePDEProgram::rotate(v, BasePDEProgram::one.operator()<Axis>(), BasePDEProgram::unary_sub(BasePDEProgram::one.operator()<Offset>()))), u1)), BasePDEProgram::mul(BasePDEProgram::binary_sub(BasePDEProgram::rotate(v, BasePDEProgram::two.operator()<Axis>(), BasePDEProgram::one.operator()<Offset>()), BasePDEProgram::rotate(v, BasePDEProgram::two.operator()<Axis>(), BasePDEProgram::unary_sub(BasePDEProgram::one.operator()<Offset>()))), u2))))));
        };
    };

    static BasePDEProgram::_snippet snippet;
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
            v0 = PDEProgram::schedule(v0, u0, u0, u1, u2, c0, c1, c2, c3, c4);
            v1 = PDEProgram::schedule(v1, u1, u0, u1, u2, c0, c1, c2, c3, c4);
            v2 = PDEProgram::schedule(v2, u2, u0, u1, u2, c0, c1, c2, c3, c4);
            u0 = PDEProgram::schedule(u0, v0, u0, u1, u2, c0, c1, c2, c3, c4);
            u1 = PDEProgram::schedule(u1, v1, u0, u1, u2, c0, c1, c2, c3, c4);
            u2 = PDEProgram::schedule(u2, v2, u0, u1, u2, c0, c1, c2, c3, c4);
        };
    };

    static PDEProgram::_all_substeps all_substeps;
    struct _binary_add {
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
    struct _psi {
        inline PDEProgram::Float operator()(const PDEProgram::Index& ix, const PDEProgram::Array& array) {
            return __array_ops.psi(ix, array);
        };
    };

    static PDEProgram::_psi psi;
    struct _rotate {
        inline PDEProgram::Array operator()(const PDEProgram::Array& a, const PDEProgram::Axis& axis, const PDEProgram::Offset& o) {
            return __array_ops.rotate(a, axis, o);
        };
    };

    static PDEProgram::_rotate rotate;
    struct _schedule {
        inline PDEProgram::Array operator()(const PDEProgram::Array& u, const PDEProgram::Array& v, const PDEProgram::Array& u0, const PDEProgram::Array& u1, const PDEProgram::Array& u2, const PDEProgram::Float& c0, const PDEProgram::Float& c1, const PDEProgram::Float& c2, const PDEProgram::Float& c3, const PDEProgram::Float& c4) {
            return __forall_ops.schedule(u, v, u0, u1, u2, c0, c1, c2, c3, c4);
        };
    };

    static PDEProgram::_schedule schedule;
    struct _snippet {
        inline PDEProgram::Array operator()(const PDEProgram::Array& u, const PDEProgram::Array& v, const PDEProgram::Array& u0, const PDEProgram::Array& u1, const PDEProgram::Array& u2, const PDEProgram::Float& c0, const PDEProgram::Float& c1, const PDEProgram::Float& c2, const PDEProgram::Float& c3, const PDEProgram::Float& c4) {
            return PDEProgram::binary_add(u, PDEProgram::mul(c4, PDEProgram::binary_sub(PDEProgram::mul(c3, PDEProgram::binary_sub(PDEProgram::mul(c1, PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::rotate(v, PDEProgram::zero(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), PDEProgram::rotate(v, PDEProgram::zero(), PDEProgram::one.operator()<Offset>())), PDEProgram::rotate(v, PDEProgram::one.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), PDEProgram::rotate(v, PDEProgram::one.operator()<Axis>(), PDEProgram::one.operator()<Offset>())), PDEProgram::rotate(v, PDEProgram::two.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), PDEProgram::rotate(v, PDEProgram::two.operator()<Axis>(), PDEProgram::one.operator()<Offset>()))), PDEProgram::mul(PDEProgram::mul(PDEProgram::three(), c2), u0))), PDEProgram::mul(c0, PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::rotate(v, PDEProgram::zero(), PDEProgram::one.operator()<Offset>()), PDEProgram::rotate(v, PDEProgram::zero(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), u0), PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::rotate(v, PDEProgram::one.operator()<Axis>(), PDEProgram::one.operator()<Offset>()), PDEProgram::rotate(v, PDEProgram::one.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), u1)), PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::rotate(v, PDEProgram::two.operator()<Axis>(), PDEProgram::one.operator()<Offset>()), PDEProgram::rotate(v, PDEProgram::two.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), u2))))));
        };
    };

    static PDEProgram::_snippet snippet;
    struct _snippet_ix {
        inline PDEProgram::Float operator()(const PDEProgram::Array& u, const PDEProgram::Array& v, const PDEProgram::Array& u0, const PDEProgram::Array& u1, const PDEProgram::Array& u2, const PDEProgram::Float& c0, const PDEProgram::Float& c1, const PDEProgram::Float& c2, const PDEProgram::Float& c3, const PDEProgram::Float& c4, const PDEProgram::Index& ix) {
            return PDEProgram::binary_add(PDEProgram::psi(ix, u), PDEProgram::mul(c4, PDEProgram::binary_sub(PDEProgram::mul(c3, PDEProgram::binary_sub(PDEProgram::mul(c1, PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::psi(PDEProgram::rotate_ix(ix, PDEProgram::zero(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), v), PDEProgram::psi(PDEProgram::rotate_ix(ix, PDEProgram::zero(), PDEProgram::one.operator()<Offset>()), v)), PDEProgram::psi(PDEProgram::rotate_ix(ix, PDEProgram::one.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), v)), PDEProgram::psi(PDEProgram::rotate_ix(ix, PDEProgram::one.operator()<Axis>(), PDEProgram::one.operator()<Offset>()), v)), PDEProgram::psi(PDEProgram::rotate_ix(ix, PDEProgram::two.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), v)), PDEProgram::psi(PDEProgram::rotate_ix(ix, PDEProgram::two.operator()<Axis>(), PDEProgram::one.operator()<Offset>()), v))), PDEProgram::mul(PDEProgram::mul(PDEProgram::three(), c2), PDEProgram::psi(ix, u0)))), PDEProgram::mul(c0, PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::psi(PDEProgram::rotate_ix(ix, PDEProgram::zero(), PDEProgram::one.operator()<Offset>()), v), PDEProgram::psi(PDEProgram::rotate_ix(ix, PDEProgram::zero(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), v)), PDEProgram::psi(ix, u0)), PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::psi(PDEProgram::rotate_ix(ix, PDEProgram::one.operator()<Axis>(), PDEProgram::one.operator()<Offset>()), v), PDEProgram::psi(PDEProgram::rotate_ix(ix, PDEProgram::one.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), v)), PDEProgram::psi(ix, u1))), PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::psi(PDEProgram::rotate_ix(ix, PDEProgram::two.operator()<Axis>(), PDEProgram::one.operator()<Offset>()), v), PDEProgram::psi(PDEProgram::rotate_ix(ix, PDEProgram::two.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), v)), PDEProgram::psi(ix, u2)))))));
        };
    };

    typedef forall_ops<PDEProgram::Array, PDEProgram::Axis, PDEProgram::Float, PDEProgram::Index, PDEProgram::Nat, PDEProgram::Offset, PDEProgram::_snippet_ix>::ScalarIndex ScalarIndex;
    struct _make_ix {
        inline PDEProgram::Index operator()(const PDEProgram::ScalarIndex& a, const PDEProgram::ScalarIndex& b, const PDEProgram::ScalarIndex& c) {
            return __forall_ops.make_ix(a, b, c);
        };
    };

    static PDEProgram::_make_ix make_ix;
private:
    static forall_ops<PDEProgram::Array, PDEProgram::Axis, PDEProgram::Float, PDEProgram::Index, PDEProgram::Nat, PDEProgram::Offset, PDEProgram::_snippet_ix> __forall_ops;
public:
    static PDEProgram::_snippet_ix snippet_ix;
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

namespace examples {
namespace pde {
namespace mg_src {
namespace pde_cpp {
struct PDEProgram0 {
    struct _two {
        template <typename T>
        inline T operator()() {
            T o;
            PDEProgram0::two0(o);
            return o;
        };
    };

    static PDEProgram0::_two two;
    struct _one {
        template <typename T>
        inline T operator()() {
            T o;
            PDEProgram0::one0(o);
            return o;
        };
    };

    static PDEProgram0::_one one;
private:
    static array_ops __array_ops;
public:
    typedef array_ops::Offset Offset;
private:
    static inline void one0(PDEProgram0::Offset& o) {
        o = __array_ops.one_offset();
    };
public:
    typedef array_ops::Nat Nat;
    typedef array_ops::Index Index;
    typedef array_ops::Float Float;
    struct _three {
        inline PDEProgram0::Float operator()() {
            return __array_ops.three_float();
        };
    };

    static PDEProgram0::_three three;
    struct _unary_sub {
        inline PDEProgram0::Float operator()(const PDEProgram0::Float& f) {
            return __array_ops.unary_sub(f);
        };
        inline PDEProgram0::Offset operator()(const PDEProgram0::Offset& o) {
            return __array_ops.unary_sub(o);
        };
    };

    static PDEProgram0::_unary_sub unary_sub;
private:
    static inline void one0(PDEProgram0::Float& o) {
        o = __array_ops.one_float();
    };
    static inline void two0(PDEProgram0::Float& o) {
        o = __array_ops.two_float();
    };
public:
    typedef array_ops::Axis Axis;
    struct _rotate_ix {
        inline PDEProgram0::Index operator()(const PDEProgram0::Index& ix, const PDEProgram0::Axis& axis, const PDEProgram0::Offset& o) {
            return __array_ops.rotate_ix(ix, axis, o);
        };
    };

    static PDEProgram0::_rotate_ix rotate_ix;
    struct _zero {
        inline PDEProgram0::Axis operator()() {
            return __array_ops.zero_axis();
        };
    };

    static PDEProgram0::_zero zero;
private:
    static inline void one0(PDEProgram0::Axis& o) {
        o = __array_ops.one_axis();
    };
    static inline void two0(PDEProgram0::Axis& o) {
        o = __array_ops.two_axis();
    };
public:
    typedef array_ops::Array Array;
    struct _all_substeps {
        inline void operator()(PDEProgram0::Array& u0, PDEProgram0::Array& u1, PDEProgram0::Array& u2, const PDEProgram0::Float& c0, const PDEProgram0::Float& c1, const PDEProgram0::Float& c2, const PDEProgram0::Float& c3, const PDEProgram0::Float& c4) {
            PDEProgram0::Array v0 = u0;
            PDEProgram0::Array v1 = u1;
            PDEProgram0::Array v2 = u2;
            v0 = PDEProgram0::snippet(v0, u0, u0, u1, u2, c0, c1, c2, c3, c4);
            v1 = PDEProgram0::snippet(v1, u1, u0, u1, u2, c0, c1, c2, c3, c4);
            v2 = PDEProgram0::snippet(v2, u2, u0, u1, u2, c0, c1, c2, c3, c4);
            u0 = PDEProgram0::snippet(u0, v0, u0, u1, u2, c0, c1, c2, c3, c4);
            u1 = PDEProgram0::snippet(u1, v1, u0, u1, u2, c0, c1, c2, c3, c4);
            u2 = PDEProgram0::snippet(u2, v2, u0, u1, u2, c0, c1, c2, c3, c4);
        };
    };

    static PDEProgram0::_all_substeps all_substeps;
    struct _binary_add {
        inline PDEProgram0::Float operator()(const PDEProgram0::Float& lhs, const PDEProgram0::Float& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        inline PDEProgram0::Array operator()(const PDEProgram0::Float& lhs, const PDEProgram0::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        inline PDEProgram0::Array operator()(const PDEProgram0::Array& lhs, const PDEProgram0::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
    };

    static PDEProgram0::_binary_add binary_add;
    struct _binary_sub {
        inline PDEProgram0::Float operator()(const PDEProgram0::Float& lhs, const PDEProgram0::Float& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        inline PDEProgram0::Array operator()(const PDEProgram0::Float& lhs, const PDEProgram0::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        inline PDEProgram0::Array operator()(const PDEProgram0::Array& lhs, const PDEProgram0::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
    };

    static PDEProgram0::_binary_sub binary_sub;
    struct _div {
        inline PDEProgram0::Float operator()(const PDEProgram0::Float& num, const PDEProgram0::Float& den) {
            return __array_ops.div(num, den);
        };
        inline PDEProgram0::Array operator()(const PDEProgram0::Float& num, const PDEProgram0::Array& den) {
            return __array_ops.div(num, den);
        };
    };

    static PDEProgram0::_div div;
    struct _mul {
        inline PDEProgram0::Float operator()(const PDEProgram0::Float& lhs, const PDEProgram0::Float& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        inline PDEProgram0::Array operator()(const PDEProgram0::Float& lhs, const PDEProgram0::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        inline PDEProgram0::Array operator()(const PDEProgram0::Array& lhs, const PDEProgram0::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
    };

    static PDEProgram0::_mul mul;
    struct _psi {
        inline PDEProgram0::Float operator()(const PDEProgram0::Index& ix, const PDEProgram0::Array& array) {
            return __array_ops.psi(ix, array);
        };
    };

    static PDEProgram0::_psi psi;
    struct _rotate {
        inline PDEProgram0::Array operator()(const PDEProgram0::Array& a, const PDEProgram0::Axis& axis, const PDEProgram0::Offset& o) {
            return __array_ops.rotate(a, axis, o);
        };
    };

    static PDEProgram0::_rotate rotate;
    struct _snippet {
        inline PDEProgram0::Array operator()(const PDEProgram0::Array& u, const PDEProgram0::Array& v, const PDEProgram0::Array& u0, const PDEProgram0::Array& u1, const PDEProgram0::Array& u2, const PDEProgram0::Float& c0, const PDEProgram0::Float& c1, const PDEProgram0::Float& c2, const PDEProgram0::Float& c3, const PDEProgram0::Float& c4) {
            return PDEProgram0::binary_add(u, PDEProgram0::mul(c4, PDEProgram0::binary_sub(PDEProgram0::mul(c3, PDEProgram0::binary_sub(PDEProgram0::mul(c1, PDEProgram0::binary_add(PDEProgram0::binary_add(PDEProgram0::binary_add(PDEProgram0::binary_add(PDEProgram0::binary_add(PDEProgram0::rotate(v, PDEProgram0::zero(), PDEProgram0::unary_sub(PDEProgram0::one.operator()<Offset>())), PDEProgram0::rotate(v, PDEProgram0::zero(), PDEProgram0::one.operator()<Offset>())), PDEProgram0::rotate(v, PDEProgram0::one.operator()<Axis>(), PDEProgram0::unary_sub(PDEProgram0::one.operator()<Offset>()))), PDEProgram0::rotate(v, PDEProgram0::one.operator()<Axis>(), PDEProgram0::one.operator()<Offset>())), PDEProgram0::rotate(v, PDEProgram0::two.operator()<Axis>(), PDEProgram0::unary_sub(PDEProgram0::one.operator()<Offset>()))), PDEProgram0::rotate(v, PDEProgram0::two.operator()<Axis>(), PDEProgram0::one.operator()<Offset>()))), PDEProgram0::mul(PDEProgram0::mul(PDEProgram0::three(), c2), u0))), PDEProgram0::mul(c0, PDEProgram0::binary_add(PDEProgram0::binary_add(PDEProgram0::mul(PDEProgram0::binary_sub(PDEProgram0::rotate(v, PDEProgram0::zero(), PDEProgram0::one.operator()<Offset>()), PDEProgram0::rotate(v, PDEProgram0::zero(), PDEProgram0::unary_sub(PDEProgram0::one.operator()<Offset>()))), u0), PDEProgram0::mul(PDEProgram0::binary_sub(PDEProgram0::rotate(v, PDEProgram0::one.operator()<Axis>(), PDEProgram0::one.operator()<Offset>()), PDEProgram0::rotate(v, PDEProgram0::one.operator()<Axis>(), PDEProgram0::unary_sub(PDEProgram0::one.operator()<Offset>()))), u1)), PDEProgram0::mul(PDEProgram0::binary_sub(PDEProgram0::rotate(v, PDEProgram0::two.operator()<Axis>(), PDEProgram0::one.operator()<Offset>()), PDEProgram0::rotate(v, PDEProgram0::two.operator()<Axis>(), PDEProgram0::unary_sub(PDEProgram0::one.operator()<Offset>()))), u2))))));
        };
    };

    static PDEProgram0::_snippet snippet;
    struct _step {
        inline void operator()(PDEProgram0::Array& u0, PDEProgram0::Array& u1, PDEProgram0::Array& u2, const PDEProgram0::Float& nu, const PDEProgram0::Float& dx, const PDEProgram0::Float& dt) {
            PDEProgram0::Float one = PDEProgram0::one.operator()<Float>();
            PDEProgram0::Float _2 = PDEProgram0::two.operator()<Float>();
            PDEProgram0::Float c0 = PDEProgram0::div(PDEProgram0::div(one, _2), dx);
            PDEProgram0::Float c1 = PDEProgram0::div(PDEProgram0::div(one, dx), dx);
            PDEProgram0::Float c2 = PDEProgram0::div(PDEProgram0::div(_2, dx), dx);
            PDEProgram0::Float c3 = nu;
            PDEProgram0::Float c4 = PDEProgram0::div(dt, _2);
            PDEProgram0::all_substeps(u0, u1, u2, c0, c1, c2, c3, c4);
        };
    };

    static PDEProgram0::_step step;
};
} // examples
} // pde
} // mg_src
} // pde_cpp

namespace examples {
namespace pde {
namespace mg_src {
namespace pde_cpp {
struct PDEProgram1 {
    struct _two {
        template <typename T>
        inline T operator()() {
            T o;
            PDEProgram1::two0(o);
            return o;
        };
    };

    static PDEProgram1::_two two;
    struct _one {
        template <typename T>
        inline T operator()() {
            T o;
            PDEProgram1::one0(o);
            return o;
        };
    };

    static PDEProgram1::_one one;
private:
    static array_ops __array_ops;
public:
    typedef array_ops::Offset Offset;
private:
    static inline void one0(PDEProgram1::Offset& o) {
        o = __array_ops.one_offset();
    };
public:
    typedef array_ops::Nat Nat;
    typedef array_ops::Index Index;
    typedef array_ops::Float Float;
    struct _three {
        inline PDEProgram1::Float operator()() {
            return __array_ops.three_float();
        };
    };

    static PDEProgram1::_three three;
    struct _unary_sub {
        inline PDEProgram1::Offset operator()(const PDEProgram1::Offset& o) {
            return __array_ops.unary_sub(o);
        };
        inline PDEProgram1::Float operator()(const PDEProgram1::Float& f) {
            return __array_ops.unary_sub(f);
        };
    };

    static PDEProgram1::_unary_sub unary_sub;
private:
    static inline void one0(PDEProgram1::Float& o) {
        o = __array_ops.one_float();
    };
    static inline void two0(PDEProgram1::Float& o) {
        o = __array_ops.two_float();
    };
public:
    typedef array_ops::Axis Axis;
    struct _rotate_ix {
        inline PDEProgram1::Index operator()(const PDEProgram1::Index& ix, const PDEProgram1::Axis& axis, const PDEProgram1::Offset& o) {
            return __array_ops.rotate_ix(ix, axis, o);
        };
    };

    static PDEProgram1::_rotate_ix rotate_ix;
    struct _zero {
        inline PDEProgram1::Axis operator()() {
            return __array_ops.zero_axis();
        };
    };

    static PDEProgram1::_zero zero;
private:
    static inline void one0(PDEProgram1::Axis& o) {
        o = __array_ops.one_axis();
    };
    static inline void two0(PDEProgram1::Axis& o) {
        o = __array_ops.two_axis();
    };
public:
    typedef array_ops::Array Array;
    struct _all_substeps {
        inline void operator()(PDEProgram1::Array& u0, PDEProgram1::Array& u1, PDEProgram1::Array& u2, const PDEProgram1::Float& c0, const PDEProgram1::Float& c1, const PDEProgram1::Float& c2, const PDEProgram1::Float& c3, const PDEProgram1::Float& c4) {
            PDEProgram1::Array v0 = u0;
            PDEProgram1::Array v1 = u1;
            PDEProgram1::Array v2 = u2;
            v0 = PDEProgram1::schedule(v0, u0, u0, u1, u2, c0, c1, c2, c3, c4);
            v1 = PDEProgram1::schedule(v1, u1, u0, u1, u2, c0, c1, c2, c3, c4);
            v2 = PDEProgram1::schedule(v2, u2, u0, u1, u2, c0, c1, c2, c3, c4);
            u0 = PDEProgram1::schedule(u0, v0, u0, u1, u2, c0, c1, c2, c3, c4);
            u1 = PDEProgram1::schedule(u1, v1, u0, u1, u2, c0, c1, c2, c3, c4);
            u2 = PDEProgram1::schedule(u2, v2, u0, u1, u2, c0, c1, c2, c3, c4);
        };
    };

    static PDEProgram1::_all_substeps all_substeps;
    struct _binary_add {
        inline PDEProgram1::Float operator()(const PDEProgram1::Float& lhs, const PDEProgram1::Float& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        inline PDEProgram1::Array operator()(const PDEProgram1::Float& lhs, const PDEProgram1::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        inline PDEProgram1::Array operator()(const PDEProgram1::Array& lhs, const PDEProgram1::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
    };

    static PDEProgram1::_binary_add binary_add;
    struct _binary_sub {
        inline PDEProgram1::Float operator()(const PDEProgram1::Float& lhs, const PDEProgram1::Float& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        inline PDEProgram1::Array operator()(const PDEProgram1::Float& lhs, const PDEProgram1::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        inline PDEProgram1::Array operator()(const PDEProgram1::Array& lhs, const PDEProgram1::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
    };

    static PDEProgram1::_binary_sub binary_sub;
    struct _div {
        inline PDEProgram1::Array operator()(const PDEProgram1::Float& num, const PDEProgram1::Array& den) {
            return __array_ops.div(num, den);
        };
        inline PDEProgram1::Float operator()(const PDEProgram1::Float& num, const PDEProgram1::Float& den) {
            return __array_ops.div(num, den);
        };
    };

    static PDEProgram1::_div div;
    struct _mul {
        inline PDEProgram1::Float operator()(const PDEProgram1::Float& lhs, const PDEProgram1::Float& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        inline PDEProgram1::Array operator()(const PDEProgram1::Float& lhs, const PDEProgram1::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        inline PDEProgram1::Array operator()(const PDEProgram1::Array& lhs, const PDEProgram1::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
    };

    static PDEProgram1::_mul mul;
    struct _psi {
        inline PDEProgram1::Float operator()(const PDEProgram1::Index& ix, const PDEProgram1::Array& array) {
            return __array_ops.psi(ix, array);
        };
    };

    static PDEProgram1::_psi psi;
    struct _rotate {
        inline PDEProgram1::Array operator()(const PDEProgram1::Array& a, const PDEProgram1::Axis& axis, const PDEProgram1::Offset& o) {
            return __array_ops.rotate(a, axis, o);
        };
    };

    static PDEProgram1::_rotate rotate;
    struct _schedule {
        inline PDEProgram1::Array operator()(const PDEProgram1::Array& u, const PDEProgram1::Array& v, const PDEProgram1::Array& u0, const PDEProgram1::Array& u1, const PDEProgram1::Array& u2, const PDEProgram1::Float& c0, const PDEProgram1::Float& c1, const PDEProgram1::Float& c2, const PDEProgram1::Float& c3, const PDEProgram1::Float& c4) {
            return __forall_ops.schedule(u, v, u0, u1, u2, c0, c1, c2, c3, c4);
        };
    };

    static PDEProgram1::_schedule schedule;
    struct _snippet {
        inline PDEProgram1::Array operator()(const PDEProgram1::Array& u, const PDEProgram1::Array& v, const PDEProgram1::Array& u0, const PDEProgram1::Array& u1, const PDEProgram1::Array& u2, const PDEProgram1::Float& c0, const PDEProgram1::Float& c1, const PDEProgram1::Float& c2, const PDEProgram1::Float& c3, const PDEProgram1::Float& c4) {
            return PDEProgram1::binary_add(u, PDEProgram1::mul(c4, PDEProgram1::binary_sub(PDEProgram1::mul(c3, PDEProgram1::binary_sub(PDEProgram1::mul(c1, PDEProgram1::binary_add(PDEProgram1::binary_add(PDEProgram1::binary_add(PDEProgram1::binary_add(PDEProgram1::binary_add(PDEProgram1::rotate(v, PDEProgram1::zero(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>())), PDEProgram1::rotate(v, PDEProgram1::zero(), PDEProgram1::one.operator()<Offset>())), PDEProgram1::rotate(v, PDEProgram1::one.operator()<Axis>(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>()))), PDEProgram1::rotate(v, PDEProgram1::one.operator()<Axis>(), PDEProgram1::one.operator()<Offset>())), PDEProgram1::rotate(v, PDEProgram1::two.operator()<Axis>(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>()))), PDEProgram1::rotate(v, PDEProgram1::two.operator()<Axis>(), PDEProgram1::one.operator()<Offset>()))), PDEProgram1::mul(PDEProgram1::mul(PDEProgram1::three(), c2), u0))), PDEProgram1::mul(c0, PDEProgram1::binary_add(PDEProgram1::binary_add(PDEProgram1::mul(PDEProgram1::binary_sub(PDEProgram1::rotate(v, PDEProgram1::zero(), PDEProgram1::one.operator()<Offset>()), PDEProgram1::rotate(v, PDEProgram1::zero(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>()))), u0), PDEProgram1::mul(PDEProgram1::binary_sub(PDEProgram1::rotate(v, PDEProgram1::one.operator()<Axis>(), PDEProgram1::one.operator()<Offset>()), PDEProgram1::rotate(v, PDEProgram1::one.operator()<Axis>(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>()))), u1)), PDEProgram1::mul(PDEProgram1::binary_sub(PDEProgram1::rotate(v, PDEProgram1::two.operator()<Axis>(), PDEProgram1::one.operator()<Offset>()), PDEProgram1::rotate(v, PDEProgram1::two.operator()<Axis>(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>()))), u2))))));
        };
    };

    static PDEProgram1::_snippet snippet;
    struct _snippet_ix {
        inline PDEProgram1::Float operator()(const PDEProgram1::Array& u, const PDEProgram1::Array& v, const PDEProgram1::Array& u0, const PDEProgram1::Array& u1, const PDEProgram1::Array& u2, const PDEProgram1::Float& c0, const PDEProgram1::Float& c1, const PDEProgram1::Float& c2, const PDEProgram1::Float& c3, const PDEProgram1::Float& c4, const PDEProgram1::Index& ix) {
            return PDEProgram1::binary_add(PDEProgram1::psi(ix, u), PDEProgram1::mul(c4, PDEProgram1::binary_sub(PDEProgram1::mul(c3, PDEProgram1::binary_sub(PDEProgram1::mul(c1, PDEProgram1::binary_add(PDEProgram1::binary_add(PDEProgram1::binary_add(PDEProgram1::binary_add(PDEProgram1::binary_add(PDEProgram1::psi(PDEProgram1::rotate_ix(ix, PDEProgram1::zero(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>())), v), PDEProgram1::psi(PDEProgram1::rotate_ix(ix, PDEProgram1::zero(), PDEProgram1::one.operator()<Offset>()), v)), PDEProgram1::psi(PDEProgram1::rotate_ix(ix, PDEProgram1::one.operator()<Axis>(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>())), v)), PDEProgram1::psi(PDEProgram1::rotate_ix(ix, PDEProgram1::one.operator()<Axis>(), PDEProgram1::one.operator()<Offset>()), v)), PDEProgram1::psi(PDEProgram1::rotate_ix(ix, PDEProgram1::two.operator()<Axis>(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>())), v)), PDEProgram1::psi(PDEProgram1::rotate_ix(ix, PDEProgram1::two.operator()<Axis>(), PDEProgram1::one.operator()<Offset>()), v))), PDEProgram1::mul(PDEProgram1::mul(PDEProgram1::three(), c2), PDEProgram1::psi(ix, u0)))), PDEProgram1::mul(c0, PDEProgram1::binary_add(PDEProgram1::binary_add(PDEProgram1::mul(PDEProgram1::binary_sub(PDEProgram1::psi(PDEProgram1::rotate_ix(ix, PDEProgram1::zero(), PDEProgram1::one.operator()<Offset>()), v), PDEProgram1::psi(PDEProgram1::rotate_ix(ix, PDEProgram1::zero(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>())), v)), PDEProgram1::psi(ix, u0)), PDEProgram1::mul(PDEProgram1::binary_sub(PDEProgram1::psi(PDEProgram1::rotate_ix(ix, PDEProgram1::one.operator()<Axis>(), PDEProgram1::one.operator()<Offset>()), v), PDEProgram1::psi(PDEProgram1::rotate_ix(ix, PDEProgram1::one.operator()<Axis>(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>())), v)), PDEProgram1::psi(ix, u1))), PDEProgram1::mul(PDEProgram1::binary_sub(PDEProgram1::psi(PDEProgram1::rotate_ix(ix, PDEProgram1::two.operator()<Axis>(), PDEProgram1::one.operator()<Offset>()), v), PDEProgram1::psi(PDEProgram1::rotate_ix(ix, PDEProgram1::two.operator()<Axis>(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>())), v)), PDEProgram1::psi(ix, u2)))))));
        };
    };

    typedef forall_ops<PDEProgram1::Array, PDEProgram1::Axis, PDEProgram1::Float, PDEProgram1::Index, PDEProgram1::Nat, PDEProgram1::Offset, PDEProgram1::_snippet_ix>::ScalarIndex ScalarIndex;
    struct _make_ix {
        inline PDEProgram1::Index operator()(const PDEProgram1::ScalarIndex& a, const PDEProgram1::ScalarIndex& b, const PDEProgram1::ScalarIndex& c) {
            return __forall_ops.make_ix(a, b, c);
        };
    };

    static PDEProgram1::_make_ix make_ix;
private:
    static forall_ops<PDEProgram1::Array, PDEProgram1::Axis, PDEProgram1::Float, PDEProgram1::Index, PDEProgram1::Nat, PDEProgram1::Offset, PDEProgram1::_snippet_ix> __forall_ops;
public:
    static PDEProgram1::_snippet_ix snippet_ix;
    struct _step {
        inline void operator()(PDEProgram1::Array& u0, PDEProgram1::Array& u1, PDEProgram1::Array& u2, const PDEProgram1::Float& nu, const PDEProgram1::Float& dx, const PDEProgram1::Float& dt) {
            PDEProgram1::Float one = PDEProgram1::one.operator()<Float>();
            PDEProgram1::Float _2 = PDEProgram1::two.operator()<Float>();
            PDEProgram1::Float c0 = PDEProgram1::div(PDEProgram1::div(one, _2), dx);
            PDEProgram1::Float c1 = PDEProgram1::div(PDEProgram1::div(one, dx), dx);
            PDEProgram1::Float c2 = PDEProgram1::div(PDEProgram1::div(_2, dx), dx);
            PDEProgram1::Float c3 = nu;
            PDEProgram1::Float c4 = PDEProgram1::div(dt, _2);
            PDEProgram1::all_substeps(u0, u1, u2, c0, c1, c2, c3, c4);
        };
    };

    static PDEProgram1::_step step;
};
} // examples
} // pde
} // mg_src
} // pde_cpp

namespace examples {
namespace pde {
namespace mg_src {
namespace pde_cpp {
struct PDEProgram2 {
    struct _two {
        template <typename T>
        inline T operator()() {
            T o;
            PDEProgram2::two0(o);
            return o;
        };
    };

    static PDEProgram2::_two two;
    struct _one {
        template <typename T>
        inline T operator()() {
            T o;
            PDEProgram2::one0(o);
            return o;
        };
    };

    static PDEProgram2::_one one;
private:
    static array_ops __array_ops;
public:
    typedef array_ops::Offset Offset;
private:
    static inline void one0(PDEProgram2::Offset& o) {
        o = __array_ops.one_offset();
    };
public:
    typedef array_ops::Nat Nat;
    struct _nbCores {
        inline PDEProgram2::Nat operator()() {
            return __forall_ops.nbCores();
        };
    };

    static PDEProgram2::_nbCores nbCores;
    typedef array_ops::Index Index;
    typedef array_ops::Float Float;
    struct _three {
        inline PDEProgram2::Float operator()() {
            return __array_ops.three_float();
        };
    };

    static PDEProgram2::_three three;
    struct _unary_sub {
        inline PDEProgram2::Float operator()(const PDEProgram2::Float& f) {
            return __array_ops.unary_sub(f);
        };
        inline PDEProgram2::Offset operator()(const PDEProgram2::Offset& o) {
            return __array_ops.unary_sub(o);
        };
    };

    static PDEProgram2::_unary_sub unary_sub;
private:
    static inline void one0(PDEProgram2::Float& o) {
        o = __array_ops.one_float();
    };
    static inline void two0(PDEProgram2::Float& o) {
        o = __array_ops.two_float();
    };
public:
    typedef array_ops::Axis Axis;
    struct _rotate_ix {
        inline PDEProgram2::Index operator()(const PDEProgram2::Index& ix, const PDEProgram2::Axis& axis, const PDEProgram2::Offset& o) {
            return __array_ops.rotate_ix(ix, axis, o);
        };
    };

    static PDEProgram2::_rotate_ix rotate_ix;
    struct _zero {
        inline PDEProgram2::Axis operator()() {
            return __array_ops.zero_axis();
        };
    };

    static PDEProgram2::_zero zero;
private:
    static inline void one0(PDEProgram2::Axis& o) {
        o = __array_ops.one_axis();
    };
    static inline void two0(PDEProgram2::Axis& o) {
        o = __array_ops.two_axis();
    };
public:
    typedef array_ops::Array Array;
    struct _all_substeps {
        inline void operator()(PDEProgram2::Array& u0, PDEProgram2::Array& u1, PDEProgram2::Array& u2, const PDEProgram2::Float& c0, const PDEProgram2::Float& c1, const PDEProgram2::Float& c2, const PDEProgram2::Float& c3, const PDEProgram2::Float& c4) {
            PDEProgram2::Array v0 = u0;
            PDEProgram2::Array v1 = u1;
            PDEProgram2::Array v2 = u2;
            v0 = PDEProgram2::schedule_threaded(v0, u0, u0, u1, u2, c0, c1, c2, c3, c4, PDEProgram2::nbCores());
            v1 = PDEProgram2::schedule_threaded(v1, u1, u0, u1, u2, c0, c1, c2, c3, c4, PDEProgram2::nbCores());
            v2 = PDEProgram2::schedule_threaded(v2, u2, u0, u1, u2, c0, c1, c2, c3, c4, PDEProgram2::nbCores());
            u0 = PDEProgram2::schedule_threaded(u0, v0, u0, u1, u2, c0, c1, c2, c3, c4, PDEProgram2::nbCores());
            u1 = PDEProgram2::schedule_threaded(u1, v1, u0, u1, u2, c0, c1, c2, c3, c4, PDEProgram2::nbCores());
            u2 = PDEProgram2::schedule_threaded(u2, v2, u0, u1, u2, c0, c1, c2, c3, c4, PDEProgram2::nbCores());
        };
    };

    static PDEProgram2::_all_substeps all_substeps;
    struct _binary_add {
        inline PDEProgram2::Array operator()(const PDEProgram2::Array& lhs, const PDEProgram2::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        inline PDEProgram2::Array operator()(const PDEProgram2::Float& lhs, const PDEProgram2::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        inline PDEProgram2::Float operator()(const PDEProgram2::Float& lhs, const PDEProgram2::Float& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
    };

    static PDEProgram2::_binary_add binary_add;
    struct _binary_sub {
        inline PDEProgram2::Array operator()(const PDEProgram2::Array& lhs, const PDEProgram2::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        inline PDEProgram2::Array operator()(const PDEProgram2::Float& lhs, const PDEProgram2::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        inline PDEProgram2::Float operator()(const PDEProgram2::Float& lhs, const PDEProgram2::Float& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
    };

    static PDEProgram2::_binary_sub binary_sub;
    struct _div {
        inline PDEProgram2::Float operator()(const PDEProgram2::Float& num, const PDEProgram2::Float& den) {
            return __array_ops.div(num, den);
        };
        inline PDEProgram2::Array operator()(const PDEProgram2::Float& num, const PDEProgram2::Array& den) {
            return __array_ops.div(num, den);
        };
    };

    static PDEProgram2::_div div;
    struct _mul {
        inline PDEProgram2::Array operator()(const PDEProgram2::Array& lhs, const PDEProgram2::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        inline PDEProgram2::Array operator()(const PDEProgram2::Float& lhs, const PDEProgram2::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        inline PDEProgram2::Float operator()(const PDEProgram2::Float& lhs, const PDEProgram2::Float& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
    };

    static PDEProgram2::_mul mul;
    struct _psi {
        inline PDEProgram2::Float operator()(const PDEProgram2::Index& ix, const PDEProgram2::Array& array) {
            return __array_ops.psi(ix, array);
        };
    };

    static PDEProgram2::_psi psi;
    struct _rotate {
        inline PDEProgram2::Array operator()(const PDEProgram2::Array& a, const PDEProgram2::Axis& axis, const PDEProgram2::Offset& o) {
            return __array_ops.rotate(a, axis, o);
        };
    };

    static PDEProgram2::_rotate rotate;
    struct _schedule {
        inline PDEProgram2::Array operator()(const PDEProgram2::Array& u, const PDEProgram2::Array& v, const PDEProgram2::Array& u0, const PDEProgram2::Array& u1, const PDEProgram2::Array& u2, const PDEProgram2::Float& c0, const PDEProgram2::Float& c1, const PDEProgram2::Float& c2, const PDEProgram2::Float& c3, const PDEProgram2::Float& c4) {
            return __forall_ops.schedule(u, v, u0, u1, u2, c0, c1, c2, c3, c4);
        };
    };

    static PDEProgram2::_schedule schedule;
    struct _schedule_threaded {
        inline PDEProgram2::Array operator()(const PDEProgram2::Array& u, const PDEProgram2::Array& v, const PDEProgram2::Array& u0, const PDEProgram2::Array& u1, const PDEProgram2::Array& u2, const PDEProgram2::Float& c0, const PDEProgram2::Float& c1, const PDEProgram2::Float& c2, const PDEProgram2::Float& c3, const PDEProgram2::Float& c4, const PDEProgram2::Nat& nbThreads) {
            return __forall_ops.schedule_threaded(u, v, u0, u1, u2, c0, c1, c2, c3, c4, nbThreads);
        };
    };

    static PDEProgram2::_schedule_threaded schedule_threaded;
    struct _snippet {
        inline PDEProgram2::Array operator()(const PDEProgram2::Array& u, const PDEProgram2::Array& v, const PDEProgram2::Array& u0, const PDEProgram2::Array& u1, const PDEProgram2::Array& u2, const PDEProgram2::Float& c0, const PDEProgram2::Float& c1, const PDEProgram2::Float& c2, const PDEProgram2::Float& c3, const PDEProgram2::Float& c4) {
            return PDEProgram2::binary_add(u, PDEProgram2::mul(c4, PDEProgram2::binary_sub(PDEProgram2::mul(c3, PDEProgram2::binary_sub(PDEProgram2::mul(c1, PDEProgram2::binary_add(PDEProgram2::binary_add(PDEProgram2::binary_add(PDEProgram2::binary_add(PDEProgram2::binary_add(PDEProgram2::rotate(v, PDEProgram2::zero(), PDEProgram2::unary_sub(PDEProgram2::one.operator()<Offset>())), PDEProgram2::rotate(v, PDEProgram2::zero(), PDEProgram2::one.operator()<Offset>())), PDEProgram2::rotate(v, PDEProgram2::one.operator()<Axis>(), PDEProgram2::unary_sub(PDEProgram2::one.operator()<Offset>()))), PDEProgram2::rotate(v, PDEProgram2::one.operator()<Axis>(), PDEProgram2::one.operator()<Offset>())), PDEProgram2::rotate(v, PDEProgram2::two.operator()<Axis>(), PDEProgram2::unary_sub(PDEProgram2::one.operator()<Offset>()))), PDEProgram2::rotate(v, PDEProgram2::two.operator()<Axis>(), PDEProgram2::one.operator()<Offset>()))), PDEProgram2::mul(PDEProgram2::mul(PDEProgram2::three(), c2), u0))), PDEProgram2::mul(c0, PDEProgram2::binary_add(PDEProgram2::binary_add(PDEProgram2::mul(PDEProgram2::binary_sub(PDEProgram2::rotate(v, PDEProgram2::zero(), PDEProgram2::one.operator()<Offset>()), PDEProgram2::rotate(v, PDEProgram2::zero(), PDEProgram2::unary_sub(PDEProgram2::one.operator()<Offset>()))), u0), PDEProgram2::mul(PDEProgram2::binary_sub(PDEProgram2::rotate(v, PDEProgram2::one.operator()<Axis>(), PDEProgram2::one.operator()<Offset>()), PDEProgram2::rotate(v, PDEProgram2::one.operator()<Axis>(), PDEProgram2::unary_sub(PDEProgram2::one.operator()<Offset>()))), u1)), PDEProgram2::mul(PDEProgram2::binary_sub(PDEProgram2::rotate(v, PDEProgram2::two.operator()<Axis>(), PDEProgram2::one.operator()<Offset>()), PDEProgram2::rotate(v, PDEProgram2::two.operator()<Axis>(), PDEProgram2::unary_sub(PDEProgram2::one.operator()<Offset>()))), u2))))));
        };
    };

    static PDEProgram2::_snippet snippet;
    struct _snippet_ix {
        inline PDEProgram2::Float operator()(const PDEProgram2::Array& u, const PDEProgram2::Array& v, const PDEProgram2::Array& u0, const PDEProgram2::Array& u1, const PDEProgram2::Array& u2, const PDEProgram2::Float& c0, const PDEProgram2::Float& c1, const PDEProgram2::Float& c2, const PDEProgram2::Float& c3, const PDEProgram2::Float& c4, const PDEProgram2::Index& ix) {
            return PDEProgram2::binary_add(PDEProgram2::psi(ix, u), PDEProgram2::mul(c4, PDEProgram2::binary_sub(PDEProgram2::mul(c3, PDEProgram2::binary_sub(PDEProgram2::mul(c1, PDEProgram2::binary_add(PDEProgram2::binary_add(PDEProgram2::binary_add(PDEProgram2::binary_add(PDEProgram2::binary_add(PDEProgram2::psi(PDEProgram2::rotate_ix(ix, PDEProgram2::zero(), PDEProgram2::unary_sub(PDEProgram2::one.operator()<Offset>())), v), PDEProgram2::psi(PDEProgram2::rotate_ix(ix, PDEProgram2::zero(), PDEProgram2::one.operator()<Offset>()), v)), PDEProgram2::psi(PDEProgram2::rotate_ix(ix, PDEProgram2::one.operator()<Axis>(), PDEProgram2::unary_sub(PDEProgram2::one.operator()<Offset>())), v)), PDEProgram2::psi(PDEProgram2::rotate_ix(ix, PDEProgram2::one.operator()<Axis>(), PDEProgram2::one.operator()<Offset>()), v)), PDEProgram2::psi(PDEProgram2::rotate_ix(ix, PDEProgram2::two.operator()<Axis>(), PDEProgram2::unary_sub(PDEProgram2::one.operator()<Offset>())), v)), PDEProgram2::psi(PDEProgram2::rotate_ix(ix, PDEProgram2::two.operator()<Axis>(), PDEProgram2::one.operator()<Offset>()), v))), PDEProgram2::mul(PDEProgram2::mul(PDEProgram2::three(), c2), PDEProgram2::psi(ix, u0)))), PDEProgram2::mul(c0, PDEProgram2::binary_add(PDEProgram2::binary_add(PDEProgram2::mul(PDEProgram2::binary_sub(PDEProgram2::psi(PDEProgram2::rotate_ix(ix, PDEProgram2::zero(), PDEProgram2::one.operator()<Offset>()), v), PDEProgram2::psi(PDEProgram2::rotate_ix(ix, PDEProgram2::zero(), PDEProgram2::unary_sub(PDEProgram2::one.operator()<Offset>())), v)), PDEProgram2::psi(ix, u0)), PDEProgram2::mul(PDEProgram2::binary_sub(PDEProgram2::psi(PDEProgram2::rotate_ix(ix, PDEProgram2::one.operator()<Axis>(), PDEProgram2::one.operator()<Offset>()), v), PDEProgram2::psi(PDEProgram2::rotate_ix(ix, PDEProgram2::one.operator()<Axis>(), PDEProgram2::unary_sub(PDEProgram2::one.operator()<Offset>())), v)), PDEProgram2::psi(ix, u1))), PDEProgram2::mul(PDEProgram2::binary_sub(PDEProgram2::psi(PDEProgram2::rotate_ix(ix, PDEProgram2::two.operator()<Axis>(), PDEProgram2::one.operator()<Offset>()), v), PDEProgram2::psi(PDEProgram2::rotate_ix(ix, PDEProgram2::two.operator()<Axis>(), PDEProgram2::unary_sub(PDEProgram2::one.operator()<Offset>())), v)), PDEProgram2::psi(ix, u2)))))));
        };
    };

    typedef forall_ops<PDEProgram2::Array, PDEProgram2::Axis, PDEProgram2::Float, PDEProgram2::Index, PDEProgram2::Nat, PDEProgram2::Offset, PDEProgram2::_snippet_ix>::ScalarIndex ScalarIndex;
    struct _make_ix {
        inline PDEProgram2::Index operator()(const PDEProgram2::ScalarIndex& a, const PDEProgram2::ScalarIndex& b, const PDEProgram2::ScalarIndex& c) {
            return __forall_ops.make_ix(a, b, c);
        };
    };

    static PDEProgram2::_make_ix make_ix;
private:
    static forall_ops<PDEProgram2::Array, PDEProgram2::Axis, PDEProgram2::Float, PDEProgram2::Index, PDEProgram2::Nat, PDEProgram2::Offset, PDEProgram2::_snippet_ix> __forall_ops;
public:
    static PDEProgram2::_snippet_ix snippet_ix;
    struct _step {
        inline void operator()(PDEProgram2::Array& u0, PDEProgram2::Array& u1, PDEProgram2::Array& u2, const PDEProgram2::Float& nu, const PDEProgram2::Float& dx, const PDEProgram2::Float& dt) {
            PDEProgram2::Float one = PDEProgram2::one.operator()<Float>();
            PDEProgram2::Float _2 = PDEProgram2::two.operator()<Float>();
            PDEProgram2::Float c0 = PDEProgram2::div(PDEProgram2::div(one, _2), dx);
            PDEProgram2::Float c1 = PDEProgram2::div(PDEProgram2::div(one, dx), dx);
            PDEProgram2::Float c2 = PDEProgram2::div(PDEProgram2::div(_2, dx), dx);
            PDEProgram2::Float c3 = nu;
            PDEProgram2::Float c4 = PDEProgram2::div(dt, _2);
            PDEProgram2::all_substeps(u0, u1, u2, c0, c1, c2, c3, c4);
        };
    };

    static PDEProgram2::_step step;
};
} // examples
} // pde
} // mg_src
} // pde_cpp