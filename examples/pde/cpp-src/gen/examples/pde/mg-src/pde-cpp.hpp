#pragma once

#include "base.hpp"
#include <cassert>


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
    struct _nbCores {
        inline PDEProgram::Nat operator()() {
            return __forall_ops.nbCores();
        };
    };

    static PDEProgram::_nbCores nbCores;
    typedef array_ops::Index Index;
    typedef array_ops::Float Float;
    struct _three {
        inline PDEProgram::Float operator()() {
            return __array_ops.three_float();
        };
    };

    static PDEProgram::_three three;
    struct _unary_sub {
        inline PDEProgram::Float operator()(const PDEProgram::Float& f) {
            return __array_ops.unary_sub(f);
        };
        inline PDEProgram::Offset operator()(const PDEProgram::Offset& o) {
            return __array_ops.unary_sub(o);
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
        inline PDEProgram::Float operator()(const PDEProgram::Float& num, const PDEProgram::Float& den) {
            return __array_ops.div(num, den);
        };
        inline PDEProgram::Array operator()(const PDEProgram::Float& num, const PDEProgram::Array& den) {
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
            return __specialize_psi_ops.forall_ix_snippet_specialized_psi_padded(u, v, u0, u1, u2, c0, c1, c2, c3, c4);
        };
    };

    static PDEProgram::_forall_ix_snippet_specialized_psi_padded forall_ix_snippet_specialized_psi_padded;
    struct _forall_ix_snippet_threaded {
        inline PDEProgram::Array operator()(const PDEProgram::Array& u, const PDEProgram::Array& v, const PDEProgram::Array& u0, const PDEProgram::Array& u1, const PDEProgram::Array& u2, const PDEProgram::Float& c0, const PDEProgram::Float& c1, const PDEProgram::Float& c2, const PDEProgram::Float& c3, const PDEProgram::Float& c4, const PDEProgram::Nat& nbThreads) {
            return __forall_ops.forall_ix_snippet_threaded(u, v, u0, u1, u2, c0, c1, c2, c3, c4, nbThreads);
        };
    };

    static PDEProgram::_forall_ix_snippet_threaded forall_ix_snippet_threaded;
    struct _forall_ix_snippet_tiled {
        inline PDEProgram::Array operator()(const PDEProgram::Array& u, const PDEProgram::Array& v, const PDEProgram::Array& u0, const PDEProgram::Array& u1, const PDEProgram::Array& u2, const PDEProgram::Float& c0, const PDEProgram::Float& c1, const PDEProgram::Float& c2, const PDEProgram::Float& c3, const PDEProgram::Float& c4) {
            return __forall_ops.forall_ix_snippet_tiled(u, v, u0, u1, u2, c0, c1, c2, c3, c4);
        };
    };

    static PDEProgram::_forall_ix_snippet_tiled forall_ix_snippet_tiled;
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

    typedef forall_ops<PDEProgram::Array, PDEProgram::Axis, PDEProgram::Float, PDEProgram::Index, PDEProgram::Nat, PDEProgram::Offset, PDEProgram::_snippet_ix>::AxisLength AxisLength;
    struct _shape_0 {
        inline PDEProgram::AxisLength operator()() {
            return __forall_ops.shape_0();
        };
    };

    static PDEProgram::_shape_0 shape_0;
    struct _shape_1 {
        inline PDEProgram::AxisLength operator()() {
            return __forall_ops.shape_1();
        };
    };

    static PDEProgram::_shape_1 shape_1;
    struct _shape_2 {
        inline PDEProgram::AxisLength operator()() {
            return __forall_ops.shape_2();
        };
    };

    static PDEProgram::_shape_2 shape_2;
    typedef forall_ops<PDEProgram::Array, PDEProgram::Axis, PDEProgram::Float, PDEProgram::Index, PDEProgram::Nat, PDEProgram::Offset, PDEProgram::_snippet_ix>::ScalarIndex ScalarIndex;
    struct _binary_add {
        inline PDEProgram::ScalarIndex operator()(const PDEProgram::ScalarIndex& six, const PDEProgram::Offset& o) {
            return __forall_ops.binary_add(six, o);
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
            return __forall_ops.ix_0(ix);
        };
    };

    static PDEProgram::_ix_0 ix_0;
    struct _ix_1 {
        inline PDEProgram::ScalarIndex operator()(const PDEProgram::Index& ix) {
            return __forall_ops.ix_1(ix);
        };
    };

    static PDEProgram::_ix_1 ix_1;
    struct _ix_2 {
        inline PDEProgram::ScalarIndex operator()(const PDEProgram::Index& ix) {
            return __forall_ops.ix_2(ix);
        };
    };

    static PDEProgram::_ix_2 ix_2;
    struct _make_ix {
        inline PDEProgram::Index operator()(const PDEProgram::ScalarIndex& ix1, const PDEProgram::ScalarIndex& ix2, const PDEProgram::ScalarIndex& ix3) {
            return __forall_ops.make_ix(ix1, ix2, ix3);
        };
    };

    static PDEProgram::_make_ix make_ix;
    struct _mod {
        inline PDEProgram::ScalarIndex operator()(const PDEProgram::ScalarIndex& six, const PDEProgram::AxisLength& sc) {
            return __forall_ops.mod(six, sc);
        };
    };

    static PDEProgram::_mod mod;
    struct _psi {
        inline PDEProgram::Float operator()(const PDEProgram::ScalarIndex& i, const PDEProgram::ScalarIndex& j, const PDEProgram::ScalarIndex& k, const PDEProgram::Array& a) {
            return __specialize_psi_ops.psi(i, j, k, a);
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
            return PDEProgram::binary_add(PDEProgram::psi(i, j, k, u), PDEProgram::mul(c4, PDEProgram::binary_sub(PDEProgram::mul(c3, PDEProgram::binary_sub(PDEProgram::mul(c1, PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::psi(PDEProgram::binary_add(i, PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), j, k, v), PDEProgram::psi(PDEProgram::binary_add(i, PDEProgram::one.operator()<Offset>()), j, k, v)), PDEProgram::psi(i, PDEProgram::binary_add(j, PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), k, v)), PDEProgram::psi(i, PDEProgram::binary_add(j, PDEProgram::one.operator()<Offset>()), k, v)), PDEProgram::psi(i, j, PDEProgram::binary_add(k, PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), v)), PDEProgram::psi(i, j, PDEProgram::binary_add(k, PDEProgram::one.operator()<Offset>()), v))), PDEProgram::mul(PDEProgram::mul(PDEProgram::three(), c2), PDEProgram::psi(i, j, k, u0)))), PDEProgram::mul(c0, PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::psi(PDEProgram::binary_add(i, PDEProgram::one.operator()<Offset>()), j, k, v), PDEProgram::psi(PDEProgram::binary_add(i, PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), j, k, v)), PDEProgram::psi(i, j, k, u0)), PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::psi(i, PDEProgram::binary_add(j, PDEProgram::one.operator()<Offset>()), k, v), PDEProgram::psi(i, PDEProgram::binary_add(j, PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), k, v)), PDEProgram::psi(i, j, k, u1))), PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::psi(i, j, PDEProgram::binary_add(k, PDEProgram::one.operator()<Offset>()), v), PDEProgram::psi(i, j, PDEProgram::binary_add(k, PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), v)), PDEProgram::psi(i, j, k, u2)))))));
        };
    };

private:
    static specialize_psi_ops<PDEProgram::Array, PDEProgram::Float, PDEProgram::Index, PDEProgram::ScalarIndex, PDEProgram::_snippet_ix_specialized> __specialize_psi_ops;
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