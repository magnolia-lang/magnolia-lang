#pragma once

#include <cassert>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>

#include <omp.h>

#define NB_CORES 2

struct array_ops {

    struct Shape {
        std::vector<size_t> components;

        Shape(std::vector<size_t> components) : components(components) {}
        
        size_t index_space_size(void) const {
            size_t total = 1;
            for (auto it = components.begin(); it != components.end(); ++it) {
                total *= *it;
            }
            return total;
        }

        size_t dim(void) const {
            return this->components.size();
        }

        bool operator==(const Shape &other) const {
            return this->components == other.components;
        }

        size_t operator[](const size_t &i) const {
            return this->components[i];
        }

        size_t &operator[](const size_t &i) {
            return this->components[i];
        }
    };

    struct Index {
        std::vector<size_t> value;
        Index(std::vector<size_t> value) : value(value) {}

        size_t dim(void) const {
            return this->value.size();
        }

        inline size_t to_linear(const Shape &shape) const {
            size_t index_space_size = shape.index_space_size();
            size_t linear_ix = 0;

            assert (this->dim() == shape.dim());

            for (auto shape_it = shape.components.begin(), ix_it = this->value.begin();
                 shape_it != shape.components.end() && ix_it != this->value.end();
                 ++shape_it, ++ix_it) {
                index_space_size /= *shape_it;
                linear_ix += *ix_it * index_space_size;
            }

            return linear_ix;
        }

        inline Index to_padded(const std::vector<size_t> &offsets) const {
            std::vector<size_t> out_index;
            for (auto ix_it = this->value.begin(), offset_it = offsets.begin();
                 ix_it != this->value.end() && offset_it != offsets.end();
                 ++ix_it, ++offset_it) {
                out_index.push_back(*ix_it + *offset_it);
            }

            return Index(out_index);
        }

    };

    struct Offset {
        int value;
        Offset(int value) : value(value) {}
        Offset() : value(0) {}
    };

    inline Offset one_offset() {
        return Offset(1);
    }

    struct Axis {
        size_t value;
        Axis(size_t value) : value(value) {}
        Axis() : value(0) {}
    };

    inline Axis zero_axis() {
        return Axis(0);
    }

    inline Axis one_axis() {
        return Axis(1);
    }

    inline Axis two_axis() {
        return Axis(2);
    }

    struct Float {
        float value;
        Float(float value) : value(value) {}
        Float() : value(0) {}
    };

    inline Float one_float() {
        return Float(1);
    }

    inline Float two_float() {
        return Float(2);
    }

    inline Float three_float() {
        return Float(3);
    }

    struct Array {
        private:
            Float *m_content;
            Shape m_shape;

        public:
            Array(const Shape &shape, Float *content) : m_shape(shape) {
                auto array_size = shape.index_space_size();
                // TODO: not this, lol
                //this->m_content = content;
                this->m_content = new Float[array_size];
                memcpy(this->m_content, content, array_size * sizeof(Float));
            }

            Array(const Shape &shape) : m_shape(shape) {
                auto array_size = shape.index_space_size();
                this->m_content = new Float[array_size];
            }

            Shape shape() const {
                return this->m_shape;
            }

            size_t size() const {
                return this->m_shape.index_space_size();
            }

            Float operator[](const Index &ix) const {
                auto linear_ix = ix.to_linear(this->m_shape);
                return this->m_content[linear_ix];
            }

            Float &operator[](const Index &ix) {
                auto linear_ix = ix.to_linear(this->m_shape);
                return this->m_content[linear_ix];
            }

            Float *unsafe_content() const {
                return this->m_content;
            }

            void rotate(const Axis &axis, const Offset &offset) {
                Shape start_shape = Shape(std::vector<size_t>(this->m_shape.components.begin(), this->m_shape.components.begin() + axis.value));
                size_t stride = Shape(std::vector<size_t>(this->m_shape.components.begin() + axis.value, this->m_shape.components.end())).index_space_size();

                auto new_content = new Float[this->m_shape.index_space_size()];

                //std::cout << "------- " << axis.value << " " << offset.value << std::endl;
                for (size_t linear_ix = 0; linear_ix < start_shape.index_space_size();
                     ++linear_ix) {

                    auto content_ix = linear_ix * stride;
                    //std::cout << content_ix << std::endl;
                    if (offset.value < 0) { // left shift
                        auto offset_value = offset.value % stride;
                        // if offset is -4 and stride is 3, this is 1, i.e.
                        // the number of elements to drop at the start of the
                        // array
                        if (offset_value > 0) {
                            offset_value = stride - offset_value;
                        }

                        memcpy(new_content + content_ix, this->m_content + content_ix + offset_value, (stride - offset_value) * sizeof(Float));
                        memcpy(new_content + content_ix + stride - offset_value, this->m_content, offset_value * sizeof(Float));
                    }
                    else { // right shift
                        auto offset_value = offset.value % stride;
                        memcpy(new_content + content_ix, this->m_content + content_ix + stride - offset_value, offset_value * sizeof(Float));
                        memcpy(new_content + content_ix + offset_value, this->m_content + content_ix, (stride - offset_value) * sizeof(Float));
                    }
                }

                this->m_content = new_content;
            }
    };

    struct PaddedArray {
        private:
            Float *m_content;
            Shape m_shape;
            std::vector<std::pair<size_t, size_t>> m_bounds;

        public:
            PaddedArray(const Array &array) : m_shape(array.shape()) {
                auto array_size = this->m_shape.index_space_size();
                // TODO: not this lol
                //this->m_content = array.unsafe_content();
                this->m_content = new Float[array_size];
                memcpy(this->m_content, array.unsafe_content(), array_size * sizeof(Float));
                

                for (auto shape_it = this->m_shape.components.begin();
                     shape_it != this->m_shape.components.end(); ++shape_it) {
                    this->m_bounds.push_back(std::make_pair(0, *shape_it));
                }
            }

            Array outer() const {
                return Array(this->m_shape, this->m_content);
            }

            std::vector<std::pair<size_t, size_t>> bounds() const {
                return this->m_bounds;
            }

            Float *unsafe_content() const {
                return this->m_content;
            }

            void cpadl(const Axis &axis) {
                if (axis.value != 0) { return; } // TODO: this is not impl

                auto old_array_size = this->m_shape.index_space_size();
                auto one_stride = old_array_size / this->m_shape.components.front();
                auto new_array_size = old_array_size + one_stride;

                Float *new_content = new Float[new_array_size];

                auto bi = this->m_bounds.front().first,
                     ei = this->m_bounds.front().second,
                     si = this->m_shape.components.front();
                //std::cout << bi + si - ei << std::endl;
                /*std::cout << bi + si - ei << std::endl;
                std::cout << old_array_size << std::endl;
                std::cout << new_array_size << std::endl;
                std::cout << one_stride << std::endl;
                std::cout << sizeof(Float) << std::endl;*/
                memcpy(new_content, this->m_content + (ei - bi - 1) * one_stride, one_stride * sizeof(Float));
                memcpy(new_content + one_stride, this->m_content, old_array_size * sizeof(Float));
                
                this->m_shape.components.front() += 1;
                this->m_bounds.front().first += 1;
                this->m_bounds.front().second += 1;
                this->m_content = new_content; 
            }

            void cpadr(const Axis &axis) {
                if (axis.value != 0) { return; } // TODO: this is not impl

                auto old_array_size = this->m_shape.index_space_size();
                auto one_stride = old_array_size / this->m_shape.components.front();
                auto new_array_size = old_array_size + one_stride;

                Float *new_content = new Float[new_array_size];

                auto bi = this->m_bounds.front().first,
                     ei = this->m_bounds.front().second,
                     si = this->m_shape.components.front();
                //std::cout << bi + si - ei << std::endl;
                /*std::cout << bi + si - ei << std::endl;
                std::cout << old_array_size << std::endl;
                std::cout << new_array_size << std::endl;
                std::cout << one_stride << std::endl;
                std::cout << sizeof(Float) << std::endl;*/
                memcpy(new_content, this->m_content, old_array_size * sizeof(Float));
                memcpy(new_content + old_array_size,
                       this->m_content + (bi + si - ei) * one_stride, // * sizeof(Float),
                       one_stride * sizeof(Float));
                
                this->m_shape.components.front() += 1;
                this->m_content = new_content; 
            }
    };


    PaddedArray asPadded(const Array &array) { return PaddedArray(array); }

    PaddedArray cpadl(const PaddedArray &array, const Axis &axis) {
        auto result = array;
        result.cpadl(axis);
        return result;
    }

    PaddedArray cpadr(const PaddedArray &array, const Axis &axis) {
        auto result = array;
        result.cpadr(axis);
        return result;
    }

    /* Float ops */
    inline Float unary_sub(Float f) {
        return Float(-f.value);
    }

    inline Float binary_add(Float lhs, Float rhs) {
        return Float(lhs.value + rhs.value);
    }

    inline Float binary_sub(Float lhs, Float rhs) {
        return Float(lhs.value - rhs.value);
    }

    inline Float mul(Float lhs, Float rhs) {
        return Float(lhs.value * rhs.value);
    }

    inline Float div(Float num, Float den) {
        return Float(num.value / den.value);
    }

    /* Scalar-Array ops */
    inline Array binary_add(const Float &lhs, const Array &rhs) {
        auto fn = [&](const Index &ix) {
            return binary_add(lhs, rhs[ix]);
        };
        
        return forall_ix(rhs.shape(), fn); 
    }

    inline Array binary_sub(const Float &lhs, const Array &rhs) {
        auto fn = [&](const Index &ix) {
            return binary_sub(lhs, rhs[ix]);
        };

        return forall_ix(rhs.shape(), fn);
    }

    inline Array mul(const Float &lhs, const Array &rhs) {
        auto fn = [&](const Index &ix) {
            return mul(lhs, rhs[ix]);
        };

        return forall_ix(rhs.shape(), fn);
    }

    inline Array div(const Float &lhs, const Array &rhs) {
        auto fn = [&](const Index &ix) {
            return div(lhs, rhs[ix]);
        };

        return forall_ix(rhs.shape(), fn);
    }

    /* Array-Array ops */
    inline Array binary_add(const Array &lhs, const Array &rhs) {
        assert (lhs.shape() == rhs.shape());
        auto fn = [&](const Index &ix) {
            return binary_add(lhs[ix], rhs[ix]);
        };

        return forall_ix(rhs.shape(), fn);
    }

    inline Array binary_sub(const Array &lhs, const Array &rhs) {
        assert (lhs.shape() == rhs.shape());
        auto fn = [&](const Index &ix) {
            return binary_sub(lhs[ix], rhs[ix]);
        };

        return forall_ix(rhs.shape(), fn);
    }

    inline Array mul(const Array &lhs, const Array &rhs) {
        assert (lhs.shape() == rhs.shape());
        auto fn = [&](const Index &ix) {
            return mul(lhs[ix], rhs[ix]);
        };

        return forall_ix(rhs.shape(), fn);
    }

    inline Array div(const Array &lhs, const Array &rhs) {
        assert (lhs.shape() == rhs.shape());
        auto fn = [&](const Index &ix) {
            return div(lhs[ix], rhs[ix]);
        };

        return forall_ix(rhs.shape(), fn);
    }

    inline Array forall_ix(const Shape &shape, auto& fn) {
        auto out_array = Array(shape);
        for (size_t linear_ix = 0; linear_ix < shape.index_space_size();
             ++linear_ix) {
            Index ix = from_linear(shape, linear_ix);
            out_array[ix] = fn(ix);
        }
        return out_array;
    }

    inline Array forall_ix_padded(std::vector<std::pair<size_t, size_t>> bounds,
                                  auto &fn) {
        std::vector<size_t> offsets;
        std::vector<size_t> shape_components;
        for (auto bounds_it = bounds.begin(); bounds_it != bounds.end(); ++bounds_it) {
            offsets.push_back((*bounds_it).first);
            shape_components.push_back((*bounds_it).second - (*bounds_it).first);
        }

        auto out_shape = Shape(shape_components);
        auto out_array = Array(out_shape);
        for (size_t linear_ix = 0; linear_ix < out_shape.index_space_size();
             ++linear_ix) {
            Index ix = from_linear(out_shape, linear_ix);
            out_array[ix] = fn(ix.to_padded(offsets));
        }

        return out_array;
    }

    // TODO: we only need to dlift for distributed anyway
    /*inline Array forall_ix_threaded(const Shape &shape, auto &fn, size_t nbThreads) {
        auto subshape = Shape(std::vector(shape.components.begin() + 1, shape.components.end()));
        auto out_array = Array(shape);
        auto subshape_ix_space_size = subshape.index_space_size();

        omp_set_num_threads(nbThreads);

        #pragma omp parallel for
        for (auto i = 0; i < nbThreads; ++i) {
            for (auto sub_linear_ix = 0; sub_linear_ix < subshape_ix_space_size; ++sub_linear_ix) {
                Index ix = from_linear(shape, i * subshape_ix_space_size + sub_linear_ix);
                out_array[ix] = fn(ix);
            }
        }

        return out_array;
    }*/

    inline Array forall_ix_threaded(const Shape &shape, auto &fn, size_t nbThreads) {
        omp_set_num_threads(nbThreads);

        Float **threaded_content = new Float*[nbThreads];
        auto thread_domain_size = shape.index_space_size() / nbThreads;

        #pragma omp parallel for private( fn )
        for (size_t tix = 0; tix < nbThreads; ++tix) {
            Float *local_array = new Float[thread_domain_size];
            threaded_content[tix] = local_array;
            for (size_t offset_ix = 0; offset_ix < thread_domain_size; ++offset_ix) {
                auto linear_ix = tix * thread_domain_size + offset_ix;
                Index ix = from_linear(shape, linear_ix);
                local_array[offset_ix] = fn(ix);
            }
        }

        Float *content = new Float[shape.index_space_size()];

        for (size_t tix = 0; tix < nbThreads; ++tix) {
            memcpy(content + tix * thread_domain_size, threaded_content[tix], thread_domain_size);
        }

        return Array(shape, content);
    }

    inline void set(const Index &ix, Array &array, const Float &value) {
        array[ix] = value;
    }

    inline Shape shape(const Array &array) {
        return array.shape();
    }

    inline Float psi(const Index &ix, const Array &array) {
        return array[ix];
    }

    Array rotate(const Array &array, const Axis &axis, const Offset &offset) {
       Array result = array;
       result.rotate(axis, offset);
       return result;
    }

    Index rotate_ix(const Index &index, const Axis &axis, const Offset &offset, const Shape &shape) {
        Index new_index = index;
        new_index.value[axis.value] = (index.value[axis.value] + offset.value + shape.components[axis.value]) % shape.components[axis.value];
        return new_index;
    }

    inline Index from_linear(const Shape &shape, size_t linear_ix) {
        size_t index_space_size = shape.index_space_size();
        std::vector<size_t> ix;

        for (auto it = shape.components.begin(); it != shape.components.end(); ++it) {
            index_space_size /= *it;
            ix.push_back(linear_ix / index_space_size);
            linear_ix %= index_space_size;
        }

        return Index(ix);
    }

    /* Offset utils */
    inline Offset unary_sub(const Offset &offset) {
        return Offset(-offset.value);
    }
};

// TODO: bypassing extend mechanism, not great
template <typename _Array, typename _Axis, typename _Float, typename _Index,
          typename _Nat, typename _Offset, typename _PaddedArray, class _snippet_ix>
struct forall_ops {
    private:
        array_ops _array_ops;
    public:
    typedef _Array Array;
    typedef _Axis Axis;
    typedef _Float Float;
    typedef _Index Index;
    typedef _Nat Nat;
    typedef _Offset Offset;
    typedef _PaddedArray PaddedArray;

    typedef array_ops::Shape Shape;

    static _snippet_ix snippet_ix;


    Array forall_ix_snippet(const Array &u, const Array &v, const Array &u0,
                           const Array &u1, const Array &u2, const Float &c0,
                           const Float &c1, const Float &c2, const Float &c3,
                           const Float &c4) {
        auto fn = [&](const Index &ix) {
            return snippet_ix(u, v, u0, u1, u2, c0, c1, c2, c3, c4, ix);
        };

        assert (u.shape().components.size() == 3);
        auto out_array = Array(u.shape());

        for (size_t i = 0; i < u.shape().components[0]; ++i) {
            for (size_t j = 0; j < u.shape().components[1]; ++j) {
                for (size_t k = 0; k < u.shape().components[2]; ++k) {
                    Index ix = Index(std::vector({ i, j, k }));
                    out_array[ix] = fn(ix);
                }
            }
        }

        return out_array;
        //return _array_ops.forall_ix(u.shape(), fn);
    }

    Array forall_ix_snippet_padded(const PaddedArray &u, const PaddedArray &v,
                            const PaddedArray &u0, const PaddedArray &u1,
                            const PaddedArray &u2, const Float &c0,
                            const Float &c1, const Float &c2, const Float &c3,
                            const Float &c4) {
        auto fn = [&](const Index &ix) {
            return snippet_ix(u.outer(), v.outer(), u0.outer(), u1.outer(),
                              u2.outer(), c0, c1, c2, c3, c4, ix);
        };
        return _array_ops.forall_ix_padded(u.bounds(), fn);
    }

    Array forall_ix_snippet_threaded(const Array &u, const Array &v,
                                     const Array &u0, const Array &u1,
                                     const Array &u2, const Float &c0,
                                     const Float &c1, const Float &c2,
                                     const Float &c3, const Float &c4,
                                     const Nat &_nbThreads) {
        size_t nbThreads = _nbThreads.value;
        auto shape = u.shape();
        omp_set_num_threads(nbThreads);

        Float **threaded_content = new Float*[nbThreads];
        size_t thread_axis_length = shape.components[0] / nbThreads;
        size_t thread_domain_size = shape.index_space_size() / nbThreads;
        
        assert (shape.components[0] % nbThreads == 0);
        assert (shape.components.size() == 3);

        #pragma omp parallel for schedule(static) firstprivate( shape ) // firstprivate( u, v, u0, u1, u2, shape )
        for (size_t tix = 0; tix < nbThreads; ++tix) {
            auto fn = [&](const Index &ix) {
                return snippet_ix(u, v, u0, u1, u2, c0, c1, c2, c3, c4, ix);
            };
            Float *local_array = new Float[thread_domain_size];
            threaded_content[tix] = local_array;
            for (size_t i = 0; i < thread_axis_length; ++i) {
                for (size_t j = 0; j < shape.components[1]; ++j) {
                    for (size_t k = 0; k < shape.components[2]; ++k) {
                //for (size_t offset_ix = 0; offset_ix < thread_domain_size; ++offset_ix) {
                //auto linear_ix = tix * thread_domain_size + offset_ix;
                //Index ix = _array_ops.from_linear(shape, linear_ix);
                        size_t offset_ix = i * (shape.components[1] * shape.components[2]) + j * shape.components[2] + k;
                        Index ix = Index(std::vector({ tix * thread_axis_length + i, j, k }));
                        local_array[offset_ix] = fn(ix);
                    }
                }
            }
        }

        Float *content = new Float[shape.index_space_size()];

        for (size_t tix = 0; tix < nbThreads; ++tix) {
            memcpy(content + tix * thread_domain_size, threaded_content[tix], thread_domain_size);
        }

        return Array(shape, content);

        //return _array_ops.forall_ix_threaded(u.shape(), fn, nbThreads.value);
    }

    Array forall_ix_snippet_tiled(const Array &u, const Array &v,
                                  const Array &u0, const Array &u1,
                                  const Array &u2, const Float &c0,
                                  const Float &c1, const Float &c2,
                                  const Float &c3, const Float &c4) {
                                  //const Nat &_nbThreads) {
        auto fn = [&](const Index &ix) {
            return snippet_ix(u, v, u0, u1, u2, c0, c1, c2, c3, c4, ix);
        };

        auto s0 = u.shape().components[0],
             s1 = u.shape().components[1],
             s2 = u.shape().components[2];

        assert (u.shape().components.size() == 3);
        auto s0tiles = 4, s1tiles = 4, s2tiles = 4;
        auto out_array = Array(u.shape());

        for (size_t ti = 0; ti < s0; ti += s0/s0tiles) {
            for (size_t tj = 0; tj < s1; tj += s1/s1tiles) {
                for (size_t tk = 0; tk < s2; tk += s2/s2tiles) {
                    for (size_t i = ti; i < ti + s0/s0tiles; i++) {
                        for (size_t j = tj; j < tj + s1/s1tiles; j++) {
                            for (size_t k = tk; k < tk + s2/s2tiles; k++) {
                                Index ix = Index(std::vector({ i, j, k }));
                                out_array[ix] = fn(ix);
                            }
                        }
                    }
                }
            }
        }

        return out_array;
    }

    Array unliftAndUnpad(const Array &array, const Axis &axis,
                         const Nat &paddingAmount) {
        Shape padded_shape = array.shape();
        auto d = padded_shape[0];

        assert (axis.value == 0); // not implemented for other axes, no time

        auto out_shape = Shape(std::vector(padded_shape.components.begin() + 1,
                                           padded_shape.components.end()));
        out_shape[0] -= paddingAmount.value * 2;
        out_shape[0] *= d;

        auto out_content = new Float[out_shape.index_space_size()];

        auto stride = out_shape.index_space_size() / out_shape[0];
        auto strides_to_keep = stride * out_shape[0] / d;
        auto strides_to_skip = stride * padded_shape[1] / d;
        auto offset = stride * paddingAmount.value;

        Float *orig_content = array.unsafe_content();

        for (size_t i = 0; i < d; ++i) {
            memcpy(out_content + i * strides_to_keep, orig_content + i * strides_to_skip + offset, strides_to_keep * sizeof(Float));
        }

        return Array(out_shape, out_content);
    }

    Array padAndLift(const Array &array, const Axis &axis, const Nat &d,
                     const Nat &paddingAmount) {
        Shape in_shape = array.shape();
        Float *in_content = array.unsafe_content();

        auto modulus = in_shape.index_space_size();
        Shape out_shape = array.shape();
        out_shape.components.insert(out_shape.components.begin(), d.value);
        out_shape[1] /= d.value;
        out_shape[1] += 2 * paddingAmount.value;

        assert (axis.value == 0); // not implemented for other axes, no time

        auto out_content = new Float[out_shape.index_space_size()];

        // TODO: fix, not correct
        /*
        auto in_stride = modulus / in_shape[0];
        auto out_stride = out_shape.index_space_size() / d.value;

        for (size_t i = 0; i < d; ++i) {
            // Get left padding
            int left_padding_offset = (((i - 2) * in_stride) + modulus) % modulus;
            memcpy(out_content + i * out_stride, in_content + left_padding_offset, 
        }
        */

        // TODO: hacky just to get something
        memcpy(out_content, in_content, in_shape.index_space_size() * sizeof(Float));
        memcpy(out_content + in_shape.index_space_size(), in_content, (out_shape.index_space_size() - in_shape.index_space_size()) * sizeof(Float));

        return Array(out_shape, out_content);
    }
};


struct hardware_info {
    struct Nat {
        size_t value;
        Nat(size_t value) : value(value) {}
        Nat() : value(0) {}
    };

    inline Nat nbCores() {
        return Nat(NB_CORES);
    }

    inline Nat one() { return Nat(1); }
};

inline array_ops::Array dumpsine(const array_ops::Shape &shape) {
    double step = 0.01;
    double PI = 3.14159265358979323846;
    double amplitude = 10.0;
    double phase = 0.0125;
    double t = 0.0;
    array_ops ops;

    auto fn = [&](const array_ops::Index &ix) {
        auto timestep = step * ix.to_linear(shape);
        return array_ops::Float(amplitude * sin(PI * t + phase));
    };
    return ops.forall_ix(shape, fn);
}
