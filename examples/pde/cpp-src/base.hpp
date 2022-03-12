#pragma once

#include <cassert>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>

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
                this->m_content = content;
                //this->m_content = new Float[array_size];
                //memcpy(this->m_content, content, array_size * sizeof(Float));
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

            void rotate(const Axis &in_axis, const Offset &in_offset) {
                auto rotate_aux = [&] (Float *content_ptr, const Axis &axis,
                                       const Offset &offset) {
                   // TODO; 
                };
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

    inline void set(const Index &ix, Array &array, const Float &value) {
        array[ix] = value;
    }

    inline Float psi(const Index &ix, const Array &array) {
        return array[ix];
    }

    Array rotate(const Array &array, const Axis &axis, const Offset &offset) {
       Array result = array;
       result.rotate(axis, offset);
       return result;
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
          typename _Offset, typename _PaddedArray, class _snippet_ix>
struct forall_ops {
    private:
        array_ops _array_ops;
    public:
    typedef _Array Array;
    typedef _Axis Axis;
    typedef _Float Float;
    typedef _Index Index;
    typedef _Offset Offset;
    typedef _PaddedArray PaddedArray;

    static _snippet_ix snippet_ix;

    Array forall_snippet_ix(const Array &u, const Array &v, const Array &u0,
                           const Array &u1, const Array &u2, const Float &c0,
                           const Float &c1, const Float &c2, const Float &c3,
                           const Float &c4) {
        auto fn = [&](const Index &ix) {
            return snippet_ix(u, v, u0, u1, u2, c0, c1, c2, c3, c4, ix);
        };
        return _array_ops.forall_ix(u.shape(), fn);
    }

    Array forall_snippet_ix_padded(const PaddedArray &u, const PaddedArray &v,
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
