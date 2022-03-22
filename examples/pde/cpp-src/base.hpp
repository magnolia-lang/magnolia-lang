#pragma once

#include <cassert>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include <omp.h>

#define NB_CORES 4
#define SIDE 256
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

    const Shape empty_shape = Shape(std::vector<size_t>());

    
    /*struct OldIndex {
        std::vector<size_t> value;
        Index(std::vector<size_t> value) : value(value) {}

        inline size_t dim(void) const {
            return this->value.size();
        }

        inline size_t to_linear(const Shape &shape) const {
            if (shape.dim() == 0) { return 0; }

            assert (shape.dim() == 3);
            assert (this->value.size() == 3);

            return this->value[0] * SIDE * SIDE + this->value[1] * SIDE + this->value[0];
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

    };*/

    struct Index {
        size_t value;
        Index(size_t value) : value(value) {};

        inline size_t to_linear(const Shape &shape) const { return this->value; }
    };

    Index emptyIndex() {
        return Index(0); //return Index(std::vector<size_t>());
    }

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

    /*struct Float {
        float value;
        Float(float value) : value(value) {}
        Float() : value(0) {}
    };*/
    typedef float Float;

    inline Float one_float() {
        return 1;
        //return Float(1);
    }

    inline Float two_float() {
        return 2;
        //return Float(2);
    }

    inline Float three_float() {
        return 3;
        //return Float(3);
    }

    //inline Float Float(const float &f) {
    //    return f;
    //}

    struct Array {
        private:
            std::shared_ptr<Float[]> m_content;
            Shape m_shape;

        public:
            Array(const Shape &shape, Float *content) : m_shape(shape) {
                auto array_size = shape.index_space_size();
                this->m_content = std::shared_ptr<Float[]>(new Float[array_size]); //array_size); //new Float[array_size]);
                //std::cout << "ooh? " << array_size << std::endl;
                memcpy(this->m_content.get(), content, array_size * sizeof(Float));
            }

            /*
            Array(const Array &array) : m_shape(array.shape()) {
                auto array_size = this->m_shape.index_space_size();
                this->m_content = std::make_shared<Float[]>(new Float[array_size]);
                memcpy(this->m_content.get(), array.unsafe_content(), array_size * sizeof(Float));
            }*/

            /*void operator delete(void *ptr) {
                std::cout << ((Array*)ptr)->shape().index_space_size() << std::endl;
            }*/

            Array(const Shape &shape) : m_shape(shape) {
                this->m_content = std::shared_ptr<Float[]>(new Float[shape.index_space_size()]); //new Float[shape.index_space_size()]);
            }

            bool is_scalar() const {
                return this->m_shape.components.size() == 0;
            }

            Shape shape() const {
                return this->m_shape;
            }

            size_t size() const {
                return this->m_shape.index_space_size();
            }

            Float as_scalar() const {
                assert (this->m_shape.components.size() == 0);
                return *this->m_content.get();
            }

            /*Float operator[](const Index &ix) const {
                auto linear_ix = ix.to_linear(this->m_shape);
                auto result_shape = Shape(std::vector(this->m_shape.components.begin() + ix.value.size(), this->m_shape.components.end()));
                return this->m_content.get()[linear_ix];
            }*/
            Array operator[](const Index &ix) const {
                return this->psi(ix);
            }

            Float &operator[](const Index &ix) {
                //assert (ix.value.size() == this->m_shape.components.size());
                auto linear_ix = ix.to_linear(this->m_shape);
                //auto result_shape = Shape(std::vector(this->m_shape.components.begin() + ix.value.size(), this->m_shape.components.end()));
                return this->m_content[linear_ix];
            }

            Array psi(const Index &ix) const {
                auto linear_ix = ix.to_linear(this->m_shape);
                //auto result_shape = Shape(std::vector(this->m_shape.components.begin() + ix.value.size(), this->m_shape.components.end()));
                auto result_shape = Shape(std::vector<size_t>());
                return Array(result_shape, this->m_content.get() + linear_ix);
            }

            /*Array &operator[](const Index &ix) {
                auto linear_ix = ix.to_linear(this->m_shape);
                auto result_shape = Shape(std::vector(this->m_shape.components.begin() + ix.value.size(), this->m_shape.components.end()));
                return Array(result_shape, this->m_content + linear_ix);
            }*/

            Float *unsafe_content() const {
                return this->m_content.get();
            }

            void rotate(const Axis &axis, const Offset &offset) {
                if (this->m_shape.components.size() == 0) return;

                auto array_size = this->m_shape.index_space_size();
                auto shape_hd = this->m_shape.components.front();
                size_t stride = this->m_shape.index_space_size() / shape_hd;

                std::shared_ptr<Float[]> new_content(new Float[array_size]); //new Float[array_size]);

                if (axis.value == 0) {
                    
                    if (offset.value < 0) {
                        auto positive_offset = -offset.value;
                        memcpy(new_content.get(), this->m_content.get() + positive_offset * stride, (shape_hd - positive_offset) * stride * sizeof(Float));
                        memcpy(new_content.get() + (shape_hd - positive_offset) * stride, this->m_content.get(), positive_offset * stride * sizeof(Float));
                    }
                    else {
                        memcpy(new_content.get(), this->m_content.get() + (shape_hd - offset.value) * stride, offset.value * stride * sizeof(Float));

                        memcpy(new_content.get() + offset.value * stride, this->m_content.get(), (shape_hd - offset.value) * stride * sizeof(Float));
                    }
                }
                else {
                    auto new_axis = Axis(axis.value - 1);
                    auto subshape = Shape(std::vector(this->m_shape.components.begin() + 1, this->m_shape.components.end()));
                    for (size_t i = 0; i < shape_hd; i += stride) {
                        auto subarray = Array(subshape, this->m_content.get() + i);
                        subarray.rotate(new_axis, offset);
                        memcpy(new_content.get() + i, subarray.unsafe_content(), stride);
                    }
                }

                this->m_content = std::move(new_content);
            }
    };

    /*
    struct _Array {
        private:
            Float m_value;
            //Float *m_content;
            Shape m_shape;
            Array *m_subarrays;

        public:
            Array(const Shape &shape, Float *content) : m_shape(shape) {
                if (shape.components.size() == 0) {
                    this->m_value = *content;
                    this->m_subarrays = NULL;
                }
                else {
                    auto total_size = shape.index_space_size();
                    auto shape_hd = shape.components.front();
                    auto subshape = Shape(std::vector(shape.components.begin() + 1, shape.components.end()));

                    this->m_subarrays = (Array*) ::operator new (shape_hd * sizeof(Array)); //new Array[shape_hd];

                    for (auto start = 0, i = 0; i < shape_hd; start += total_size / shape_hd, i += 1) {
                        this->m_subarrays[i] = Array(subshape, content + start);
                    }
                }
            }

            Array(const Shape &shape) : m_shape(shape) {
                if (shape.components.size() == 0) {
                    this->m_subarrays = NULL;
                }
                else {
                    auto shape_hd = shape.components.front();
                    auto subshape = Shape(std::vector(shape.components.begin() + 1, shape.components.end()));
                    this->m_subarrays = (Array*) ::operator new (shape_hd * sizeof(Array));
                    for (auto i = 0; i < shape_hd; ++i) {
                        this->m_subarrays[i] = Array(subshape);
                    }
                }
            }

            bool is_scalar() const {
                return this->m_shape.components.size() == 0;
            }

            Shape shape() const {
                return this->m_shape;
            }

            size_t size() const {
                return this->m_shape.index_space_size();
            }

            Float &as_scalar() {
                assert (this->m_shape.components.size() == 0);
                return this->m_value;
            }

            Array operator[](const Index &ix) const {
                if (ix.value.size() == 0) {
                    return *this;
                }
                else {
                    auto subix = Index(std::vector(ix.value.begin() + 1, ix.value.end()));
                    return this->m_subarrays[ix.value.front()][subix];
                }
                //auto linear_ix = ix.to_linear(this->m_shape);
                //auto result_shape = Shape(std::vector(this->m_shape.components.begin() + ix.value.size(), this->m_shape.components.end()));
                //return Array(result_shape, this->m_content + linear_ix);
            }

            Array &operator[](const Index &ix) {
                if (ix.value.size() == 0) {
                    return *this;
                }
                else {
                    auto subix = Index(std::vector(ix.value.begin() + 1, ix.value.end()));
                    return this->m_subarrays[ix.value.front()][subix];
                }
                //auto linear_ix = ix.to_linear(this->m_shape);
                //auto result_shape = Shape(std::vector(this->m_shape.components.begin() + ix.value.size(), this->m_shape.components.end()));
                //return Array(result_shape, this->m_content + linear_ix);
            }

            // TODO: clean up and remove
            Float *unsafe_content() const {
                return NULL; //this->m_content;
            }

            void rotate(const Axis &axis, const Offset &offset) {
                if (this->m_shape.components.size() == 0) return;

                auto shape_hd = this->m_shape.components.front();
                if (axis.value == 0) {
                    Array *new_subarrays = (Array*) ::operator new (shape_hd * sizeof(Array));
                    //std::cout << offset.value << std::endl;
                    if (offset.value < 0) {
                        auto positive_offset = -offset.value;
                        memcpy(new_subarrays, this->m_subarrays + positive_offset, (shape_hd - positive_offset) * sizeof(Array));
                        memcpy(new_subarrays + shape_hd - positive_offset, this->m_subarrays, positive_offset * sizeof(Array));
                    }
                    else {
                        memcpy(new_subarrays, this->m_subarrays + shape_hd - offset.value, offset.value * sizeof(Array));
                        memcpy(new_subarrays + offset.value, this->m_subarrays, (shape_hd - offset.value) * sizeof(Array));
                        //memcpy(new_subarrays, this->m_suba
                    }
                    //delete this->m_subarrays;
                    this->m_subarrays = new_subarrays;
                }
                else {
                    auto new_axis = Axis(axis.value - 1);
                    for (auto subarray = this->m_subarrays; subarray < this->m_subarrays + shape_hd; ++subarray) {
                        subarray->rotate(new_axis, offset);
                    }
                }
            }

    };*/

    /*
    type Array;
    type Index;
    type LinearArray;
    type LinearIndex;
    type Range;
    type Shape;
    type Stride;
    */

    // Always contiguous
    struct Range {
        size_t start;
        size_t end;

        Range(size_t end) : start(0), end(end) {}
        Range(size_t start, size_t end) : start(start), end(end) {}
    };

    struct Stride {
        size_t value;
        Stride(size_t value) : value(value) {}
    };

    struct LinearIndex {
        size_t value;
        LinearIndex(size_t value) : value(value) {}
    };

    Range binary_add(const LinearIndex &lix, const Range &range) {
        return Range(lix.value + range.start, lix.value + range.end);
    }

    LinearIndex mul(const LinearIndex &lix, const Stride &stride) {
        return LinearIndex(lix.value * stride.value);
    }

    struct LinearArray {
        private:
            Float *m_content;
            size_t m_length;
        public:
            LinearArray(size_t length, Float *content) : m_length(length) {
                /*
                if (content != NULL) {
                    this->m_content = new Float[length];
                    memcpy(this->m_content, content, length * sizeof(Float));
                }
                else {
                    this->m_content = NULL;
                }*/
                this->m_content = content;
            }

            void set_content(Float *content) { this->m_content = content; }

            Float &operator[](const LinearIndex &lix) {
                return this->m_content[lix.value];
            }

            Float *unsafe_content() const {
                return this->m_content;
            }

            Float operator[](const LinearIndex &lix) const {
                return this->m_content[lix.value];
            }

            Float operator[](size_t i) const {
                return this->m_content[i];
            }

            Float &operator[](size_t i) {
                return this->m_content[i];
            }

            LinearArray operator[](const Range &range) const {
                return LinearArray(range.end - range.start, this->m_content + range.start);
            }

            size_t length() const {
                return this->m_length;
            }
    };

    LinearArray elementsAt(const LinearArray &lia, const Range &range) {
        return lia[range];
    }

    Range iota(const Stride &stride) {
        return Range(stride.value);
    }

    LinearIndex start(const Index &ix, const Shape &shape) {
        return LinearIndex(ix.to_linear(shape));
    }

    Stride stride(const Index &ix, const Shape &shape) {
        return Stride(shape.index_space_size()); // TODO: dummy, this is wrong
        //return Stride(Shape(std::vector(shape.components.begin() + ix.value.size(), shape.components.end())).index_space_size());
    }

    LinearArray rav(const Array &array) {
        size_t length = array.shape().index_space_size();

        auto result = LinearArray(length, NULL);
        result.set_content(array.unsafe_content());
        return result;
        /*Float *elements = new Float[length];

        for (size_t i = 0; i < length; ++i) {
            Index ix = from_linear(array.shape(), i);
            elements[i] = array[ix].as_scalar();
        }*/

        //return LinearArray(length, array.unsafe_content());
    }

    Array toArray(const LinearArray &lia, const Shape &shape) {
        return Array(shape, lia.unsafe_content());
    }

    Shape uniqueShape() {
        return Shape(std::vector<size_t>({ SIDE, SIDE, SIDE }));
    }

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
        return -f; //Float(-f.value);
    }

    inline Float binary_add(Float lhs, Float rhs) {
        return lhs + rhs; //Float(lhs.value + rhs.value);
    }

    inline Float binary_sub(Float lhs, Float rhs) {
        return lhs - rhs;
        //return Float(lhs.value - rhs.value);
    }

    inline Float mul(Float lhs, Float rhs) {
        return lhs * rhs;
        //return Float(lhs.value * rhs.value);
    }

    inline Float div(Float num, Float den) {
        return num / den;
        //return Float(num.value / den.value);
    }

    /* Scalar-Array ops */
    inline Array binary_add(const Float &lhs, const Array &rhs) {
        auto fn = [&](const Index &ix) {
            assert (rhs[ix].is_scalar());
            Float value = binary_add(lhs, rhs[ix].as_scalar());
            return Array(empty_shape, &value);
        };
        
        return forall_ix(rhs.shape(), fn); 
    }

    inline Array binary_sub(const Float &lhs, const Array &rhs) {
        auto fn = [&](const Index &ix) {
            assert (rhs[ix].is_scalar());
            Float value = binary_sub(lhs, rhs[ix].as_scalar());
            return Array(empty_shape, &value);
        };

        return forall_ix(rhs.shape(), fn);
    }

    inline Array mul(const Float &lhs, const Array &rhs) {
        auto fn = [&](const Index &ix) {
            assert (rhs[ix].is_scalar());
            Float value = mul(lhs, rhs[ix].as_scalar());
            return Array(empty_shape, &value);
        };

        return forall_ix(rhs.shape(), fn);
    }

    inline Array div(const Float &lhs, const Array &rhs) {
        auto fn = [&](const Index &ix) {
            assert (rhs[ix].is_scalar());
            Float value = div(lhs, rhs[ix].as_scalar());
            return Array(empty_shape, &value);
        };

        return forall_ix(rhs.shape(), fn);
    }

    /* Array-Array ops */
    inline Array binary_add(const Array &lhs, const Array &rhs) {
        auto fn = [&](const Index &ix) {
            assert (lhs[ix].is_scalar());
            auto result = binary_add(lhs[ix].as_scalar(), rhs[ix].as_scalar());
            return Array(empty_shape, &result);
        };

        return forall_ix(rhs.shape(), fn);
    }

    inline Array binary_sub(const Array &lhs, const Array &rhs) {
        assert (lhs.shape() == rhs.shape());
        auto fn = [&](const Index &ix) {
            assert (lhs[ix].is_scalar());
            auto result = binary_sub(lhs[ix].as_scalar(), rhs[ix].as_scalar());
            return Array(empty_shape, &result);
        };

        return forall_ix(rhs.shape(), fn);
    }

    inline Array mul(const Array &lhs, const Array &rhs) {
        assert (lhs.shape() == rhs.shape());
        auto fn = [&](const Index &ix) {
            assert (lhs[ix].is_scalar());
            auto result = mul(lhs[ix].as_scalar(), rhs[ix].as_scalar());
            return Array(empty_shape, &result);
        };

        return forall_ix(rhs.shape(), fn);
    }

    inline Array div(const Array &lhs, const Array &rhs) {
        assert (lhs.shape() == rhs.shape());
        auto fn = [&](const Index &ix) {
            assert (lhs[ix].is_scalar());
            auto result = div(lhs[ix].as_scalar(), rhs[ix].as_scalar());
            return Array(empty_shape, &result);
        };

        return forall_ix(rhs.shape(), fn);
    }

    inline Array forall_ix(const Shape &shape, auto& fn) {
        auto out_array = Array(shape);
        for (size_t linear_ix = 0; linear_ix < shape.index_space_size();
             ++linear_ix) {
            Index ix = from_linear(shape, linear_ix);
            //std::cout << out_array[ix].value << std::endl;
            //std::cout << linear_ix << std::endl;
            Float scalar_value = fn(ix).as_scalar();
            //std::cout << "casting?" << std::endl;
            out_array[ix] = scalar_value;
            //std::cout << "death" << std::endl;
        }
        return out_array;
    }

    /* Float-LinearArray ops */
    inline LinearArray binary_add(const Float &lhs, const LinearArray &rhs) {
        Float *out_content = new Float[rhs.length()];
        for (size_t i = 0; i < rhs.length(); ++i) {
            out_content[i] = binary_add(lhs, rhs[i]);
        }
        return LinearArray(rhs.length(), out_content);
    }

    inline LinearArray binary_sub(const Float &lhs, const LinearArray &rhs) {
        Float *out_content = new Float[rhs.length()];
        for (size_t i = 0; i < rhs.length(); ++i) {
            out_content[i] = binary_sub(lhs, rhs[i]);
        }
        return LinearArray(rhs.length(), out_content);
    }

    inline LinearArray mul(const Float &lhs, const LinearArray &rhs) {
        Float *out_content = new Float[rhs.length()];
        for (size_t i = 0; i < rhs.length(); ++i) {
            out_content[i] = mul(lhs, rhs[i]);
        }
        return LinearArray(rhs.length(), out_content);
    }

    inline LinearArray div(const Float &lhs, const LinearArray &rhs) {
        Float *out_content = new Float[rhs.length()];
        for (size_t i = 0; i < rhs.length(); ++i) {
            out_content[i] = div(lhs, rhs[i]);
        }
        return LinearArray(rhs.length(), out_content);
    }

    /* LinearArray-LinearArray ops */
    inline LinearArray binary_add(const LinearArray &lhs, const LinearArray &rhs) {
        assert (lhs.length() == rhs.length());
        Float *out_content = new Float[lhs.length()];
        for (size_t i = 0; i < lhs.length(); ++i) {
            out_content[i] = binary_add(lhs[i], rhs[i]);
        }
        return LinearArray(lhs.length(), out_content);
    }

    inline LinearArray binary_sub(const LinearArray &lhs, const LinearArray &rhs) {
        assert (lhs.length() == rhs.length());
        Float *out_content = new Float[lhs.length()];
        for (size_t i = 0; i < lhs.length(); ++i) {
            out_content[i] = binary_sub(lhs[i], rhs[i]);
        }
        return LinearArray(lhs.length(), out_content);
    }

    inline LinearArray mul(const LinearArray &lhs, const LinearArray &rhs) {
        assert (lhs.length() == rhs.length());
        Float *out_content = new Float[lhs.length()];
        for (size_t i = 0; i < lhs.length(); ++i) {
            out_content[i] = mul(lhs[i], rhs[i]);
        }
        return LinearArray(lhs.length(), out_content);
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
        
        // TODO debug
        /*
        for (size_t linear_ix = 0; linear_ix < out_shape.index_space_size();
             ++linear_ix) {
            Index ix = from_linear(out_shape, linear_ix);
            out_array[ix] = fn(ix.to_padded(offsets)).as_scalar();
        }
        */

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

    inline Shape shape(const Array &array) {
        return array.shape();
    }

    inline Shape subshape(const Index &ix, const Shape &shape) {
        // TODO: wrong
        return Shape(std::vector<size_t>());
        // return Shape(std::vector(shape.components.begin() + ix.value.size(), shape.components.end()));
    }

    inline Array psi(const Index &ix, const Array &array) {
        return array.psi(ix);
    }

    Array rotate(const Array &array, const Axis &axis, const Offset &offset) {
       Array result = array;
       result.rotate(axis, offset);
       return result;
    }

    Index rotate_ix(const Index &index, const Axis &axis, const Offset &offset, const Shape &shape) {
        Index new_index = index;
        auto total_ix_space_size = shape.index_space_size();
        new_index.value = (index.value + offset.value * (Shape(std::vector(shape.components.begin() + axis.value + 1, shape.components.end())).index_space_size())) % total_ix_space_size;
        //new_index.value[axis.value] = (index.value[axis.value] + offset.value + shape.components[axis.value]) % shape.components[axis.value];
        return new_index;
    }

    inline Index from_linear(const Shape &shape, size_t linear_ix) {
        /*size_t index_space_size = shape.index_space_size();
        std::vector<size_t> ix;

        for (auto it = shape.components.begin(); it != shape.components.end(); ++it) {
            index_space_size /= *it;
            ix.push_back(linear_ix / index_space_size);
            linear_ix %= index_space_size;
        }

        return Index(ix);*/
        return Index(linear_ix);
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

    _snippet_ix snippet_ix;


    Array forall_ix_snippet(const Array &u, const Array &v, const Array &u0,
                           const Array &u1, const Array &u2, const Float &c0,
                           const Float &c1, const Float &c2, const Float &c3,
                           const Float &c4) {
        assert (u.shape().components.size() == 3);
        auto out_array = Array(u.shape());
        auto out_ptr = out_array.unsafe_content();

        std::cout << "in" << std::endl;
        for (size_t linear_ix = 0; linear_ix < u.shape().index_space_size(); ++linear_ix) {
            auto ix = Index(linear_ix);
            auto result = snippet_ix(u, v, u0, u1, u2, c0, c1, c2, c3, c4, ix);
            out_ptr[linear_ix] = result.as_scalar(); //fn(ix).as_scalar();
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
        std::cout << "in" << std::endl;
        size_t nbThreads = _nbThreads.value;
        auto shape = u.shape();
        omp_set_num_threads(nbThreads);

        size_t thread_axis_length = shape.components[0] / nbThreads;
        size_t thread_domain_size = shape.index_space_size() / nbThreads;
        
        assert (shape.components[0] % nbThreads == 0);
        assert (shape.components.size() == 3);

        auto result = Array(shape);
        auto content_ptr = result.unsafe_content();

        #pragma omp parallel for schedule(static) firstprivate(shape, c0, c1, c2, c3, c4) //firstprivate( u, v, u0, u1, u2, shape, thread_axis_length, thread_domain_size, nbThreads )
        for (size_t linear_ix = 0; linear_ix < SIDE * SIDE * SIDE; ++linear_ix) {
            Index ix = Index(linear_ix);
            content_ptr[linear_ix] = snippet_ix(u, v, u0, u1, u2, c0, c1, c2, c3, c4, ix).as_scalar();
        }

        return result; //Array(shape, content);
        //return Array(u.shape()); // TODO: resolve

        //return _array_ops.forall_ix_threaded(u.shape(), fn, nbThreads.value);
    }

    /*Array forall_ix_snippet_threaded(const Array &u, const Array &v,
                                     const Array &u0, const Array &u1,
                                     const Array &u2, const Float &c0,
                                     const Float &c1, const Float &c2,
                                     const Float &c3, const Float &c4,
                                     const Nat &_nbThreads) {
        size_t nbThreads = _nbThreads.value;
        auto shape = u.shape();
        omp_set_num_threads(nbThreads);

        assert (shape.components[0] % nbThreads == 0);
        assert (shape.components.size() == 3);

        Array out_array = Array(shape);
        auto fn = [&](const Index &ix) {
            return snippet_ix(u, v, u0, u1, u2, c0, c1, c2, c3, c4, ix);
        };
        
        #pragma omp parallel for schedule(static) // firstprivate( u, v, u0, u1, u2, shape, thread_axis_length, thread_domain_size, nbThreads )
        for (size_t i = 0; i < u.shape().components[0]; ++i) {
            for (size_t j = 0; j < u.shape().components[1]; ++j) {
                for (size_t k = 0; k < u.shape().components[2]; ++k) {
                    Index ix = Index(std::vector({ i, j, k }));
                    out_array[ix] = fn(ix);
                }
            }
        }
        return out_array;
    }*/

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
        auto linear_ix = 0;
        
        #pragma omp tile sizes(16, 1, 1024)
        for (size_t i = 0; i < s0; ++i) {
            for (size_t j = 0; j < s1; ++j) {
                for (size_t k = 0; k < s2; ++k) {
                    Index ix = Index(linear_ix); //std::vector({ i, j, k }));
                    out_array[ix] = fn(ix).as_scalar();
                    linear_ix += 1;
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
    array_ops ops;

    auto fn = [&](const array_ops::Index &ix) {
        auto timestep = step * ix.to_linear(shape);
        auto val = array_ops::Float(amplitude * sin(PI * timestep + phase));
        return array_ops::Array(array_ops::Shape(std::vector<size_t>()), &val); //&array_ops::Float(amplitude * sin(PI * t + phase)));
    };
    return ops.forall_ix(shape, fn);
}
