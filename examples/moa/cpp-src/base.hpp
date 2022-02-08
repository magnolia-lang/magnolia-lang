#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

template <typename _Element>
struct array {

    typedef _Element Element;
    typedef size_t UInt32;
    typedef std::vector<UInt32> Index;
    typedef std::vector<UInt32> Shape;

    struct Array {

        private:

            Element * _content;
            Shape _sh;

        public:

            Array(Shape shape) {
                _sh = shape;

                using std::begin;
                using std::end;

                auto array_size = std::accumulate(begin(_sh),
                                                  end(_sh), 1,
                                                  std::multiplies<UInt32>());

                _content = (Element *)calloc(
                    array_size, sizeof(Element));
            }

            inline UInt32 _dim() {
                return _sh.size();
            }

            inline Shape _shape() {
                return _sh;
            }

            inline Element _get(const UInt32 ix) {
                return _content[ix];
            }

            inline void _set(const UInt32 ix, Element val) {
                _content[ix] = val;
            }

            inline UInt32 _get_shape_elem(const UInt32 i) {
                return _sh.at(i);
            }

            inline UInt32 _total() {
                using std::begin;
                using std::end;

                return std::accumulate(begin(_sh),
                                                  end(_sh), 1,
                                                  std::multiplies<UInt32>());
            }
    };


    inline Array get(Array a, Index ix) {

        // total
        if (ix.size() == dim(a)) {

            Shape sh = shape(a);

            UInt32 accum = 0;

            for (int i = 0; i < dim(a); i++) {

                std::vector<UInt32>::const_iterator first = sh.begin() + (i+1);
                std::vector<UInt32>::const_iterator last = sh.begin() + dim(a);

                std::vector<UInt32> subshape(first, last);

                int reduced = std::accumulate(begin(subshape),
                                                  end(subshape), 1,
                                                  std::multiplies<UInt32>());

                accum += ix.at(i) * reduced;
            }

            Shape res_shape = create_shape1(1);
            Array res = Array(res_shape);

            res._set(0,a._get(accum));
            return res;

        }
         // partial
        else if (ix.size() < dim(a)) {

            Shape sh = shape(a);

            // calculates the remaining index space not given in the partial ix
            std::vector<UInt32>::const_iterator first = sh.begin() + (ix.size());
            std::vector<UInt32>::const_iterator last = sh.begin() + dim(a);

            std::vector<UInt32> subshape(first, last);

            Array sub_array = Array(subshape);

            UInt32 total_elems = total(sub_array);

            std::vector<Index> indices;

            // populates a vector with empty total indices
            for (auto i = 0; i < total(sub_array); i++) {
                std::vector<UInt32> total;

                for (auto j = 0; j < ix.size(); j++) {
                    total.push_back(ix.at(j));
                }

                indices.push_back(total);
            }

            // keeps track of how deep down in the subshape we are
            UInt32 level = 1;
            // generates all total indices
            for (auto i = 0; i < subshape.size(); i++) {
                UInt32 current_sh_ix = subshape.at(i);
                UInt32 split = total_elems/(current_sh_ix*level);
                UInt32 current_ix = 0;
                int counter = 0;

                for (auto j = 0; j < indices.size(); j++) {
                    if (counter == split) {
                        counter = 0;
                        if (current_ix == current_sh_ix - 1) {
                            current_ix = 0;
                        }
                        else {
                            current_ix++;
                        }

                    }
                    indices.at(j).push_back(current_ix);
                    counter++;
                }
                level += level;
            }

            for (auto i = 0; i < total_elems; i++) {
                sub_array._set(i, unwrap_scalar(get(a, indices.at(i))));
            }

            return sub_array;
        }

        // invalid index
        else {
            std::cout << "Invalid index, out of bounds" << std::endl;
        }
    }

    inline void set(Array a, Index ix, const Element val) {

        if (ix.size() == dim(a)) {
            std::vector<int> multiplier;
            multiplier.push_back(1);

            UInt32 acc = 0;

            for (auto i = dim(a) - 1; i > 0; i--) {
                auto mult = std::accumulate(begin(multiplier),end(multiplier),        1, std::multiplies<int>());
                acc += ix.at(i)*mult;
                multiplier.push_back(multiplier.back()+1);
            }

            a._set(acc, val);
        }

        else {
            std::cout << "Total index required by set" << std::endl;
        }

    }

    inline UInt32 get_shape_elem(Array a, const UInt32 i) {
        return a._get_shape_elem(i);
    }

    inline UInt32 dim(Array a) {
        return a._dim();
    }

    inline Shape shape(Array a) {
        return a._shape();
    }

    inline UInt32 total(Array a) {
        return a._total();
    }

    /*
    array/shape/index creation util
    */

    inline Array create_array(const Shape &sh) {
        auto arr = Array(sh);
        return arr;
    }

    inline Shape create_shape1(const UInt32 a) {
        Shape sh;
        sh.push_back(a);
        return sh;
    }

    inline Shape create_shape2(const UInt32 a, const UInt32 b) {
        Shape sh;
        sh.push_back(a);
        sh.push_back(b);
        return sh;
    }
    inline Shape create_shape3(const UInt32 a, const UInt32 b, const UInt32 c) {
        Shape sh;
        sh.push_back(a);
        sh.push_back(b);
        sh.push_back(c);
        return sh;
    }


    inline Index create_index1(const UInt32 a) {
        Index ix;
        ix.push_back(a);
        return ix;
    }

    inline Index create_index2(const UInt32 a, const UInt32 b) {
        Index ix;
        ix.push_back(a);
        ix.push_back(b);
        return ix;
    }

    inline Index create_index3(const UInt32 a, const UInt32 b, const UInt32 c) {
        Index ix;
        ix.push_back(a);
        ix.push_back(b);
        ix.push_back(c);

        return ix;
    }

    /*
    Wrappers/unwrappers/util
    */

    inline Element unwrap_scalar(Array a) {
        return a._get(0);
    }

    /*
    Test arrays
    */

   inline Array test_array3_3() {

       Array a = Array(create_shape2(3,3));

       std::vector<int> v = {6,8,3,8,1,1,4,3,2};

       for (auto i = 0; i < a._total(); i++) {
           a._set(i, v.at(i));
       }

       return a;
   }

   inline Array test_array3_2_2() {

       Array a = Array(create_shape3(3,2,2));

       std::vector<int> v = {2,5,7,8,6,1,1,3,5,6,3,5};

       for (auto i = 0; i < a._total(); i++) {
           a._set(i, v.at(i));
       }

       return a;
   }

    inline Index test_index() {
        Index ix;
        ix.push_back(1);
        ix.push_back(0);
        ix.push_back(0);

        return ix;
    }

    /*
    IO functions
    */

    inline void print_array(Array a) {

        for (auto i = 0; i < (int) a._total(); i++) {
            std::cout << a._get(i) << " ";
        }
        std::cout << std::endl;
    }

    inline void print_index(const Index &ix) {
        for (auto i = 0; i < ix.size(); i++) {
            std::cout << ix.at(i) << " ";
        }
        std::cout << std::endl;
    }

    inline void print_shape(const Shape &sh) {
        for (auto i = 0; i < sh.size(); i++) {
            std::cout << sh.at(i) << " ";
        }
        std::cout << std::endl;
    }
};

struct int32_utils {
    typedef int Int32;

    inline Int32 zero() {return 0; }
    inline Int32 one() {return 1;}

    inline Int32 add(const Int32 a, const Int32 b) {
        return a + b;
    }
    inline Int32 sub(const Int32 a, const Int32 b) {
        return a - b;
    }
    inline Int32 mult(const Int32 a, const Int32 b) {
        return a * b;
    }
    inline bool equals(const Int32 a, const Int32 b) {
        return a == b;
    }
    inline bool isLowerThan(const Int32 a, const Int32 b) {
        return a < b;
    }
};


struct float64_utils {
    typedef double Float64;

    inline Float64 zero() {return 0.0; }
    inline Float64 one() {return 1.0;}

    inline Float64 add(const Float64 a, const Float64 b) {
        return a + b;
    }
    inline Float64 sub(const Float64 a, const Float64 b) {
        return a - b;
    }
    inline Float64 mult(const Float64 a, const Float64 b) {
        return a * b;
    }
    inline bool equals(const Float64 a, const Float64 b) {
        return a == b;
    }
    inline bool isLowerThan(const Float64 a, const Float64 b) {
        return a < b;
    }
};