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

            accum += ix.back();

            Shape res_shape = create_shape_1(1);
            Array res = Array(res_shape);

            res._set(0,a._get(accum));
            return res;

        }
         // partial
        else if (ix.size() < dim(a)) {
            std::cout << "not implemented" << std::endl;
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

    inline Array create_array(const Shape &sh) {
        auto arr = Array(sh);
        return arr;
    }

    inline Shape create_shape_1(const UInt32 a) {
        Shape sh;
        sh.push_back(a);
        return sh;
    }
    inline Shape create_shape_3(const UInt32 a, const UInt32 b, const UInt32 c) {
        Shape sh;

        sh.push_back(a);
        sh.push_back(b);
        sh.push_back(c);

        return sh;
    }

    inline Index test_index() {
        Index ix;
        ix.push_back(1);
        ix.push_back(0);
        ix.push_back(0);

        return ix;
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
    Wrappers/unwrappers
    */

    inline Element unwrap_scalar(Array a) {
        return a._get(0);
    }

    /*
    Test arrays
    */

   inline Array test_array1() {

       Array a = Array(create_shape_3(3,2,2));

       std::vector<int> v = {2,5,7,8,6,1,1,3,5,6,3,5};

       for (auto i = 0; i < a._total(); i++) {
           a._set(i, v.at(i));
       }

       return a;
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