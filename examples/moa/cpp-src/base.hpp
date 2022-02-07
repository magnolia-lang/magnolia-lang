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
            Shape _shape;

        public:

            Array(Shape shape) {
                _shape = shape;

                using std::begin;
                using std::end;

                auto array_size = std::accumulate(begin(_shape),
                                                  end(_shape), 1,
                                                  std::multiplies<UInt32>());

                _content = (Element *)calloc(
                    array_size, sizeof(Element));
            }

            inline UInt32 _dim() {
                return _shape.size();
            }

            inline Element _get(const UInt32 ix) {
                return _content[ix];
            }

            inline UInt32 _get_shape_elem(const UInt32 i) {
                return _shape.at(i);
            }

            inline UInt32 _total() {
                using std::begin;
                using std::end;

                return std::accumulate(begin(_shape),
                                                  end(_shape), 1,
                                                  std::multiplies<UInt32>());
            }
    };


    inline Array get(Array a, Index ix) {
        UInt32 accum;

        // total
        if (ix.size() == dim(a)) {

            UInt32 acc;

            for (auto i = dim(a); i > 0; i--) {
                acc +=
            }
        }
         // partial
        else if (ix.size() < dim(a)) {

        }

        // invalid index
        else {
            std::cout << "Invalid index, out of bounds" << std::endl;
        }
    }

    inline UInt32 get_shape_elem(Array a, const UInt32 i) {
        return a._get_shape_elem(i);
    }

    inline Array create_array(const Shape &sh) {
        auto arr = Array(sh);
        return arr;
    }

    inline Shape create_shape_3(const UInt32 a, const UInt32 b, const UInt32 c) {
        Shape sh;

        sh.push_back(a);
        sh.push_back(b);
        sh.push_back(c);

        return sh;
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