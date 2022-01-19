#include <string>
#include <iostream>
#include <cstdlib>

template <typename _Element>
struct matrix {
    typedef size_t Size;
    typedef _Element Element;

    typedef std::string Result;

    struct Matrix {
        private:
            Element * content;
        public:
            size_t _row, _col;

            Matrix(size_t row, size_t col) {
                _row = row;
                _col = col;

                content = (Element *)calloc(row*col,sizeof(Element));
            }
            inline void set(size_t i, size_t j, Element &val) {
                content[i*_col + j] = val;
            }
            inline Element get(size_t i, size_t j) {
                return content[i*_col + j];
            }

    };

    inline matrix::Matrix zeros(const Size row, const Size col) {

        return Matrix(row,col);
    }
    inline matrix::Matrix create_matrix(const Size row, const Size col) {
        auto m = Matrix(row,col);
        std::cout << "Enter elements:" << std::endl;
        for (auto i = 0; i < row; i++) {
            for (auto j = 0; j < col; j++) {
                Element num;
                std::cin >> num;
                m.set(i,j, num);
            }
        }
        return m;
    }

    inline matrix::Matrix rand_matrix(const Size row, const Size col, const int upper_bound) {
        Matrix matr = zeros(row,col);
        for (auto i = 0; i < row; i++) {
            for (auto j = 0; j < col; j++) {
                auto r = rand() % upper_bound;
                matr.set(i,j,r);
            }
        }

        return matr;
    }

    inline matrix::Matrix mmult(Matrix a, Matrix b) {
        Matrix res = Matrix(a._row, b._col);
        for (auto i = 0; i < a._row; i++) {
            for (auto j = 0; j < b._col; j++) {
                for (auto k = 0; k < a._col; k++) {
                    auto new_value = res.get(i,j) + (a.get(i,k)*b.get(k,j));
                    res.set(i,j,new_value);
                }
            }
        }
        return res;
    }

    inline void printMatrix(Matrix a) {
        for (auto i = 0; i < a._row; i++) {
            for (auto j = 0; j < a._col; j++) {
                std::cout << a.get(i,j) << " ";
            }
            std::cout << std::endl;
        }
    }

};

struct int32_utils {
    typedef int Int32;

    inline Int32 zero() {return 0; }
    inline Int32 one() {return 1;}
    inline Int32 add(const Int32 a, const Int32 b) {
        return a + b;
    }
    inline Int32 mult(const Int32 a, const Int32 b) {
        return a * b;
    }
    inline bool isLowerThan(const Int32 a, const Int32 b) {
        return a < b;
    }
};