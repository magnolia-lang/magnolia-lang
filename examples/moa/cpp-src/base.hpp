#include <string>
#include <iostream>
#include <cstdlib>
#include <vector>

template <typename _Integer>
struct matrix {
    typedef _Integer Integer;
    typedef size_t Size;
    typedef std::vector<Size> Index;
    typedef std::vector<Size> Shape;

    struct Matrix {
        private:
            Integer * content;
            Shape shape;

        public:
            Size _row, _col;

            Matrix(Size row, Size col) {
                _row = row;
                _col = col;

                if (_row == 1) {
                    shape.push_back(_col);
                }
                else {
                    shape.push_back(_row);
                    shape.push_back(_col);
                }


                content = (Integer *)calloc(row*col,sizeof(Integer));
            }

            inline void _set(size_t i, size_t j, Integer &val) {
                content[i*_col + j] = val;
            }
            inline Integer _get(size_t i, size_t j) {
                return content[i*_col + j];
            }

            inline Shape _shape() {
                return shape;
            }
            inline size_t _access_shape(size_t i) {
                return shape.at(i);
            }
            inline Size _size() {
                return _row * _col;
            }
    };

    /*
    Getters and setters
    */
    inline void set(Matrix m, const size_t i,const size_t j, Integer val) {
        m._set(i,j,val);
    }

    inline matrix::Matrix get(Matrix m, Index index) {
        if (index.size() > shape(m).size() || index.size() == 0) {
            std::cout << "Index has to be of length 1-shape(m).size()";
        }


        else if (shape(m).size() == 1){
            auto res = zeros(1,1);
            set(res, 0, 0, m._get(0, index.at(0)));
            return res;
        }
        // TODO generalize
        else if (index.size() == shape(m).size()){
            auto res = zeros(1,1);
            set(res, 0, 0, m._get(index.at(0), index.at(1)));
            return res;
        }

        else {
            auto res = zeros(1, shape(m).at(1));
            for (auto i = 0; i < shape(res).at(0); i++) {
                set(res, 0, i, m._get(index.at(0), i));
            }
            return res;
        }
    }

    inline Shape shape(Matrix m) {
        return m._shape();
    }
    inline size_t access_shape(Matrix m, size_t i) {
                return m._access_shape(i);
            }

    inline Size size(Matrix m) {
        return m._size();
    }

    inline Size size(Shape s) {
        return s.size();
    }

    inline matrix::Matrix transpose(Matrix m) {

        if (shape(m).size() < 2) {
            std::cout << "size has to be > 1" << std::endl;
        }
        else {
            auto res = zeros(access_shape(m,1), access_shape(m,0));

            for (auto i = 0; i < shape(m).at(0); i++) {
                for (auto j = 0; j < shape(m).at(1); j++) {
                    set(res,j,i,m._get(i,j));
                }
            }
            return res;
        }



    }

    inline matrix::Matrix zeros(const Size row, const Size col) {
        return Matrix(row,col);
    }
    inline matrix::Matrix create_matrix(const Size row, const Size col) {
        auto m = Matrix(row,col);
        std::cout << "Enter elements:" << std::endl;
        for (auto i = 0; i < row; i++) {
            for (auto j = 0; j < col; j++) {
                Integer num;
                std::cin >> num;
                set(m,i,j,num);
            }
        }
        return m;
    }

    inline matrix::Index single_I(const size_t i) {
        Index ix;
        ix.push_back(i);
        return ix;
    }

    inline matrix::Index create_index(const size_t i, const size_t j) {
        Index ix;
        ix.push_back(i);
        ix.push_back(j);
        return ix;
    }


    inline Integer unwrap_scalar(Matrix m) {
        return m._get(0,0);
    }

    inline Integer sizeToInteger(const Size s) {
        return (Integer) s;
    }

    inline Size integerToSize(const Integer i) {
        return (Size) i;
    }

    /*
    IO print functions
    */

    inline void print_index(Index i) {
        for (auto a = 0; a < i.size(); a++) {
            std::cout << i.at(a) << " ";
        }
        std::cout << std::endl;
    }
    inline void print_number(Integer a) {
        std::cout << a << std::endl;
    }

    inline void print_matrix(Matrix a) {
        for (auto i = 0; i < a._row; i++) {
            for (auto j = 0; j < a._col; j++) {
                std::cout << a._get(i,j) << " ";
            }
            std::cout << std::endl;
        }
    }

    inline void print_shape(Matrix a) {
        Shape sh = shape(a);
        for (auto i = 0; i < sh.size(); i++) {
            std::cout << sh.at(i) << " ";
        }
        std::cout << std::endl;
    }

    /*
    Test indices and patrices
    */

    inline matrix::Index test_partial_index() {
        Index ix;
        ix.push_back(0);
        return ix;
    }

    inline matrix::Index test_total_index() {
        Index ix;
        ix.push_back(1);
        ix.push_back(2);
        return ix;
    }

    inline matrix::Matrix test_vector() {
        auto m = Matrix(1,5);
        size_t elems[5] = {4,3,2,6,1};
        for (auto i = 0; i < 5; i++) {
            set(m,0,i,elems[i]);
        }
        return m;
    }
    inline matrix::Matrix test_matrix() {
        auto m = Matrix(5,5);
        size_t elems[25] = {1,4,2,3,5,3,7,5,9,8,2,4,1,7,8,4,2,1,9,7,5,5,5,3,1};
        auto counter = 0;
        for (auto i = 0; i < 5; i++) {
            for (auto j = 0; j < 5; j++) {

                set(m,i,j,elems[counter++]);
            }
        }
        return m;
    }

    inline matrix::Matrix test_matrix2() {
        auto m = Matrix(5,3);
        size_t elems[25] = {6,8,5,3,3,3,3,1,4,1,3,4,5,8,0};
        auto counter = 0;
        for (auto i = 0; i < 5; i++) {
            for (auto j = 0; j < 3; j++) {

                set(m,i,j,elems[counter++]);
            }
        }
        return m;
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

template <typename _Context1, typename _Context2, typename _State1, typename _State2, typename _State3, class _body, class _cond>
struct while_loop2_3 {
	typedef _Context1 Context1;
	typedef _Context2 Context2;
	typedef _State1 State1;
	typedef _State2 State2;
	typedef _State3 State3;

	_body body;
	_cond cond;
	void repeat(const Context1 &context1, const Context2 &context2, State1 &state1, State2 &state2, State3 &state3) {
		while (cond(context1, context2, state1, state2, state3)) body(context1, context2, state1, state2, state3);
	}
};


template <typename _Context1, typename _Context2, typename _State1, typename _State2, class _body, class _cond>
struct while_loop2_2 {
	typedef _Context1 Context1;
	typedef _Context2 Context2;
	typedef _State1 State1;
	typedef _State2 State2;

	_body body;
	_cond cond;
	void repeat(const Context1 &context1, const Context2 &context2, State1 &state1, State2 &state2) {
		while (cond(context1, context2, state1, state2)) body(context1, context2, state1, state2);
	}
};

template <typename _Context1, typename _State1, typename _State2, class _body, class _cond>
struct while_loop1_2 {
	typedef _Context1 Context1;
	typedef _State1 State1;
	typedef _State2 State2;

	_body body;
	_cond cond;
	void repeat(const Context1 &context1, State1 &state1, State2 &state2) {
		while (cond(context1, state1, state2)) body(context1, state1, state2);
	}
};
