#include <string>
#include <iostream>
#include <cstdlib>
#include <vector>

template <typename _Integer>
struct matrix {
    typedef _Integer Integer;
    typedef size_t Index;
    typedef size_t Size;
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

                shape.push_back(_row);
                shape.push_back(_col);

                content = (Integer *)calloc(row*col,sizeof(Integer));
            }

            inline void _set(Index i, Index j, Integer &val) {
                content[i*_col + j] = val;
            }
            inline Integer _get(Index i, Index j) {
                return content[i*_col + j];
            }
            inline Shape _get_shape() {
                return shape;
            }
            inline size_t _access_shape(Index i) {
                return shape.at(i);
            }
            inline Size _size() {
                return _row * _col;
            }
    };

    inline void set(Matrix m, const Index i,const Index j, Integer val) {
        m._set(i,j,val);
    }

    inline Integer get(Matrix m, Index i, Index j) {
        return m._get(i,j);
    }

    inline Shape get_shape(Matrix m) {
        return m._get_shape();
    }
    inline size_t access_shape(Matrix m, Index i) {
                return m._access_shape(i);
            }

    inline Size size(Matrix m) {
        return m._size();
    }

    inline Integer indexToInteger(const Index i) {
        return (Integer) i;
    }

    inline Index integerToIndex(const Integer i) {
        return (Index) i;
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

    inline void print_number(Integer a) {
        std::cout << a << std::endl;
    }

    inline void print_matrix(Matrix a) {
        for (auto i = 0; i < a._row; i++) {
            for (auto j = 0; j < a._col; j++) {
                std::cout << get(a,i,j) << " ";
            }
            std::cout << std::endl;
        }
    }

    inline void print_shape(Matrix a) {
        Shape sh = get_shape(a);
        for (auto i = 0; i < sh.size(); i++) {
            std::cout << sh.at(i) << " ";
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


// matrix mult templates
template <typename _Context1, typename _Context2, typename _Context3, typename _Context4, typename _Context5, typename _Context6, typename _Context7, typename _State1, typename _State2, class _body, class _cond>
struct while_loop7_2 {
	typedef _Context1 Context1;
	typedef _Context2 Context2;
	typedef _Context3 Context3;
	typedef _Context4 Context4;
	typedef _Context5 Context5;
	typedef _Context6 Context6;
	typedef _Context7 Context7;
	typedef _State1 State1;
	typedef _State2 State2;

	_body body;
	_cond cond;
	void repeat(const Context1 &context1, const Context2 &context2, const Context3 &context3, const Context4 &context4, const Context5 &context5, const Context6 &context6, const Context7 &context7, State1 &state1, State2 &state2) {
		while (cond(context1, context2, context3, context4, context5, context6, context7, state1, state2)) body(context1, context2, context3, context4, context5, context6, context7, state1, state2);
	}
};


template <typename _Context1, typename _Context2, typename _Context3, typename _Context4, typename _Context5, typename _Context6, typename _State1, typename _State2, class _body, class _cond>
struct while_loop6_2 {
	typedef _Context1 Context1;
	typedef _Context2 Context2;
	typedef _Context3 Context3;
	typedef _Context4 Context4;
	typedef _Context5 Context5;
	typedef _Context6 Context6;
	typedef _State1 State1;
	typedef _State2 State2;

	_body body;
	_cond cond;
	void repeat(const Context1 &context1, const Context2 &context2, const Context3 &context3, const Context4 &context4, const Context5 &context5, const Context6 &context6, State1 &state1, State2 &state2) {
		while (cond(context1, context2, context3, context4, context5, context6, state1, state2)) body(context1, context2, context3, context4, context5, context6, state1, state2);
	}
};

template <typename _Context1, typename _Context2, typename _Context3, typename _Context4, typename _Context5, typename _State1, typename _State2, class _body, class _cond>
struct while_loop5_2 {
	typedef _Context1 Context1;
	typedef _Context2 Context2;
	typedef _Context3 Context3;
	typedef _Context4 Context4;
	typedef _Context5 Context5;
	typedef _State1 State1;
	typedef _State2 State2;

	_body body;
	_cond cond;
	void repeat(const Context1 &context1, const Context2 &context2, const Context3 &context3, const Context4 &context4, const Context5 &context5, State1 &state1, State2 &state2) {
		while (cond(context1, context2, context3, context4, context5, state1, state2)) body(context1, context2, context3, context4, context5, state1, state2);
	}
};
