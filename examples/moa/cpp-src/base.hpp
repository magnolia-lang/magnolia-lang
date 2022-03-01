#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

template <typename _Element>
struct array {

    typedef _Element Element;
    typedef int Int;
    typedef float Float;
    typedef std::vector<Int> Index;
    typedef std::vector<Index> IndexContainer;
    typedef std::vector<Int> Shape;

    struct Array {

        public:

            Shape _sh;
            Element * _content;

            // empty constructor needed when instantiating PaddedArray
            Array() {
                _sh.push_back(1);
                _content = (Element *) calloc(1, sizeof(Element));
            }

            Array(Shape shape) {
                _sh = shape;

                using std::begin;
                using std::end;

                auto array_size = std::accumulate(begin(_sh),
                                                  end(_sh), 1,
                                                  std::multiplies<Int>());

                _content = (Element *)calloc(
                    array_size, sizeof(Element));
            }

            inline Int _dim() {
                return _sh.size();
            }

            inline Shape _shape() {
                return _sh;
            }

            inline Element _get(const Int ix) {
                return _content[ix];
            }

            inline void _set(const Int ix, Element val) {
                _content[ix] = val;
            }

            inline Int _get_shape_elem(const Int i) {
                return _sh.at(i);
            }

            inline Int _total() {
                using std::begin;
                using std::end;

                return std::accumulate(begin(_sh),
                                                  end(_sh), 1,
                                                  std::multiplies<Int>());
            }
    };

    struct PaddedArray : public Array {

        public:

            Shape _padded_sh;

            PaddedArray(const Shape unpadded_sh, const Shape padded_sh, const Array padded) {

                _padded_sh = padded_sh;

                this -> _sh = unpadded_sh;
                this -> _content = padded._content;
            }

            inline Int _padded_dim() {
                return _padded_sh.size();
            }

            inline Shape _padded_shape() {
                return _padded_sh;
            }

            inline Element _padded_get(const Int ix) {
                return this -> _content[ix];
            }

            inline void _padded_set(const Int ix, Element val) {
                this -> _content[ix] = val;
            }

            inline Int _padded_get_shape_elem(const Int i) {
                return _padded_sh.at(i);
            }

            inline Int _padded_total() {
                using std::begin;
                using std::end;

                return std::accumulate(begin(_padded_sh),
                                                  end(_padded_sh), 1,
                                                  std::multiplies<Int>());
            }

    };

    /*
    Operations on unpadded arrays
    */

    inline Array get(Array a, Int ix) {
        if (ix > total(a)) {
            std::cout << "get:Index out of bounds: " << ix << std::endl;
        }
        else {
            Shape res_shape = create_shape1(1);
            Array res = Array(res_shape);

            res._set(0,a._get(ix));
            return res;
        }
    }
    inline Array get(PaddedArray a, Int ix) {
        if (ix > padded_total(a)) {
            std::cout << "get:Index out of bounds: " << ix << std::endl;
        }
        else {
            Shape res_shape = create_shape1(1);
            Array res = Array(res_shape);

            res._set(0,a._get(ix));
            return res;
        }
    }

    inline void set(Array a, Int ix, const Element e) {
        if (ix > total(a)) {
            std::cout << "set:Index out of bounds: " << ix << std::endl;
        }
        else {
            a._set(ix,e);
        }
    }

    inline void set(PaddedArray a, Int ix, const Element e) {
        if (ix > padded_total(a)) {
            std::cout << "set:Index out of bounds: " << ix << std::endl;
        }
        else {
            a._set(ix,e);
        }
    }

    inline Array get(Array a, Index ix) {

        // total
        if (ix.size() == dim(a)) {

            Shape sh = shape(a);

            Int accum = 0;

            for (int i = 0; i < dim(a); i++) {

                if(ix.at(i) > sh.at(i)) {
                    std::cout << "Access out of bounds: " << ix.at(i)
                              << ". Max is : " << sh.at(i) << std::endl;
                }

                std::vector<Int>::const_iterator first = sh.begin() + (i+1);
                std::vector<Int>::const_iterator last = sh.begin() + dim(a);

                std::vector<Int> subshape(first, last);

                int reduced = std::accumulate(begin(subshape),
                                                  end(subshape), 1,
                                                  std::multiplies<Int>());

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
            std::vector<Int>::const_iterator first = sh.begin() + (ix.size());
            std::vector<Int>::const_iterator last = sh.begin() + dim(a);
            std::vector<Int> subshape(first, last);

            Array sub_array = Array(subshape);

            Int total_elems = total(sub_array);

            std::vector<Index> indices;

            // populates a vector with empty total indices
            for (auto i = 0; i < total(sub_array); i++) {
                std::vector<Int> total;

                for (auto j = 0; j < ix.size(); j++) {
                    total.push_back(ix.at(j));
                }

                indices.push_back(total);
            }

            // generates all total indices
            for (auto i = 0; i < subshape.size(); i++) {
                Int current_sh_ix = subshape.at(i);
                Int split = (i == subshape.size() - 1) ? 1 : subshape.at(i+1);
                Int current_ix = 0;
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

            // special cases for dim(a) == 1, 2
            if (dim(a) == 1) {
                a._set(ix.at(0), val);
            }
            else if (dim(a) == 2) {
                auto i = ix.at(0);
                auto j = ix.at(1);

                a._set(i*shape(a).at(1) + j, val);
            }

            else {
                std::vector<int> multiplier;
                multiplier.push_back(1);
                Int acc = 0;

                Shape sh = shape(a);

                for (int i = 0; i < dim(a); i++) {

                if(ix.at(i) > sh.at(i)) {
                    std::cout << "Access out of bounds: " << ix.at(i)
                              << ". Max is : " << sh.at(i) << std::endl;
                }

                std::vector<Int>::const_iterator first = sh.begin() + (i+1);
                std::vector<Int>::const_iterator last = sh.begin() + dim(a);

                std::vector<Int> subshape(first, last);

                int reduced = std::accumulate(begin(subshape),
                                                  end(subshape), 1,
                                                  std::multiplies<Int>());

                acc += ix.at(i) * reduced;
            }

                a._set(acc, val);
            }


        }

        else {
            std::cout << "Total index required by set" << std::endl;
        }
    }

    inline Int dim(Array a) {
        return a._dim();
    }

    inline Shape shape(Array a) {
        return a._shape();
    }

    inline Int total(Array a) {
        return a._total();
    }

    inline Int total(Shape s) {
        return s.size();
    }

    inline Int total(IndexContainer ix) {
        return ix.size();
    }

    /*
    operations on padded arrays
    */
    inline Int padded_dim(PaddedArray a) {
        return a._padded_dim();
    }
    inline Shape padded_shape(PaddedArray a) {
        return a._padded_shape();
    }

    inline Int padded_total(PaddedArray a) {
        return a._padded_total();
    }

    /*
    array/shape/index creation util
    */

    inline Array create_array(const Shape &sh) {
        auto arr = Array(sh);
        return arr;
    }

    inline PaddedArray create_padded_array(Shape unpadded_shape, Shape padded_shape, Array padded_array) {
        return PaddedArray(unpadded_shape, padded_shape, padded_array);
    }

    inline Shape create_shape1(const Int a) {
        Shape sh;
        sh.push_back(a);
        return sh;
    }

    inline Shape create_shape2(const Int a, const Int b) {
        Shape sh;
        sh.push_back(a);
        sh.push_back(b);
        return sh;
    }
    inline Shape create_shape3(const Int a, const Int b, const Int c) {
        Shape sh;
        sh.push_back(a);
        sh.push_back(b);
        sh.push_back(c);
        return sh;
    }

    static bool compare_indices(Index a, Index b) {

        for (auto i = 0; i < a.size(); i++) {

            if (a.at(i) < b.at(i)) {
                return true;
            }
            else if (a.at(i) == b.at(i))
                continue;

            else
                return false;
        }

        return false;

    }
    inline IndexContainer create_total_indices(Array a) {

        auto sh = shape(a);
        Int total_elems = total(a);

        IndexContainer indices;


        // populates a vector with empty total indices
        for (auto i = 0; i < total_elems; i++) {
            std::vector<Int> total;

            indices.push_back(total);
        }

        for (auto i = 0; i < sh.size(); i++) {
                Int current_sh_ix = sh.at(i);
                Int split = (i == sh.size() - 1) ? 1 : sh.at(i+1);
                Int current_ix = 0;
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
            }

        std::sort(indices.begin(), indices.end(), compare_indices);
        return indices;
    }

    inline IndexContainer create_total_indices(PaddedArray a) {

        auto sh = padded_shape(a);
        Int total_elems = padded_total(a);

        IndexContainer indices;


        // populates a vector with empty total indices
        for (auto i = 0; i < total_elems; i++) {
            std::vector<Int> total;

            indices.push_back(total);
        }

        for (auto i = 0; i < sh.size(); i++) {
                Int current_sh_ix = sh.at(i);
                Int split = (i == sh.size() - 1) ? 1 : sh.at(i+1);
                Int current_ix = 0;
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
            }

        return indices;
    }

    inline IndexContainer create_partial_indices(Array a, Int level) {

        if (level > total(shape(a))-1) {
            std::cout << "create_partial_indices: Level can't be higher than number of dimensions" << std:: endl;
        } else {

            IndexContainer total_indices = create_total_indices(a);
            IndexContainer partial_indices;

            for (Index ix : total_indices) {

                auto size = total(ix);

                while(size > level) {
                    ix.pop_back();
                    size = total(ix);
                }

                partial_indices.push_back(ix);
            }

            // remove duplicates
            auto last = std::unique(partial_indices.begin(), partial_indices.end());
            partial_indices.erase(last, partial_indices.end());

            return partial_indices;

        }

    }

    inline IndexContainer create_partial_indices(PaddedArray a, Int level) {

        if (level > total(padded_shape(a))-1) {
            std::cout << "Level can't be higher than number of dimensions" << std:: endl;
        } else {

            IndexContainer total_indices = create_total_indices(a);
            IndexContainer partial_indices;

            for (Index ix : total_indices) {

                auto size = total(ix);

                while(size > level) {
                    ix.pop_back();
                    size = total(ix);
                }

                partial_indices.push_back(ix);
            }

            // remove duplicates
            auto last = std::unique(partial_indices.begin(), partial_indices.end());
            partial_indices.erase(last, partial_indices.end());

            return partial_indices;

        }

    }

    inline Index create_index1(const Int a) {
        Index ix;
        ix.push_back(a);
        return ix;
    }

    inline Index create_index2(const Int a, const Int b) {
        Index ix;
        ix.push_back(a);
        ix.push_back(b);
        return ix;
    }

    inline Index create_index3(const Int a, const Int b, const Int c) {
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

    inline Element int_elem(const Int a) {
        return (Element) a;
    }

    inline Int elem_int(const Element a) {
        return (Int) a;
    }
    inline Element float_elem(const Float a) {
        return (Element) a;
    }
    inline Float elem_float(const Element a) {
        return (Float) a;
    }

    inline Int get_shape_elem(Array a, const Int i) {
        return a._get_shape_elem(i);
    }

    inline Int get_index_elem(Index ix, const Int i) {
        return ix.at(i);
    }

    inline Index get_index_ixc(const IndexContainer ixc, const Int ix) {
        return ixc.at(ix);
    }

    inline Index drop_index_elem(Index ix, const Int i) {
        Index res;

        for (auto j = 0; j < ix.size(); j++) {
            if (i == j) continue;
            res.push_back(ix.at(j));
        }

        return res;
    }

    inline Shape drop_shape_elem(Array a, const Int i) {
        Shape res;
        for (auto j = 0; j < shape(a).size(); j++) {
            if (i == j) continue;
            res.push_back(shape(a).at(j));
        }
        return res;
    }

    inline Int padded_get_shape_elem(PaddedArray a, const Int i) {
        return a._padded_get_shape_elem(i);
    }

    inline Shape padded_drop_shape_elem(PaddedArray a, const Int i) {
        Shape res;
        for (auto j = 0; j < padded_shape(a).size(); j++) {
            if (i == j) continue;
            res.push_back(shape(a).at(j));
        }
        return res;
    }

    inline Index cat_index(Index a, Index b) {
        Index res;

        for(const auto &ix : a)
            res.push_back(ix);

        for (const auto &ix: b)
            res.push_back(ix);

        return res;
    }

    inline Shape cat_shape(Shape a, Shape b) {
        Shape res;

        for (const auto &i : a) {
            res.push_back(i);
        }
        for (const auto &i: b) {
            res.push_back(i);
        }

        return res;
    }

    inline Index reverse_index(Index ix) {
        Index res = ix;
        reverse(res.begin(), res.end());
        return res;
    }

    inline Shape reverse_shape(Shape s) {
        Shape res = s;
        reverse(res.begin(), res.end());
        return res;
    }

    inline Array padded_to_unpadded(const PaddedArray a) {
        Array res = create_array(padded_shape(a));

        for (auto i = 0; i < padded_total(a); i++) {
            set(res, i, unwrap_scalar(get(a,i)));
        }

        return res;
    }

    /*
    Test arrays
    */

    inline Array test_vector2() {

       Array a = Array(create_shape1(2));

       std::vector<int> v = {1,2};

       for (auto i = 0; i < a._total(); i++) {
           a._set(i, v.at(i));
       }

       return a;
   }

    inline Array test_vector3() {

       Array a = Array(create_shape1(3));

       std::vector<int> v = {7,7,7};

       for (auto i = 0; i < a._total(); i++) {
           a._set(i, v.at(i));
       }

       return a;
   }

    inline Array test_vector5() {

       Array a = Array(create_shape1(5));

       std::vector<int> v = {1,2,3,4,5};

       for (auto i = 0; i < a._total(); i++) {
           a._set(i, v.at(i));
       }

       return a;
   }



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

   inline Array test_array3_2_2F() {

       Array a = Array(create_shape3(3,2,2));

       std::vector<double> v = {3.0,5.0,7.0,8.0,5.0,1.0,1.0,3.0,5.0,6.0,3.0,5.0};

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
    IO
    */

    inline void print_array(Array a) {

        for (auto i = 0; i < (int) a._total(); i++) {
            std::cout << a._get(i) << " ";
        }
        std::cout << std::endl;

    }

    inline void print_parray(PaddedArray a) {

        for (auto i = 0; i < (int) a._padded_total(); i++) {
            std::cout << a._padded_get(i) << " ";
        }
        std::cout << std::endl;
    }

    inline void print_index(const Index &ix) {
        std::cout << "< ";
        for (auto i = 0; i < ix.size(); i++) {
            std::cout << ix.at(i) << " ";
        }
        std::cout << ">" << std::endl;
    }

    inline void print_index_container(const IndexContainer &ixc) {
        for (auto i = 0; i < ixc.size(); i++) {
            print_index(ixc.at(i));
        }
    }

    inline void print_shape(const Shape &sh) {
        std::cout << "< ";
        for (auto i = 0; i < sh.size(); i++) {
            std::cout << sh.at(i) << " ";
        }
        std::cout << ">" << std::endl;
    }
    inline void print_element(const Element &e) {
        std::cout << e << std::endl;
    }

    inline void print_int(const Int &i) {
        std::cout << i << std::endl;
    }

    /*
    internal int/float operations
    */
    inline Int zero() {return 0; }
    inline Int one() {return 1;}

    inline Int binary_add(const Int a, const Int b) {
        return a + b;
    }
    inline Int binary_sub(const Int a, const Int b) {
        return a - b;
    }
    inline Int mul(const Int a, const Int b) {
        return a * b;
    }
    inline Int div(const Int a, const Int b) {
        return a / b;
    }
    inline Int unary_sub(const Int a) {
        return -a;
    }
    inline Int abs(const Int a) {
        return std::abs(a);
    }
    inline bool lt(const Int a, const Int b) {
        return a < b;
    }
    inline bool le(const Int a, const Int b) {
        return a <= b;
    }

    inline Float zeroF() {return 0.0; }
    inline Float oneF() {return 1.0;}

    inline Float binary_add(const Float a, const Float b) {
        return a + b;
    }
    inline Float binary_sub(const Float a, const Float b) {
        return a - b;
    }
    inline Float mul(const Float a, const Float b) {
        return a * b;
    }
    inline Float div(const Float a, const Float b) {
        return a / b;
    }
    inline Float unary_sub(const Float a) {
        return -a;
    }
    inline Float abs(const Float a) {
        return std::abs(a);
    }
    inline bool lt(const Float a, const Float b) {
        return a < b;
    }
    inline bool le(const Float a, const Float b) {
        return a <= b;
    }


};

struct int64_utils {
    typedef long int Int64;

    inline Int64 zero() {return 0; }
    inline Int64 one() {return 1;}

    inline Int64 binary_add(const Int64 a, const Int64 b) {
        return a + b;
    }
    inline Int64 binary_sub(const Int64 a, const Int64 b) {
        return a - b;
    }
    inline Int64 mul(const Int64 a, const Int64 b) {
        return a * b;
    }
    inline Int64 div(const Int64 a, const Int64 b) {
        return a / b;
    }
    inline Int64 unary_sub(const Int64 a) {
        return -a;
    }
    inline Int64 abs(const Int64 a) {
        return std::abs(a);
    }
    inline bool eq(const Int64 a, const Int64 b) {
        return a == b;
    }
    inline bool lt(const Int64 a, const Int64 b) {
        return a < b;
    }
    inline bool le(const Int64 a, const Int64 b) {
        return a <= b;
    }
};


struct float64_utils {
    typedef double Float64;

    inline Float64 zero() {return 0.0; }
    inline Float64 one() {return 1.0;}

    inline Float64 binary_add(const Float64 a, const Float64 b) {
        return a + b;
    }
    inline Float64 binary_sub(const Float64 a, const Float64 b) {
        return a - b;
    }
    inline Float64 mul(const Float64 a, const Float64 b) {
        return a * b;
    }
    inline Float64 div(const Float64 a, const Float64 b) {
        return a / b;
    }
    inline Float64 unary_sub(const Float64 a) {
        return -a;
    }
    inline Float64 abs(const Float64 a) {
        return std::abs(a);
    }
    inline bool eq(const Float64 a, const Float64 b) {
        return a == b;
    }
    inline bool lt(const Float64 a, const Float64 b) {
        return a < b;
    }
    inline bool le(const Float64 a, const Float64 b) {
        return a <= b;
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

template <typename _Context1, typename _Context2, typename _Context3, typename _Context4, typename _State1, typename _State2, class _body, class _cond>
struct while_loop4_2 {
	typedef _Context1 Context1;
	typedef _Context2 Context2;
	typedef _Context3 Context3;
	typedef _Context4 Context4;
	typedef _State1 State1;
	typedef _State2 State2;

	_body body;
	_cond cond;
	void repeat(const Context1 &context1, const Context2 &context2, const Context3 &context3, const Context4 &context4, State1 &state1, State2 &state2) {
		while (cond(context1, context2, context3, context4, state1, state2)) body(context1, context2, context3, context4, state1, state2);
	}
};

template <typename _Context1, typename _Context2, typename _Context3, typename _State1, typename _State2, class _body, class _cond>
struct while_loop3_2 {
	typedef _Context1 Context1;
	typedef _Context2 Context2;
	typedef _Context3 Context3;
	typedef _State1 State1;
	typedef _State2 State2;

	_body body;
	_cond cond;
	void repeat(const Context1 &context1, const Context2 &context2, const Context3 &context3, State1 &state1, State2 &state2) {
		while (cond(context1, context2, context3, state1, state2)) body(context1, context2, context3, state1, state2);
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

template <typename _Context1, typename _State1, class _body, class _cond>
struct while_loop1_1 {
	typedef _Context1 Context1;
	typedef _State1 State1;

	_body body;
	_cond cond;
	void repeat(const Context1 &context1, State1 &state1) {
		while (cond(context1, state1)) body(context1, state1);
	}
};
