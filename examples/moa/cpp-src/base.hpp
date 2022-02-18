#include <algorithm>
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
    typedef std::vector<Index> IndexContainer;
    typedef std::vector<UInt32> Shape;

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

    struct PaddedArray : public Array {

        public:

            Shape _padded_sh;

            PaddedArray(const Shape unpadded_sh, const Shape padded_sh, const Array padded) {

                _padded_sh = padded_sh;

                this -> _sh = unpadded_sh;
                this -> _content = padded._content;
            }

            inline UInt32 _padded_dim() {
                return _padded_sh.size();
            }

            inline Shape _padded_shape() {
                return _padded_sh;
            }

            inline Element _padded_get(const UInt32 ix) {
                return this -> _content[ix];
            }

            inline void _padded_set(const UInt32 ix, Element val) {
                this -> _content[ix] = val;
            }

            inline UInt32 _padded_get_shape_elem(const UInt32 i) {
                return _padded_sh.at(i);
            }

            inline UInt32 _padded_total() {
                using std::begin;
                using std::end;

                return std::accumulate(begin(_padded_sh),
                                                  end(_padded_sh), 1,
                                                  std::multiplies<UInt32>());
            }

    };

    /*
    Operations on unpadded arrays
    */

    inline Array get(Array a, UInt32 ix) {
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
    inline Array get(PaddedArray a, UInt32 ix) {
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

    inline void set(Array a, UInt32 ix, const Element e) {
        if (ix > total(a)) {
            std::cout << "set:Index out of bounds: " << ix << std::endl;
        }
        else {
            a._set(ix,e);
        }
    }

    inline void set(PaddedArray a, UInt32 ix, const Element e) {
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

            UInt32 accum = 0;

            for (int i = 0; i < dim(a); i++) {

                if(ix.at(i) > sh.at(i)) {
                    std::cout << "Access out of bounds: " << ix.at(i)
                              << ". Max is : " << sh.at(i) << std::endl;
                }

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

            // generates all total indices
            for (auto i = 0; i < subshape.size(); i++) {
                UInt32 current_sh_ix = subshape.at(i);
                UInt32 split = (i == subshape.size() - 1) ? 1 : subshape.at(i+1);
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

                UInt32 acc = 0;

                for (auto i = dim(a) - 1; i > 0; i--) {
                    auto mult = std::accumulate(begin(multiplier),end(multiplier),        1, std::multiplies<int>());

                    acc += ix.at(i)*mult;
                    multiplier.push_back(multiplier.back()+1);
                }

                acc += ix.back();

                a._set(acc, val);
            }


        }

        else {
            std::cout << "Total index required by set" << std::endl;
        }
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

    inline UInt32 total(Shape s) {
        return s.size();
    }

    inline UInt32 total(IndexContainer ix) {
        return ix.size();
    }

    /*
    operations on padded arrays
    */
    inline UInt32 padded_dim(PaddedArray a) {
        return a._padded_dim();
    }
    inline Shape padded_shape(PaddedArray a) {
        return a._padded_shape();
    }

    inline UInt32 padded_total(PaddedArray a) {
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

    inline IndexContainer create_valid_indices(Array a) {

        auto sh = shape(a);
        UInt32 total_elems = total(a);

        IndexContainer indices;


        // populates a vector with empty total indices
        for (auto i = 0; i < total_elems; i++) {
            std::vector<UInt32> total;

            indices.push_back(total);
        }

        for (auto i = 0; i < sh.size(); i++) {
                UInt32 current_sh_ix = sh.at(i);
                UInt32 split = (i == sh.size() - 1) ? 1 : sh.at(i+1);
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
            }

        return indices;
    }

    inline IndexContainer create_valid_indices(PaddedArray a) {

        auto sh = padded_shape(a);
        UInt32 total_elems = padded_total(a);

        IndexContainer indices;


        // populates a vector with empty total indices
        for (auto i = 0; i < total_elems; i++) {
            std::vector<UInt32> total;

            indices.push_back(total);
        }

        for (auto i = 0; i < sh.size(); i++) {
                UInt32 current_sh_ix = sh.at(i);
                UInt32 split = (i == sh.size() - 1) ? 1 : sh.at(i+1);
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
            }

        return indices;
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

    inline Element uint_elem(const UInt32 a) {
        return (Element) a;
    }

    inline UInt32 elem_uint(const Element a) {
        return (UInt32) a;
    }

    inline UInt32 get_shape_elem(Array a, const UInt32 i) {
        return a._get_shape_elem(i);
    }

    inline UInt32 get_index_elem(Index ix, const UInt32 i) {
        return ix.at(i);
    }

    inline Index get_index_ixc(const IndexContainer ixc, const UInt32 ix) {
        return ixc.at(ix);
    }

    inline Index drop_index_elem(Index ix, const UInt32 i) {
        Index res;

        for (auto j = 0; j < ix.size(); j++) {
            if (i == j) continue;
            res.push_back(ix.at(j));
        }

        return res;
    }

    inline Shape drop_shape_elem(Array a, const UInt32 i) {
        Shape res;
        for (auto j = 0; j < shape(a).size(); j++) {
            if (i == j) continue;
            res.push_back(shape(a).at(j));
        }
        return res;
    }

    inline UInt32 padded_get_shape_elem(PaddedArray a, const UInt32 i) {
        return a._padded_get_shape_elem(i);
    }

    inline Shape padded_drop_shape_elem(PaddedArray a, const UInt32 i) {
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

    inline void print_uint(const UInt32 &i) {
        std::cout << i << std::endl;
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
