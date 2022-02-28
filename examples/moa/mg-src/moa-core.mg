package examples.moa.mg-src.moa-core
    imports examples.moa.mg-src.array-ops,
            examples.moa.mg-src.externals.while-loops;

/*
* Barebone MoA API with core operations
* @author Marius kleppe Larnøy
* @since 2022-01-11
*/

implementation MoaOps = {

    use ArrayOps;

}

signature VectorReductionSig = {

    require type Element;

    type Array;
    type Int;

    function bop(e1: Element, e2: Element): Element;

    function id(): Element;

}

implementation VectorReductionImpl = {

    use MoaOps;
    use VectorReductionSig;

    predicate reduce_cond(input: Array, res: Element, c: Int) = {
        value int_elem(c) < int_elem(total(input));
    }

    procedure reduce_body(obs input: Array, upd res: Element, upd c: Int) {

        var current_element = unwrap_scalar(get(input, c));
        res = bop(res, current_element);
        c = elem_int(int_elem(c) + one());

    }

    use WhileLoop1_2[Context1 => Array,
                     State1 => Element,
                     State2 => Int,
                     body => reduce_body,
                     cond => reduce_cond];

    function reduce(a: Array): Element = {

        var result = id();

        var counter = elem_int(zero());

        call repeat(a, result, counter);

        value result;
    }
}

implementation Ravel = {

    use MoaOps;

    predicate ravel_cond(input: Array, flat: Array, c: Int) {
        value int_elem(c) < int_elem(total(input));
    }

    procedure ravel_body(obs input: Array, upd flat: Array, upd c: Int) {

        var linear_ix = create_index1(c);

        call set(flat, linear_ix, unwrap_scalar(get(input, c)));

        c = elem_int(int_elem(c) + one());

    }

    use WhileLoop1_2[Context1 => Array,
                     State1 => Array,
                     State2 => Int,
                     body => ravel_body,
                     cond => ravel_cond,
                     repeat => rav_repeat];


    function ravel(a: Array): Array = {

        var c = elem_int(zero());

        var flat = create_array(create_shape1(total(a)));

        call rav_repeat(a, flat, c);

        value flat;
    }

}

implementation Reshape = {

    use MoaOps;

    procedure reshape_body(obs old_array: Array, upd new_array: Array, upd counter: Int) {
        call set(new_array, counter, unwrap_scalar(get(old_array, counter)));
        counter = elem_int(int_elem(counter) + one());
    }
    predicate reshape_cond(old_array: Array, new_array: Array, counter: Int) {
       value int_elem(counter) < int_elem(total(new_array));
    }

    use WhileLoop1_2[Context1 => Array,
                     State1 => Array,
                     State2 => Int,
                     body => reshape_body,
                     cond => reshape_cond,
                     repeat => reshape_repeat];

    function reshape(input_array: Array, s: Shape): Array
        guard total(shape(input_array)) == total(s) {

        var new_array = create_array(s);

        var counter = elem_int(zero());

        call reshape_repeat(input_array, new_array, counter);

        value new_array;
    }

}


implementation Catenation = {

    use Reshape;

    /*########################################
        Vector catenation, i.e. cat(v1, v2)
      ########################################
    */

    /*
    cat_vec_body performs one iteration of putting the element currently
    indexed in its correct position in the catenated vector
    */
    procedure cat_vec_body(obs v1: Array,
                           obs v2: Array,
                           upd res: Array,
                           upd counter: Int) {

        var v1_bound = int_elem(total(v1));
        var ix: Index;

        // conditional determining if we should access v1 or v2
        if int_elem(counter) < int_elem(total(v1)) then {
            ix = create_index1(counter);
            call set(res, ix, unwrap_scalar(get(v1, ix)));
        } else {
            ix = create_index1(elem_int(int_elem(counter) - v1_bound));
            var res_ix = create_index1(counter);
            call set(res, res_ix, unwrap_scalar(get(v2, ix)));
        };

        counter = elem_int(int_elem(counter) + one());
    }

    // determines upper bound for the iterator
    predicate cat_vec_cond(v1: Array, v2: Array, res: Array, counter: Int) {
        value int_elem(counter) < int_elem(total(res));
    }

    use WhileLoop2_2[Context1 => Array,
                     Context2 => Array,
                     State1 => Array,
                     State2 => Int,
                     body => cat_vec_body,
                     cond => cat_vec_cond,
                     repeat => cat_vec_repeat];


    /*
    cat_vec takes two vectors as inputs, does a shape analysis, and then calls the cat_vec_repeat procedure with a correctly shaped updatable result vector argument
    */
    function cat_vec(vector1: Array, vector2: Array): Array
        guard dim(vector1) == elem_int(one()) && dim(vector2) == elem_int(one()) {

        var res_shape = create_shape1(elem_int(
                                          int_elem(total(vector1)) +
                                          int_elem(total(vector2))));

        var res = create_array(res_shape);
        var counter = elem_int(zero());
        call cat_vec_repeat(vector1, vector2, res, counter);

        value res;

    }

    /*########################################
        Array and vector catenation, i.e. cat(a, vec)
      ########################################
    */



    /*########################################
        Array catenation, i.e. cat(a1, a2)
      ########################################
    */
    procedure cat_body(obs array1: Array,
                       obs array2: Array,
                       upd counter: Int,
                       upd res: Array) {

        var s_0 = int_elem(total(array1));

        if int_elem(counter) < s_0 then {
            call set(res, counter,
                    unwrap_scalar(get(array1,counter)));
            counter = elem_int(int_elem(counter) + one());
        } else {
            var ix = elem_int(int_elem(counter) - s_0);
            call set(res, counter,
                unwrap_scalar(get(array2, ix)));
            counter = elem_int(int_elem(counter) + one());
        };
    }

    predicate cat_cond(array1: Array, array2: Array, counter: Int, res: Array) {
        var upper_bound = int_elem(total(res));
        value int_elem(counter) < upper_bound;
    }
    use WhileLoop2_2[Context1 => Array,
                     Context2 => Array,
                     State1 => Int,
                     State2 => Array,
                     body => cat_body,
                     cond => cat_cond,
                     repeat => cat_repeat];


    /*
    cat takes two arrays as inputs, does a shape analysis, and then calls the cat_repeat procedure with a correctly shaped updatable result array argument
    */

    /*
    TODO: IN ALL LITERATURE, CAT IS DEFINED IN TERMS OF PARTIAL INDEXING, what

    */
    function cat(array1: Array, array2: Array): Array
        guard drop_shape_elem(array1, elem_int(zero())) ==
              drop_shape_elem(array2, elem_int(zero())) {

        /*
        shape of the catenated array is given by:
        (take(1, a1) + take(1, a2)) 'cat' drop(1, a1)
        */
        var take_a1 = int_elem(get_shape_elem(array1, elem_int(zero())));
        var take_a2 = int_elem(get_shape_elem(array2, elem_int(zero())));
        var drop_a1 = drop_shape_elem(array1, elem_int(zero()));

        var result_shape = cat_shape(create_shape1(elem_int(take_a1 + take_a2)), drop_a1);

        var res = create_array(result_shape);

        var counter = elem_int(zero());
        call cat_repeat(array1, array2, counter, res);

        value res;
    }
}

implementation Padding = {

    use Catenation;
    /*
    circular padr and padl definition.
    overloaded to both accept unpadded and padded arrays as arguments,
    this is to allow composition.
    */
    function circular_padr(a: PaddedArray, ix: Int): PaddedArray = {

        // store unpadded shape
        var unpadded_shape = shape(a);

        // extract slice from a
        var padding = get(a, create_index1(ix));

        // "conform" shape of the padding to match a
        var reshape_shape = cat_shape(create_shape1(elem_int(one())), shape(padding));
        var reshaped_padding = reshape(padding, reshape_shape);

        // cat the slice to a
        var catenated_array = cat(padded_to_unpadded(a), reshaped_padding);

        var padded_shape = shape(catenated_array);

        var res = create_padded_array(unpadded_shape, padded_shape, catenated_array);

        value res;

    }


    function circular_padr(a: Array, ix: Int): PaddedArray = {

        var padding = get(a, create_index1(ix));
        var reshape_shape = cat_shape(create_shape1(elem_int(one())), shape(padding));
        var reshaped_padding = reshape(padding, reshape_shape);

        var catenated_array = cat(a, reshaped_padding);
        var unpadded_shape = shape(a);
        var padded_shape = shape(catenated_array);

        var res = create_padded_array(unpadded_shape, padded_shape, catenated_array);

        value res;
    }

    function circular_padl(a: PaddedArray, ix: Int): PaddedArray = {
        var padding = get(a, create_index1(ix));

        var reshape_shape = cat_shape(create_shape1(elem_int(one())), shape(padding));
        var reshaped_padding = reshape(padding, reshape_shape);

        var catenated_array = cat(reshaped_padding, padded_to_unpadded(a));

        var unpadded_shape = shape(a);
        var padded_shape = shape(catenated_array);

        var res = create_padded_array(unpadded_shape, padded_shape, catenated_array);

        value res;
    }

    function circular_padl(a: Array, ix: Int): PaddedArray = {
        var padding = get(a, create_index1(ix));

        var reshape_shape = cat_shape(create_shape1(elem_int(one())), shape(padding));
        var reshaped_padding = reshape(padding, reshape_shape);

        var catenated_array = cat(reshaped_padding, a);

        var unpadded_shape = shape(a);
        var padded_shape = shape(catenated_array);

        var res = create_padded_array(unpadded_shape, padded_shape, catenated_array);

        value res;
    }

}

implementation TakeDrop = {

    use Padding;

    predicate take_cond(a: Array, i: Int, res: Array, c: Int) {

        value int_elem(c) < int_elem(i);

    }

    procedure take_body(obs a: Array, obs i: Int, upd res: Array, upd c: Int) {

        var ix = create_index1(c);
        var slice = get(a, ix);

        slice = reshape(slice, cat_shape(create_shape1(elem_int(one())),
                                          shape(slice)));

        res = cat(res, slice);

        c = elem_int(int_elem(c) + one());

    }

    use WhileLoop2_2[Context1 => Array,
                     Context2 => Int,
                     State1 => Array,
                     State2 => Int,
                     body => take_body,
                     cond => take_cond,
                     repeat => take_repeat];

    function take(i: Int, a: Array): Array = {

        var drop_sh_0 = drop_shape_elem(a, elem_int(zero()));

        var res_array: Array;

        if int_elem(i) < zero() then {

            var drop_value = elem_int(int_elem(get_shape_elem(a, elem_int(zero()))) + int_elem(i));
            res_array = drop(drop_value, a);

        } else {

            res_array = get(a, create_index1(elem_int(zero())));

            res_array = reshape(res_array, cat_shape(create_shape1(elem_int(one())), shape(res_array)));
            var c = elem_int(one());
            call take_repeat(a, i, res_array, c);

        };

        value res_array;

    }

    predicate drop_cond(a: Array, i: Int, res: Array, c: Int) {

        value int_elem(c) < int_elem(get_shape_elem(a, elem_int(zero())));

    }

    procedure drop_body(obs a: Array, obs i: Int, upd res: Array, upd c: Int) {

        var slice = get(a, create_index1(c));
        slice = reshape(slice, cat_shape(create_shape1(elem_int(one())),
                                          shape(slice)));

        res = cat(res, slice);

        c = elem_int(int_elem(c) + one());
    }

    use WhileLoop2_2[Context1 => Array,
                     Context2 => Int,
                     State1 => Array,
                     State2 => Int,
                     body => drop_body,
                     cond => drop_cond,
                     repeat => drop_repeat];

    function drop(i: Int, a: Array): Array {

        var drop_sh_0 = drop_shape_elem(a, elem_int(zero()));

        var res_array: Array;

        if int_elem(i) < zero() then {

            var take_value = elem_int(int_elem(get_shape_elem(a,elem_int(zero()))) + int_elem(i));

            res_array = take(take_value, a);

        } else {

            res_array = get(a, create_index1(i));

            res_array = reshape(res_array, cat_shape(create_shape1(elem_int(one())), shape(res_array)));
            var c = elem_int(int_elem(i) + one());
            call drop_repeat(a, i, res_array, c);

        };

        value res_array;
    }
}

implementation Transformations = {

    use TakeDrop;

    /*
    #######
    rotate
    #######
    */

    function rotate(sigma: Int, ax: Int, a: Array): Array
        guard int_elem(ax) < int_elem(dim(a)) = {

        // create partial indices of length j
        var ix_space = create_partial_indices(a, ax);

        var c = elem_int(zero());
        var i = get_index_ixc(ix_space, c);

        var res: Array;

        if zero() < int_elem(sigma) then {

            var e1 = drop(sigma, get(a, i));
            var e2 = take(sigma, get(a, i));

            res = cat(e1,e2);

        } else {

            var e1 = take(elem_int(-int_elem(sigma)), get(a, i));
            var e2 = drop(elem_int(-int_elem(sigma)), get(a, i));
            call print_array(e1);
            call print_array(e2);
            res = cat(e1,e2);

        };

        value res;
    }


    /*
    #######
    Unpadded reverse
    #######
    */
    procedure reverse_body(obs input: Array, obs indices: IndexContainer, upd res: Array, upd c: Int) = {

        var ix = get_index_ixc(indices, c);
        var elem = unwrap_scalar(get(input, ix));

        var sh_0 = get_shape_elem(input, elem_int(zero()));
        var ix_0 = get_index_elem(ix, elem_int(zero()));
        var new_ix_0 = int_elem(sh_0) - (int_elem(ix_0) + one());

        var new_ix = cat_index(create_index1(elem_int(new_ix_0)), drop_index_elem(ix, elem_int(zero())));

        call set(res, new_ix, elem);

        c = elem_int(int_elem(c) + one());
    }

    predicate reverse_cond(input: Array, indices: IndexContainer, res: Array, c: Int) = {

        value int_elem(c) < int_elem(total(indices));
    }

    use WhileLoop2_2[Context1 => Array,
                     Context2 => IndexContainer,
                     State1 => Array,
                     State2 => Int,
                     body => reverse_body,
                     cond => reverse_cond,
                     repeat => reverse_repeat];

    function reverse(a: Array): Array = {

        var res_array = create_array(shape(a));

        var valid_indices = create_total_indices(res_array);
        var counter = elem_int(zero());

        call reverse_repeat(a, valid_indices, res_array, counter);

        value res_array;
    }
    /*
    #####################
    Unpadded transpose
    #####################
    */
    predicate upper_bound(a: Array, i: IndexContainer, res: Array, c: Int) = {
        value int_elem(c) < int_elem(total(i));
    }

    procedure transpose_body(obs a: Array,
                             obs ixc: IndexContainer,
                             upd res: Array,
                             upd c: Int) {

        var current_ix = get_index_ixc(ixc, c);

        var current_element = unwrap_scalar(get(a, reverse_index(current_ix)));
        call set(res, current_ix, current_element);
        c = elem_int(int_elem(c) + one());
    }

     use WhileLoop2_2[Context1 => Array,
                     Context2 => IndexContainer,
                     State1 => Array,
                     State2 => Int,
                     body => transpose_body,
                     cond => upper_bound,
                     repeat => transpose_repeat];

    function transpose(a: Array): Array = {

        var transposed_array = create_array(reverse_shape(shape(a)));

        var ix_space = create_total_indices(transposed_array);
        var counter = elem_int(zero());
        call transpose_repeat(a, ix_space, transposed_array, counter);

        value transposed_array;
    }

    /*
    #####################
    Padded transpose
    #####################
    */

    predicate padded_upper_bound(a: PaddedArray, i: IndexContainer, res: PaddedArray, c: Int) = {
        value int_elem(c) < int_elem(total(i));
    }

    procedure padded_transpose_body(obs a: PaddedArray,
                                    obs ixc: IndexContainer,
                                    upd res: PaddedArray,
                                    upd c: Int) {

        var current_ix = get_index_ixc(ixc, c);

        var current_element = unwrap_scalar(get(a, reverse_index(current_ix)));
        call set(res, current_ix, current_element);
        c = elem_int(int_elem(c) + one());
    }

    use WhileLoop2_2[Context1 => PaddedArray,
                     Context2 => IndexContainer,
                     State1 => PaddedArray,
                     State2 => Int,
                     body => padded_transpose_body,
                     cond => padded_upper_bound,
                     repeat => padded_transpose_repeat];

    function transpose(a: PaddedArray): PaddedArray = {

        var reshaped_array = create_array(padded_shape(a));
        var transposed_array = create_padded_array(
            reverse_shape(shape(a)),                                         reverse_shape(padded_shape(a)), reshaped_array);

        var ix_space = create_total_indices(transposed_array);
        var counter = elem_int(zero());
        call padded_transpose_repeat(a, ix_space, transposed_array, counter);

        value transposed_array;
    }

}

concept BMap = {

    require type Element;

    type Array;

    function bop(e1: Element, e2: Element): Element;
    function bopmap(e: Element, a: Array): Array;
}

implementation BMapVectorImpl = {

    use BMap;
    use MoaOps;



    procedure bmapvector_body(obs e: Element,
                              upd v: Array, upd c: Int) {

        var new_value = bop(e, unwrap_scalar(get(v, c)));

        call set(v, c, new_value);

        c = elem_int(int_elem(c) + one());
    }

    predicate bmapvector_cond(e: Element, v: Array, c: Int) {
        value int_elem(c) < int_elem(total(v));
    }

    use WhileLoop1_2[Context1 => Element,
                     State1 => Array,
                     State2 => Int,
                     body => bmapvector_body,
                     cond => bmapvector_cond,
                     repeat => bmapvector_repeat];

    procedure bopmap_vec(obs e: Element, upd a: Array) = {
        var counter = elem_int(zero());
        call bmapvector_repeat(e, a, counter);
    }
}