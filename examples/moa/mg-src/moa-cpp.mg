package examples.moa.mg-src.moa-cpp
    imports examples.moa.mg-src.externals.array-externals,
            examples.moa.mg-src.externals.number-types-externals,
            examples.moa.mg-src.externals.while-loops;

/*
* Barebone MoA API with core operations
* @author Marius kleppe Larnøy
* @since 2022-01-11
*/

implementation MoaOps = {

    use ExtOps;

    require function zero(): Element;
    require function one(): Element;
    require function add(a: Element, b: Element): Element;
    require function sub(a: Element, b: Element): Element;
    require function mult(a: Element, b: Element): Element;
    require predicate equals(a: Element, b: Element);
    require predicate isLowerThan(a: Element, b: Element);
}

implementation Reshape = {

    use MoaOps;

    procedure reshape_body(obs old_array: Array, upd new_array: Array, upd counter: UInt32) {
        call set(new_array, counter, unwrap_scalar(get(old_array, counter)));
        counter = elem_uint(add(uint_elem(counter), one()));
    }
    predicate reshape_cond(old_array: Array, new_array: Array, counter: UInt32) {
       value isLowerThan(uint_elem(counter), uint_elem(total(new_array)));
    }

    use WhileLoop1_2[Context1 => Array,
                     State1 => Array,
                     State2 => UInt32,
                     body => reshape_body,
                     cond => reshape_cond,
                     repeat => reshape_repeat];

    function reshape(input_array: Array, s: Shape): Array
        guard total(shape(input_array)) == total(s) {

        var new_array = create_array(s);

        var counter = elem_uint(zero());

        call reshape_repeat(input_array, new_array, counter);

        value new_array;
    }

}

implementation Transformations = {

    use Reshape;
    /*
    #####################
    Unpadded transpose
    #####################
    */
    predicate upper_bound(a: Array, i: IndexContainer, res: Array, c: UInt32) = {
        value isLowerThan(uint_elem(c), uint_elem(total(i)));
    }

    procedure transpose_body(obs a: Array,
                             obs ixc: IndexContainer,
                             upd res: Array,
                             upd c: UInt32) {

        var current_ix = get_index_ixc(ixc, c);

        var current_element = unwrap_scalar(get(a, reverse_index(current_ix)));
        call set(res, current_ix, current_element);
        c = elem_uint(add(uint_elem(c), one()));
    }

     use WhileLoop2_2[Context1 => Array,
                     Context2 => IndexContainer,
                     State1 => Array,
                     State2 => UInt32,
                     body => transpose_body,
                     cond => upper_bound,
                     repeat => transpose_repeat];

    function transpose(a: Array): Array = {

        var transposed_array = create_array(reverse_shape(shape(a)));

        var ix_space = create_valid_indices(transposed_array);
        var counter = elem_uint(zero());
        call transpose_repeat(a, ix_space, transposed_array, counter);

        value transposed_array;
    }

    /*
    #####################
    Padded transpose
    #####################
    */

    predicate padded_upper_bound(a: PaddedArray, i: IndexContainer, res: PaddedArray, c: UInt32) = {
        value isLowerThan(uint_elem(c), uint_elem(total(i)));
    }

    procedure padded_transpose_body(obs a: PaddedArray,
                                    obs ixc: IndexContainer,
                                    upd res: PaddedArray,
                                    upd c: UInt32) {

        var current_ix = get_index_ixc(ixc, c);

        var current_element = unwrap_scalar(get(a, reverse_index(current_ix)));
        call set(res, current_ix, current_element);
        c = elem_uint(add(uint_elem(c), one()));
    }

    use WhileLoop2_2[Context1 => PaddedArray,
                     Context2 => IndexContainer,
                     State1 => PaddedArray,
                     State2 => UInt32,
                     body => padded_transpose_body,
                     cond => padded_upper_bound,
                     repeat => padded_transpose_repeat];

    function transpose(a: PaddedArray): PaddedArray = {

        var reshaped_array = create_array(padded_shape(a));
        var transposed_array = create_padded_array(
            reverse_shape(shape(a)),                                         reverse_shape(padded_shape(a)), reshaped_array);

        var ix_space = create_valid_indices(transposed_array);
        var counter = elem_uint(zero());
        call padded_transpose_repeat(a, ix_space, transposed_array, counter);

        value transposed_array;
    }
    /*
    #######
    Unpadded reverse
    #######
    */
    procedure reverse_body(obs input: Array, obs indices: IndexContainer, upd res: Array, upd c: UInt32) = {

        var ix = get_index_ixc(indices, c);
        var elem = unwrap_scalar(get(input, ix));

        var sh_0 = get_shape_elem(input, elem_uint(zero()));
        var ix_0 = get_index_elem(ix, elem_uint(zero()));
        var new_ix_0 = sub(uint_elem(sh_0), add(uint_elem(ix_0), one()));

        var new_ix = cat_index(create_index1(elem_uint(new_ix_0)), drop_index_elem(ix, elem_uint(zero())));
        call print_index(new_ix);
        call set(res, new_ix, elem);

        c = elem_uint(add(uint_elem(c), one()));
    }

    predicate reverse_cond(input: Array, indices: IndexContainer, res: Array, c: UInt32) = {

        value isLowerThan(uint_elem(c), uint_elem(total(indices)));
    }

    use WhileLoop2_2[Context1 => Array,
                     Context2 => IndexContainer,
                     State1 => Array,
                     State2 => UInt32,
                     body => reverse_body,
                     cond => reverse_cond,
                     repeat => reverse_repeat];

    function reverse(a: Array): Array = {

        var res_array = create_array(shape(a));

        var valid_indices = create_valid_indices(res_array);

        var counter = elem_uint(zero());

        call reverse_repeat(a, valid_indices, res_array, counter);

        value res_array;
    }

    // procedure rotate(upd a: Array, step: UInt32);
}

implementation Catenation = {

    use Transformations;

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
                           upd counter: UInt32) {

        var v1_bound = uint_elem(total(v1));
        var ix: Index;

        // conditional determining if we should access v1 or v2
        if isLowerThan(uint_elem(counter), uint_elem(total(v1))) then {
            ix = create_index1(counter);
            call set(res, ix, unwrap_scalar(get(v1, ix)));
        } else {
            ix = create_index1(elem_uint(sub(uint_elem(counter), v1_bound)));
            var res_ix = create_index1(counter);
            call set(res, res_ix, unwrap_scalar(get(v2, ix)));
        };

        counter = elem_uint(add(uint_elem(counter), one()));
    }

    // determines upper bound for the iterator
    predicate cat_vec_cond(v1: Array, v2: Array, res: Array, counter: UInt32) {
        value isLowerThan(uint_elem(counter), uint_elem(total(res)));
    }

    use WhileLoop2_2[Context1 => Array,
                     Context2 => Array,
                     State1 => Array,
                     State2 => UInt32,
                     body => cat_vec_body,
                     cond => cat_vec_cond,
                     repeat => cat_vec_repeat];


    /*
    cat_vec takes two vectors as inputs, does a shape analysis, and then calls the cat_vec_repeat procedure with a correctly shaped updatable result vector argument
    */
    function cat_vec(vector1: Array, vector2: Array): Array
        guard dim(vector1) == elem_uint(one()) && dim(vector2) == elem_uint(one()) {

        var res_shape = create_shape1(elem_uint(add(
                                          uint_elem(total(vector1)),
                                          uint_elem(total(vector2)))));

        var res = create_array(res_shape);
        var counter = elem_uint(zero());
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
                       upd counter: UInt32,
                       upd res: Array) {

        var s_0 = uint_elem(total(array1));

        if isLowerThan(uint_elem(counter), s_0) then {
            call set(res, counter,
                    unwrap_scalar(get(array1,counter)));
            counter = elem_uint(add(uint_elem(counter), one()));
        } else {
            var ix = elem_uint(sub(uint_elem(counter), s_0));
            call set(res, counter,
                unwrap_scalar(get(array2, ix)));
            counter = elem_uint(add(uint_elem(counter), one()));
        };
    }

    predicate cat_cond(array1: Array, array2: Array, counter: UInt32, res: Array) {
        var upper_bound = uint_elem(total(res));
        value isLowerThan(uint_elem(counter), upper_bound);
    }
    use WhileLoop2_2[Context1 => Array,
                     Context2 => Array,
                     State1 => UInt32,
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
        guard drop_shape_elem(array1, elem_uint(zero())) ==
              drop_shape_elem(array2, elem_uint(zero())) {

        /*
        shape of the catenated array is given by:
        (take(1, a1) + take(1, a2)) 'cat' drop(1, a1)
        */
        var take_a1 = uint_elem(get_shape_elem(array1, elem_uint(zero())));
        var take_a2 = uint_elem(get_shape_elem(array2, elem_uint(zero())));
        var drop_a1 = drop_shape_elem(array1, elem_uint(zero()));

        var result_shape = cat_shape(create_shape1(elem_uint(add(take_a1, take_a2))), drop_a1);

        var res = create_array(result_shape);

        var counter = elem_uint(zero());
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
    function circular_padr(a: PaddedArray, ix: UInt32): PaddedArray = {

        // store unpadded shape
        var unpadded_shape = shape(a);

        // extract slice from a
        var padding = get(a, create_index1(ix));

        // "conform" shape of the padding to match a
        var reshape_shape = cat_shape(create_shape1(elem_uint(one())), shape(padding));
        var reshaped_padding = reshape(padding, reshape_shape);

        // cat the slice to a
        var catenated_array = cat(padded_to_unpadded(a), reshaped_padding);

        var padded_shape = shape(catenated_array);

        var res = create_padded_array(unpadded_shape, padded_shape, catenated_array);

        value res;

    }


    function circular_padr(a: Array, ix: UInt32): PaddedArray = {

        var padding = get(a, create_index1(ix));
        var reshape_shape = cat_shape(create_shape1(elem_uint(one())), shape(padding));
        var reshaped_padding = reshape(padding, reshape_shape);

        var catenated_array = cat(a, reshaped_padding);
        var unpadded_shape = shape(a);
        var padded_shape = shape(catenated_array);

        var res = create_padded_array(unpadded_shape, padded_shape, catenated_array);

        value res;
    }

    function circular_padl(a: PaddedArray, ix: UInt32): PaddedArray = {
        var padding = get(a, create_index1(ix));

        var reshape_shape = cat_shape(create_shape1(elem_uint(one())), shape(padding));
        var reshaped_padding = reshape(padding, reshape_shape);

        var catenated_array = cat(reshaped_padding, padded_to_unpadded(a));

        var unpadded_shape = shape(a);
        var padded_shape = shape(catenated_array);

        var res = create_padded_array(unpadded_shape, padded_shape, catenated_array);

        value res;
    }

    function circular_padl(a: Array, ix: UInt32): PaddedArray = {
        var padding = get(a, create_index1(ix));

        var reshape_shape = cat_shape(create_shape1(elem_uint(one())), shape(padding));
        var reshaped_padding = reshape(padding, reshape_shape);

        var catenated_array = cat(reshaped_padding, a);

        var unpadded_shape = shape(a);
        var padded_shape = shape(catenated_array);

        var res = create_padded_array(unpadded_shape, padded_shape, catenated_array);

        value res;
    }
}

program ArrayProgram = {

    use Int32Utils;
    use Float64Utils;

    use Padding[Element => Int32];
    //use Padding[Element => Float64];

}