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

signature VectorReductionSig = {

    require type Element;

    type Array;
    type UInt32;

    function bop(e1: Element, e2: Element): Element;

    function id(): Element;

}

implementation VectorReduction = {

    use MoaOps;
    use VectorReductionSig;

    predicate reduce_cond(input: Array, res: Element, c: UInt32) = {
        value isLowerThan(uint_elem(c), uint_elem(total(input)));
    }

    procedure reduce_body(obs input: Array, upd res: Element, upd c: UInt32) {

        var current_element = unwrap_scalar(get(input, c));
        res = bop(res, current_element);
        c = elem_uint(add(uint_elem(c), one()));

    }

    use WhileLoop1_2[Context1 => Array,
                     State1 => Element,
                     State2 => UInt32,
                     body => reduce_body,
                     cond => reduce_cond];

    function reduce(a: Array): Element = {

        var result = id();

        var counter = elem_uint(zero());

        call repeat(a, result, counter);

        value result;
    }
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

/*
Implementation for the n-d inner product
*/
implementation InnerProduct = {

    use Padding;

    procedure ip_body(obs i: IndexContainer,
                      obs j: IndexContainer,
                      obs k: IndexContainer,
                      upd res: Array,
                      upd c: UInt32) = {



    }

    function inner_product(a: Array, b: Array): Array = {

        /*
        shape(ip(a,b)) = cat(drop(-1,shape(a)), drop(1, shape(b)))
        */

        var shape_a_last_ix = elem_uint(sub(uint_elem(total(shape(a))), one()));

        var sh_a_drop_last = drop_shape_elem(a, shape_a_last_ix);
        var sh_b_drop_first = drop_shape_elem(b, elem_uint(zero()));

        var ip_shape = cat_shape(sh_a_drop_last, sh_b_drop_first);

        var ip_array = create_array(ip_shape);

        // valid index vectors for ip

        var indices_a = create_valid_indices(create_array(sh_a_drop_last));
        var indices_b = create_valid_indices(create_array(sh_b_drop_first));

        // reduction indices within inner product
        var sh_a_take_last = create_shape1(get_shape_elem(a, shape_a_last_ix));
        var indices_k = create_valid_indices(create_array(sh_a_take_last));

        //call print_index_container(indices_k);
        //call print_index_container(create_valid_indices(ip_array));
        value ip_array;
    }

}


implementation BMap = {

    require type Element;

    type Array;

    function bop(e1: Element, e2: Element): Element;
    //function bopmap(e: Element, a: Array): Array;
}

implementation BMapVectorImpl = {

    use BMap;
    use MoaOps;



    procedure bmapvector_body(obs e: Element,
                              upd v: Array, upd c: UInt32) {

        var new_value = bop(e, unwrap_scalar(get(v, c)));

        call set(v, c, new_value);

        c = elem_uint(add(uint_elem(c), one()));
    }

    predicate bmapvector_cond(e: Element, v: Array, c: UInt32) {
        value isLowerThan(uint_elem(c), uint_elem(total(v)));
    }

    use WhileLoop1_2[Context1 => Element,
                     State1 => Array,
                     State2 => UInt32,
                     body => bmapvector_body,
                     cond => bmapvector_cond,
                     repeat => bmapvector_repeat];

    procedure bopmap_vec(obs e: Element, upd a: Array) = {
        var counter = elem_uint(zero());
        call bmapvector_repeat(e, a, counter);
    }
}


implementation MatMult2D = {

    use Padding;

    use VectorReduction[bop => add, id => zero,
                        reduce => reduce_vec_add,
                        reduce_body => reduce_body_add,
                        repeat => repeat_reduce_vec_add];

    use VectorReduction[bop => mult, id => one,
                        repeat => repeat_reduce_vec_mult,
                        reduce => reduce_vec_mult];

    use BMapVectorImpl[bop => add, bopmap_vec => bopmap_vec_add,
                       bmapvector_body => bmapvector_body_add,
                       bmapvector_repeat => bmapvector_repeat_add];
    use BMapVectorImpl[bop => mult, bopmap_vec => bopmap_vec_mult];



    procedure pointwise_add_vector_body(obs v: Array,
                                        upd res: Array,
                                        upd c: UInt32) {

        var elem = unwrap_scalar(get(v, c));

        call set(res, c, add(unwrap_scalar(get(res,c)),elem));

        c = elem_uint(add(uint_elem(c), one()));
    }

    predicate pointwise_cond(v: Array, res: Array, c: UInt32) {
        value isLowerThan(uint_elem(c), uint_elem(total(res)));
    }

    use WhileLoop1_2[Context1 => Array,
                     State1 => Array,
                     State2 => UInt32,
                     body => pointwise_add_vector_body,
                     cond => pointwise_cond,
                     repeat => pointwise_repeat];

    function pointwise_add(a1: Array, a2: Array): Array = {

        var res = create_array(shape(a1));

        var c = elem_uint(zero());
        call pointwise_repeat(a1, res, c);

        c = elem_uint(zero());

        call pointwise_repeat(a2, res, c);

        value res;

    }

    // iterates k for each iteration of (i,j) psi (a x b)

    procedure inner_matmult_body(obs a1: Array, obs a2: Array,
                              obs ix_i: Index,
                              obs ix_j: Index,
                              obs ixc_k: IndexContainer,
                              upd res: Array,
                              upd counter: UInt32) {

        var ix_k = get_index_ixc(ixc_k, counter);

        var i_cat_k = cat_index(ix_i, ix_k);

        var ik_from_a1 = get(a1, i_cat_k);
        var k_from_a2 = get(a2, ix_k);

        call bopmap_vec_mult(unwrap_scalar(ik_from_a1), k_from_a2);

        call print_array(k_from_a2);

        res = pointwise_add(res, k_from_a2);

        counter = elem_uint(add(uint_elem(counter), one()));

    }

    predicate inner_matmult_cond(a1: Array, a2: Array,
                                 ix_i: Index, ix_j: Index,
                                 ixc_k: IndexContainer, res: Array,
                                 c: UInt32) {

        value isLowerThan(uint_elem(c), uint_elem(total(ixc_k)));
    }

    use WhileLoop5_2[Context1 => Array,
                     Context2 => Array,
                     Context3 => Index,
                     Context4 => Index,
                     Context5 => IndexContainer,
                     State1 => Array,
                     State2 => UInt32,
                     body => inner_matmult_body,
                     cond => inner_matmult_cond,
                     repeat => inner_matmult_repeat];

    // iterates j

    procedure middle_matmult_body(obs a1: Array, obs a2: Array,
                                  obs ix_i: Index, obs ixc_j: IndexContainer,
                                  obs ixc_k: IndexContainer,
                                  upd res: Array,
                                  upd counter: UInt32) {

        var ix_j = get_index_ixc(ixc_j, counter);

        var inner_counter = elem_uint(zero());

        var res_vec = create_array(shape(get(a1, ix_j)));

        call inner_matmult_repeat(a1,a2,ix_i,ix_j,ixc_k,res_vec,inner_counter);

        var reduced = reduce_vec_add(res_vec);
        call print_index(cat_index(ix_i, ix_j));

        call set(res, cat_index(ix_i, ix_j), reduced);

        counter = elem_uint(add(uint_elem(counter), one()));
    }

    predicate middle_matmult_cond(a1: Array, a2: Array,
                                 ix_i: Index, ixc_j: IndexContainer,
                                 ixc_k: IndexContainer, res: Array,
                                 c: UInt32) {

        value isLowerThan(uint_elem(c), uint_elem(total(ixc_j)));
    }

    use WhileLoop5_2[Context1 => Array,
                     Context2 => Array,
                     Context3 => Index,
                     Context4 => IndexContainer,
                     Context5 => IndexContainer,
                     State1 => Array,
                     State2 => UInt32,
                     body => middle_matmult_body,
                     cond => middle_matmult_cond,
                     repeat => middle_matmult_repeat];

    procedure matmult2d_body(obs a1: Array, obs a2: Array,
                             obs ixc_i: IndexContainer,
                             obs ixc_j: IndexContainer,
                             obs ixc_k: IndexContainer,
                             upd res: Array,
                             upd counter: UInt32) {

        var ix_i = get_index_ixc(ixc_i, counter);

        var middle_counter = elem_uint(zero());

        call middle_matmult_repeat(a1,a2,ix_i,ixc_j,ixc_k,res,middle_counter);

        counter = elem_uint(add(uint_elem(counter), one()));
    }

    predicate matmult2d_cond(a1: Array, a2: Array,
                                 ixc_i: IndexContainer, ixc_j: IndexContainer,
                                 ixc_k: IndexContainer, res: Array,
                                 c: UInt32) {

      value isLowerThan(uint_elem(c), uint_elem(total(ixc_i)));
    }

    use WhileLoop5_2[Context1 => Array,
                     Context2 => Array,
                     Context3 => IndexContainer,
                     Context4 => IndexContainer,
                     Context5 => IndexContainer,
                     State1 => Array,
                     State2 => UInt32,
                     body => matmult2d_body,
                     cond => matmult2d_cond,
                     repeat => matmult2d_repeat];


    function matmult2d(a1: Array, a2: Array): Array = {

        var shape_a1_last_ix = elem_uint(sub(uint_elem(total(shape(a1))), one()));

        var sh_a1_drop_last = drop_shape_elem(a1, shape_a1_last_ix);
        var sh_a2_drop_first = drop_shape_elem(a2, elem_uint(zero()));

        var ip_shape = cat_shape(sh_a1_drop_last, sh_a2_drop_first);

        var ip_array = create_array(ip_shape);

        var indices_a1 = create_valid_indices(create_array(sh_a1_drop_last));
        var indices_a2 = create_valid_indices(create_array(sh_a2_drop_first));

        // reduction indices within inner product
        var sh_a1_take_last = create_shape1(get_shape_elem(a1, shape_a1_last_ix));
        var indices_k = create_valid_indices(create_array(sh_a1_take_last));

        var counter = elem_uint(zero());
        call matmult2d_repeat(a1,a2,indices_a1,indices_a2,indices_k,ip_array,counter);

        value ip_array;

    }
}
program ArrayProgram = {

    use Int32Utils;
    use Float64Utils;

    use MatMult2D[Element => Int32];
    //use Padding[Element => Float64];

}