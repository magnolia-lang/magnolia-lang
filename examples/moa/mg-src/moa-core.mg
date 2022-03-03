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
/*
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
        value c < total(input);
    }

    procedure reduce_body(obs input: Array, upd res: Element, upd c: Int) {

        var current_element = unwrap_scalar(get(input, c));
        res = bop(res, current_element);
        c = c + one(): Int;

    }

    use WhileLoop1_2[Context1 => Array,
                     State1 => Element,
                     State2 => Int,
                     body => reduce_body,
                     cond => reduce_cond];

    function reduce(a: Array): Element = {

        var result = id();

        var counter = zero(): Int;

        call repeat(a, result, counter);

        value result;
    }
}
*/

/*
    ravel takes an array as input and flattens it to a vector

    i.e. dim(ravel(a)) = 1
*/
implementation Ravel = {

    use MoaOps;

    predicate ravel_cond(input: Array, flat: Array, c: Int) {
        value c < total(input);
    }

    procedure ravel_body(obs input: Array, upd flat: Array, upd c: Int) {

        var linear_ix = create_index1(c);

        call set(flat, linear_ix, unwrap_scalar(get(input, c)));

        c = c + one(): Int;

    }

    use WhileLoop1_2[Context1 => Array,
                     State1 => Array,
                     State2 => Int,
                     body => ravel_body,
                     cond => ravel_cond,
                     repeat => rav_repeat];


    function ravel(a: Array): Array = {

        var c = zero(): Int;

        var flat = create_array(create_shape1(total(a)));

        call rav_repeat(a, flat, c);

        value flat;
    }

}

/*
    reshape takes as input an array a1 and a shape s.
    If the total number of elements in the input array a1 matches
    exactly the number of elements the shape s can hold then reshape a1 to
    have shape s.

    total(shape(a)) == total(s)
*/
implementation Reshape = {

    use MoaOps;

    procedure reshape_body(obs old_array: Array, upd new_array: Array, upd counter: Int) {
        call set(new_array, counter, unwrap_scalar(get(old_array, counter)));
        counter = counter + one(): Int;
    }
    predicate reshape_cond(old_array: Array, new_array: Array, counter: Int) {
       value counter < total(new_array);
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

        var counter = zero(): Int;

        call reshape_repeat(input_array, new_array, counter);

        value new_array;
    }

}

/*

    Catenation of arrays. Concatenates the input arrays a1 and a2 on the primary axis.

    Precondition: drop(0, shape(a1)) == drop(0, shape(a2))


*/
implementation Catenation = {

    use Reshape;

    procedure cat_body(obs a: Array,
                       obs b: Array,
                       obs ixc: IndexContainer,
                       upd res: Array,
                       upd c: Int) {

        var ix = get_index_ixc(ixc, c);

        var s0 = get_shape_elem(a, zero(): Int);
        var i0 = get_index_elem(ix, zero(): Int);

        if i0 < s0 then {

            call set(res, ix, get(a,ix));

        } else {

            var new_ix = create_index1(i0 - s0);
            call set(res, ix, get(b, new_ix));
        };

        c = c + one(): Int;

    }

    predicate cat_cond(a: Array, b: Array, ixc: IndexContainer,
                       res: Array, c: Int) {
        value c < total(ixc);
    }

    use WhileLoop3_2[Context1 => Array,
                     Context2 => Array,
                     Context3 => IndexContainer,
                     State1 => Array,
                     State2 => Int,
                     body => cat_body,
                     cond => cat_cond,
                     repeat => cat_repeat];

    function cat(a: Array, b: Array): Array
        guard drop_shape_elem(a, zero(): Int) ==
              drop_shape_elem(b, zero(): Int) {

        var drop_s0 = drop_shape_elem(a, zero(): Int);

        var s0a_s0b = create_shape1(get_shape_elem(a, zero(): Int) +
                                    get_shape_elem(b, zero(): Int));

        var res_shape = cat_shape(s0a_s0b, drop_s0);
        var res = create_array(res_shape);

        var ixc = create_partial_indices(res, one(): Int);

        var c = zero(): Int;

        call cat_repeat(a,b,ixc,res,c);

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
        var reshape_shape = cat_shape(create_shape1(one(): Int), shape(padding));
        var reshaped_padding = reshape(padding, reshape_shape);

        // cat the slice to a
        var catenated_array = cat(padded_to_unpadded(a), reshaped_padding);

        var padded_shape = shape(catenated_array);

        var res = create_padded_array(unpadded_shape, padded_shape, catenated_array);

        value res;

    }


    function circular_padr(a: Array, ix: Int): PaddedArray = {

        var padding = get(a, create_index1(ix));
        var reshape_shape = cat_shape(create_shape1(one(): Int), shape(padding));
        var reshaped_padding = reshape(padding, reshape_shape);

        var catenated_array = cat(a, reshaped_padding);
        var unpadded_shape = shape(a);
        var padded_shape = shape(catenated_array);

        var res = create_padded_array(unpadded_shape, padded_shape, catenated_array);

        value res;
    }

    function circular_padl(a: PaddedArray, ix: Int): PaddedArray = {
        var padding = get(a, create_index1(ix));

        var reshape_shape = cat_shape(create_shape1(one(): Int), shape(padding));
        var reshaped_padding = reshape(padding, reshape_shape);

        var catenated_array = cat(reshaped_padding, padded_to_unpadded(a));

        var unpadded_shape = shape(a);
        var padded_shape = shape(catenated_array);

        var res = create_padded_array(unpadded_shape, padded_shape, catenated_array);

        value res;
    }

    function circular_padl(a: Array, ix: Int): PaddedArray = {
        var padding = get(a, create_index1(ix));

        var reshape_shape = cat_shape(create_shape1(one(): Int), shape(padding));
        var reshaped_padding = reshape(padding, reshape_shape);

        var catenated_array = cat(reshaped_padding, a);

        var unpadded_shape = shape(a);
        var padded_shape = shape(catenated_array);

        var res = create_padded_array(unpadded_shape, padded_shape, catenated_array);

        value res;
    }

}

/*

    Take and drop.

    For parameters i: Int and a: Array, take(i,a) returns the i first elements
    of a. If i <= 0, return the |i| last elements of a.

    For parameters i: Int and a: Array, drop(i,a) returns a with the i first elements dropped. If i <= 0, drop the |i| last elements of a.

*/
implementation TakeDrop = {

    use Padding;

    predicate take_cond(a: Array, t: Int, ixc: IndexContainer, res: Array, c: Int) {

        value c < abs(t);

    }

    procedure take_body(obs a: Array, obs t: Int,
                        obs ixc: IndexContainer,
                        upd res: Array, upd c: Int) {

        var ix = get_index_ixc(ixc, c);

        if zero(): Int <= t then {

            call set(res, ix, get(a, ix));

        } else {

            var s0 = get_shape_elem(a, zero(): Int);
            var i0 = get_index_elem(ix, zero(): Int);

            var new_ix = create_index1(s0 - abs(t) + i0);

            call set(res, ix, get(a,new_ix));

        };

        c = c + one(): Int;

    }

    use WhileLoop3_2[Context1 => Array,
                     Context2 => Int,
                     Context3 => IndexContainer,
                     State1 => Array,
                     State2 => Int,
                     body => take_body,
                     cond => take_cond,
                     repeat => take_repeat];

    function take(t: Int, a: Array): Array = {

        var drop_sh_0 = drop_shape_elem(a, zero(): Int);

        var res = create_array(cat_shape(create_shape1(abs(t)), drop_sh_0));

        var ixc = create_partial_indices(res, one(): Int);

        var c = zero(): Int;

        call take_repeat(a,t,ixc,res,c);

        value res;

    }

    predicate drop_cond(a: Array, t: Int, ixc: IndexContainer, res: Array, c: Int) {

        value c < total(ixc);

    }

    procedure drop_body(obs a: Array, obs t: Int, obs ixc: IndexContainer,
                        upd res: Array, upd c: Int) {

        var ix = get_index_ixc(ixc, c);

        if zero(): Int <= t then {

            var i0 = get_index_elem(ix, zero(): Int);
            var new_ix = create_index1(i0 + t);

            call set(res, ix, get(a, new_ix));

        } else {
            call set(res, ix, get(a, ix));
        };

        c = c + one(): Int;
    }

    use WhileLoop3_2[Context1 => Array,
                     Context2 => Int,
                     Context3 => IndexContainer,
                     State1 => Array,
                     State2 => Int,
                     body => drop_body,
                     cond => drop_cond,
                     repeat => drop_repeat];

    function drop(t: Int, a: Array): Array {

        var s0 = get_shape_elem(a, zero(): Int);

        var drop_sh_0 = drop_shape_elem(a, zero(): Int);

        var res_shape = cat_shape(create_shape1(s0 - abs(t)), drop_sh_0);
        var res = create_array(res_shape);

        var ixc = create_partial_indices(res, one(): Int);

        var c = zero(): Int;

        call drop_repeat(a,t,ixc,res,c);

        value res;
    }
}

implementation Transformations = {

    use TakeDrop;

    predicate rotate_cond(a: Array, ixc: IndexContainer, sigma: Int, res: Array, c: Int) {

        value c < total(ixc);
    }

    procedure rotate_body(obs a: Array, obs ixc: IndexContainer,
                          obs sigma: Int, upd res: Array, upd c: Int) {

        var ix = get_index_ixc(ixc, c);

        if zero(): Int <= sigma then {

            var e1 = take(-sigma, get(a, ix));
            var e2 = drop(-sigma, get(a, ix));

            call set(res, ix, cat(e1,e2));

        } else {

            var e1 = drop(sigma, get(a, ix));
            var e2 = take(sigma, get(a, ix));

            call set(res, ix, cat(e1,e2));

        };

        c = c + one(): Int;
    }

    use WhileLoop3_2[Context1 => Array,
                     Context2 => IndexContainer,
                     Context3 => Int,
                     State1 => Array,
                     State2 => Int,
                     body => rotate_body,
                     cond => rotate_cond,
                     repeat => rotate_repeat];

    // rotate array a distance sigma along axis j
    function rotate(sigma: Int, j: Int, a: Array): Array
        guard j < dim(a) = {

        // create partial indices of length j
        var ix_space = create_partial_indices(a, j);

        var res = create_array(shape(a));

        var c = zero(): Int;

        call rotate_repeat(a,ix_space,sigma,res,c);

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

        var sh_0 = get_shape_elem(input, zero(): Int);
        var ix_0 = get_index_elem(ix, zero(): Int);
        var new_ix_0 = sh_0 - (ix_0 + one(): Int);

        var new_ix = cat_index(create_index1(new_ix_0),
                               drop_index_elem(ix, zero(): Int));

        call set(res, new_ix, elem);

        c = c + one(): Int;
    }

    predicate reverse_cond(input: Array, indices: IndexContainer, res: Array, c: Int) = {

        value c < total(indices);
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
        var counter = zero(): Int;

        call reverse_repeat(a, valid_indices, res_array, counter);

        value res_array;
    }
    /*
    #####################
    Unpadded transpose
    #####################
    */
    predicate upper_bound(a: Array, i: IndexContainer, res: Array, c: Int) = {
        value c < total(i);
    }

    procedure transpose_body(obs a: Array,
                             obs ixc: IndexContainer,
                             upd res: Array,
                             upd c: Int) {

        var current_ix = get_index_ixc(ixc, c);

        var current_element = unwrap_scalar(get(a, reverse_index(current_ix)));
        call set(res, current_ix, current_element);
        c = c + one(): Int;
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
        var counter = zero(): Int;
        call transpose_repeat(a, ix_space, transposed_array, counter);

        value transposed_array;
    }

    /*
    #####################
    Padded transpose
    #####################
    */

    predicate padded_upper_bound(a: PaddedArray, i: IndexContainer, res: PaddedArray, c: Int) = {
        value c < total(i);
    }

    procedure padded_transpose_body(obs a: PaddedArray,
                                    obs ixc: IndexContainer,
                                    upd res: PaddedArray,
                                    upd c: Int) {

        var current_ix = get_index_ixc(ixc, c);

        var current_element = unwrap_scalar(get(a, reverse_index(current_ix)));
        call set(res, current_ix, current_element);
        c = c + one(): Int;
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
        var counter = zero(): Int;
        call padded_transpose_repeat(a, ix_space, transposed_array, counter);

        value transposed_array;
    }

}