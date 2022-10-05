package examples.moa.mg-src.inner-product
    imports examples.moa.mg-src.moa-cpp;


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

        var shape_a_last_ix = elem_uint(uint_elem(total(shape(a))) - one());

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

implementation MatMult2D = {

    use Padding;

    use VectorReductionImpl[bop => _+_, id => zero,
                            reduce => reduce_vec_add,
                            reduce_body => reduce_body_add,
                            repeat => repeat_reduce_vec_add];

    use VectorReductionImpl[bop => _*_, id => one,
                            repeat => repeat_reduce_vec_mult,
                            reduce => reduce_vec_mult];

    use BMapVectorImpl[bop => _+_, bopmap_vec => bopmap_vec_add,
                       bmapvector_body => bmapvector_body_add,
                       bmapvector_repeat => bmapvector_repeat_add];
    use BMapVectorImpl[bop => _*_, bopmap_vec => bopmap_vec_mult];



    procedure pointwise_add_vector_body(obs v: Array,
                                        upd res: Array,
                                        upd c: UInt32) {

        var elem = unwrap_scalar(get(v, c));

        call set(res, c, unwrap_scalar(get(res,c)) + elem);

        c = elem_uint(uint_elem(c) + one());
    }

    predicate pointwise_cond(v: Array, res: Array, c: UInt32) {
        value uint_elem(c) < uint_elem(total(res));
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

        counter = elem_uint(uint_elem(counter) + one());

    }

    predicate inner_matmult_cond(a1: Array, a2: Array,
                                 ix_i: Index, ix_j: Index,
                                 ixc_k: IndexContainer, res: Array,
                                 c: UInt32) {

        value uint_elem(c) < uint_elem(total(ixc_k));
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

        counter = elem_uint(uint_elem(counter) + one());
    }

    predicate middle_matmult_cond(a1: Array, a2: Array,
                                 ix_i: Index, ixc_j: IndexContainer,
                                 ixc_k: IndexContainer, res: Array,
                                 c: UInt32) {

        value uint_elem(c) < uint_elem(total(ixc_j));
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

        counter = elem_uint(uint_elem(counter) + one());
    }

    predicate matmult2d_cond(a1: Array, a2: Array,
                                 ixc_i: IndexContainer, ixc_j: IndexContainer,
                                 ixc_k: IndexContainer, res: Array,
                                 c: UInt32) {

      value uint_elem(c) < uint_elem(total(ixc_i));
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

        var shape_a1_last_ix = elem_uint(uint_elem(total(shape(a1))) - one());

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