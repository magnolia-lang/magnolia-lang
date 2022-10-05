package examples.moa.mg-src.array-ops
    imports examples.moa.mg-src.externals.array-externals,
            examples.moa.mg-src.externals.number-types-externals,
            examples.moa.mg-src.externals.while-loops;

concept MappedOps = {

    require type Element;

    require function zero(): Element;
    require function one(): Element;

    require function _+_(a: Element, b: Element): Element;
    require function _-_(a: Element, b: Element): Element;
    require function _*_(a: Element, b: Element): Element;
    require function _/_(a: Element, b: Element): Element;
    require function -_(a: Element): Element;

    require predicate _<_(a: Element, b: Element);
    require predicate _==_(a: Element, b: Element);

    type Array;
    type Index;

    function _+_(a: Array, b: Array): Array;
    function _-_(a: Array, b: Array): Array;
    function _*_(a: Array, b: Array): Array;
    function _/_(a: Array, b: Array): Array;
    function -_(a: Array): Array;

    predicate _==_(a: Array, b: Array);

    // scalar left arguments
    function _+_(a: Element, b: Array): Array;
    function _-_(a: Element, b: Array): Array;
    function _*_(a: Element, b: Array): Array;
    function _/_(a: Element, b: Array): Array;

    function get(a: Array, ix: Index): Array;
    function unwrap_scalar(a: Array): Element;


    axiom binaryMap(a: Array, b: Array, ix: Index) {
        assert unwrap_scalar(get(a+b, ix)) ==
        unwrap_scalar(get(a, ix)) + unwrap_scalar(get(b, ix));

        assert unwrap_scalar(get(a-b, ix)) ==
        unwrap_scalar(get(a, ix)) - unwrap_scalar(get(b, ix));

        assert unwrap_scalar(get(a*b, ix)) ==
        unwrap_scalar(get(a, ix)) * unwrap_scalar(get(b, ix));

        assert unwrap_scalar(get(a/b, ix)) ==
        unwrap_scalar(get(a, ix)) / unwrap_scalar(get(b, ix));
    }

    axiom scalarLeftMap(e: Element, a: Array, ix: Index) {
        assert unwrap_scalar(get(e+a, ix)) ==
            e + unwrap_scalar(get(a, ix));
        assert unwrap_scalar(get(e-a, ix)) ==
            e - unwrap_scalar(get(a, ix));
        assert unwrap_scalar(get(e*a, ix)) ==
            e * unwrap_scalar(get(a, ix));
        assert unwrap_scalar(get(e/a, ix)) ==
            e / unwrap_scalar(get(a, ix));
    }

    axiom unaryMap(a: Array, ix: Index) {
        assert unwrap_scalar(get(-a, ix)) == - unwrap_scalar(get(a, ix));
    }

}

renaming ArithmeticsRenaming = [binary_add => _+_,
                                binary_sub => _-_,
                                mul => _*_,
                                div => _/_,
                                unary_sub => -_,
                                le => _<=_,
                                lt => _<_];


implementation BopmapOpsImpl = {

    use ExtOps[ArithmeticsRenaming];

    require function bop(a: Element, b: Element): Element;

    predicate mapped_ops_cond(a: Array,
                              b: Array,
                              ix_space: IndexContainer,
                              res: Array,
                              c: Int) {

        value c < size(ix_space);
    }

    procedure bopmap_body(obs a: Array, obs b: Array, obs ix_space: IndexContainer, upd res: Array, upd c: Int) {

        var ix = get_index_ixc(ix_space, c);
        var new_value = bop(unwrap_scalar(get(a, ix)), unwrap_scalar(get(b, ix)));


        call set(res, ix, new_value);

        c = c + one(): Int;
    }

    use WhileLoop3_2[Context1 => Array,
                     Context2 => Array,
                     Context3 => IndexContainer,
                     State1 => Array,
                     State2 => Int,
                     body => bopmap_body,
                     cond => mapped_ops_cond,
                     repeat => bopmap_repeat];

    function bopmap(a: Array, b: Array): Array guard shape(a) == shape(b) = {

        var ix_space = create_total_indices(a);

        var res = create_array(shape(a));

        var counter = zero(): Int;

        call bopmap_repeat(a,b,ix_space,res,counter);

        value res;
    }
}

implementation ScalarLeftMapImpl = {

    use ExtOps[ArithmeticsRenaming];

    require function bop(a: Element, b: Element): Element;

    predicate leftmap_cond(e: Element,
                              ix_space: IndexContainer,
                              a: Array,
                              c: Int) {

        value c < size(ix_space);
    }

    procedure leftmap_body(obs e: Element,  obs ix_space: IndexContainer,
                       upd a: Array, upd c: Int) {

        var ix = get_index_ixc(ix_space, c);
        var new_value = bop(e, unwrap_scalar(get(a, ix)));

        call set(a, ix, new_value);

        c = c + one(): Int;
    }

    use WhileLoop2_2[Context1 => Element,
                     Context2 => IndexContainer,
                     State1 => Array,
                     State2 => Int,
                     body => leftmap_body,
                     cond => leftmap_cond,
                     repeat => leftmap_repeat];

    function leftmap(e: Element, a: Array): Array = {

        var ix_space = create_total_indices(a);

        var upd_a = a;

        var counter = zero(): Int;

        call leftmap_repeat(e,ix_space, upd_a, counter);

        value upd_a;
    }
}

implementation UnaryMapImpl = {

    use ExtOps[ArithmeticsRenaming];

    require function -_(a: Element): Element;

    predicate unary_sub_cond(ix_space: IndexContainer, a: Array, c: Int) {
        value c < size(ix_space);
    }

    procedure unary_sub_body(obs ix_space: IndexContainer,
                             upd a: Array,
                             upd c: Int) {

        var ix = get_index_ixc(ix_space, c);

        var new_value = - unwrap_scalar(get(a, ix));
        call set(a, ix, new_value);

        c = c + one(): Int;
    }

    use WhileLoop1_2[Context1 => IndexContainer,
                     State1 => Array,
                     State2 => Int,
                     body => unary_sub_body,
                     cond => unary_sub_cond,
                     repeat => unary_sub_repeat];


    function unary_sub(a: Array): Array = {

        var ix_space = create_total_indices(a);

        var a_upd = a;

        var counter = zero(): Int;

        call unary_sub_repeat(ix_space, a_upd, counter);

        value a_upd;

    }

}
