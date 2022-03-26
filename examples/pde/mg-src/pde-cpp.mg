package examples.pde.mg-src.pde-cpp;

implementation PDE = {
    type Array;
    type Float;
    type Axis;
    type Offset;

    /* Float ops */
    require function -_(f: Float): Float;
    require function _+_(lhs: Float, rhs: Float): Float;
    require function _-_(lhs: Float, rhs: Float): Float;
    require function _*_(lhs: Float, rhs: Float): Float;
    require function _/_(num: Float, den: Float): Float;
    require function one(): Float;
    require function two(): Float;
    require function three(): Float;

    /* Scalar-Array ops */
    require function _+_(lhs: Float, rhs: Array): Array;
    require function _-_(lhs: Float, rhs: Array): Array;
    require function _*_(lhs: Float, rhs: Array): Array;
    require function _/_(num: Float, den: Array): Array;

    /* Array-Array ops */
    require function _+_(lhs: Array, rhs: Array): Array;
    require function _-_(lhs: Array, rhs: Array): Array;
    require function _*_(lhs: Array, rhs: Array): Array;

    /* Rotate */
    require function rotate(a: Array, axis: Axis, o: Offset): Array;

    /* Axis utils */
    require function zero(): Axis;
    require function one(): Axis;
    require function two(): Axis;

    /* Offset utils */
    require function one(): Offset;
    require function -_(o: Offset): Offset;

    /* Solver */
    procedure step(upd u0: Array, upd u1: Array, upd u2: Array,
                   obs nu: Float, obs dx: Float, obs dt: Float) {
        var one = one(): Float;
        var _2 = two(): Float;

        var c0 = one/_2/dx;
        var c1 = one/dx/dx;
        var c2 = _2/dx/dx;
        var c3 = nu;
        var c4 = dt/_2;

        call all_substeps(u0, u1, u2, c0, c1, c2, c3, c4);
    }

    // TODO: possibly instantiate v variables inside?
    procedure all_substeps(upd u0: Array, upd u1: Array, upd u2: Array,
                           obs c0: Float, obs c1: Float, obs c2: Float,
                           obs c3: Float, obs c4: Float) {
        var v0 = u0;
        var v1 = u1;
        var v2 = u2;

        // TODO: transpose as AoS instead of SoA
        v0 = forall_ix_snippet(v0, u0, u0, u1, u2, c0, c1, c2, c3, c4);
        v1 = forall_ix_snippet(v1, u1, u0, u1, u2, c0, c1, c2, c3, c4);
        v2 = forall_ix_snippet(v2, u2, u0, u1, u2, c0, c1, c2, c3, c4);
        u0 = forall_ix_snippet(u0, v0, u0, u1, u2, c0, c1, c2, c3, c4);
        u1 = forall_ix_snippet(u1, v1, u0, u1, u2, c0, c1, c2, c3, c4);
        u2 = forall_ix_snippet(u2, v2, u0, u1, u2, c0, c1, c2, c3, c4);
    }

    require function forall_ix_snippet(u: Array, v: Array,
                                       u0: Array, u1: Array, u2: Array,
                                       c0: Float, c1: Float, c2: Float,
                                       c3: Float, c4: Float): Array;
    type Index;

    require function psi(ix: Index, array: Array): Float;

    function snippet_ix(u: Array, v: Array, u0: Array,
                        u1: Array, u2: Array, c0: Float,
                        c1: Float, c2: Float, c3: Float,
                        c4: Float, ix: Index): Float {
        var zero = zero();
        var one = one(): Offset;
        var two = two(): Axis;

        var result = psi(ix, u + c4 * (c3 * (c1 * (rotate(v, zero, -one) + rotate(v, zero, one) + rotate(v, one(): Axis, -one) + rotate(v, one(): Axis, one) + rotate(v, two, -one) + rotate(v, two, one)) - three() * c2 * u0) - c0 * ((rotate(v, zero, one) - rotate(v, zero, -one)) * u0 + (rotate(v, one(): Axis, one) - rotate(v, one(): Axis, -one)) * u1 + (rotate(v, two, one) - rotate(v, two, -one)) * u2)));

        value result;
        //call set(ix, u, result);
    }
}

program PDEProgram = {
    use PDE[-_ => unary_sub, _-_ => binary_sub];
    use ExtArrayOps[ one_float => one
                   , two_float => two
                   , three_float => three
                   , zero_axis => zero
                   , one_axis => one
                   , two_axis => two
                   , one_offset => one
                   ];
    use ExtExtendMissingBypass;
}


// This is because snippet_ix can not be required in ExtArrayOps, because
// then the types could not be ordered properly. We need to impleement the
// extend mechanism in Magnolia to avoid having to require the types here.
implementation ExtExtendMissingBypass = external C++ base.forall_ops {
    require type Float;
    require type Array;
    require type Offset;
    require type Axis;
    require type Index;
    require type Nat;

    require function snippet_ix(u: Array, v: Array,
                                u0: Array, u1: Array, u2: Array,
                                c0: Float, c1: Float, c2: Float,
                                c3: Float, c4: Float, ix: Index): Float;

    function forall_ix_snippet(u: Array, v: Array,
                               u0: Array, u1: Array, u2: Array,
                               c0: Float, c1: Float, c2: Float,
                               c3: Float, c4: Float): Array;

    /* OF Lift Cores extension */
    function forall_ix_snippet_threaded(u: Array, v: Array, u0: Array,
                                        u1: Array, u2: Array, c0: Float,
                                        c1: Float, c2: Float, c3: Float,
                                        c4: Float, nbThreads: Nat): Array;
    function nbCores(): Nat;

    /* OF Tiled extension */
    function forall_ix_snippet_tiled(u: Array, v: Array, u0: Array,
                                     u1: Array, u2: Array, c0: Float,
                                     c1: Float, c2: Float, c3: Float,
                                     c4: Float): Array;

    /* OF Pad extension */
    procedure refill_all_padding(upd a: Array);

    function forall_ix_snippet_padded(u: Array, v: Array,
        u0: Array, u1: Array, u2: Array, c0: Float,
        c1: Float, c2: Float, c3: Float, c4: Float): Array;

    function rotate_ix_padded(ix: Index, axis: Axis, offset: Offset): Index;
    // type PaddedArray;
    // type PaddingAmount;

    // function cpadlr(a: Array, axis: Axis, n: PaddingAmount)
    //     : PaddedArray;
    // function inner(pa: PaddedArray): Array;
    // function paddingAmount(): PaddingAmount;
    // function forall_ix_snippet_padded(u: PaddedArray, v: PaddedArray,
    //     u0: PaddedArray, u1: PaddedArray, u2: PaddedArray, c0: Float,
    //     c1: Float, c2: Float, c3: Float, c4: Float): PaddedArray;
    // function padded_rotate_ix()

}

implementation ExtArrayOps = external C++ base.array_ops {
    type Float;
    type Array;
    type Offset;
    type Axis;
    type Index;
    type Nat;

    function psi(ix: Index, array: Array): Float;
    //procedure set(obs ix: Index, upd array: Array, obs v: Float);

    /* Float ops */
    function unary_sub(f: Float): Float;
    function _+_(lhs: Float, rhs: Float): Float;
    function binary_sub(lhs: Float, rhs: Float): Float; // TODO: look at bug here
    function _*_(lhs: Float, rhs: Float): Float;
    function _/_(num: Float, den: Float): Float;
    function one_float(): Float;
    function two_float(): Float;
    function three_float(): Float;

    /* Scalar-Array ops */
    function _+_(lhs: Float, rhs: Array): Array;
    function binary_sub(lhs: Float, rhs: Array): Array;
    function _*_(lhs: Float, rhs: Array): Array;
    function _/_(num: Float, den: Array): Array;

    /* Array-Array ops */
    function _+_(lhs: Array, rhs: Array): Array;
    function binary_sub(lhs: Array, rhs: Array): Array;
    function _*_(lhs: Array, rhs: Array): Array;

    /* Rotate */
    function rotate(a: Array, axis: Axis, o: Offset): Array;
    function rotate_ix(ix: Index, axis: Axis, o: Offset): Index;

    /* Axis utils */
    function zero_axis(): Axis;
    function one_axis(): Axis;
    function two_axis(): Axis;

    /* Offset utils */
    function one_offset(): Offset;
    function unary_sub(o: Offset): Offset;
}

concept DNFGenericBinopRule = {
    type E;
    type Array;
    type Index;

    function binop(lhs: E, rhs: E): E;
    function binop(lhs: E, rhs: Array): Array;
    function binop(lhs: Array, rhs: Array): Array;
    function psi(ix: Index, array: Array): E;

    // R1
    axiom binopArrayRule(ix: Index, lhs: Array, rhs: Array) {
        assert psi(ix, binop(lhs, rhs)) == binop(psi(ix, lhs), psi(ix, rhs));
    }

    // R2
    axiom binopScalarRule(ix: Index, lhs: E, rhs: Array) {
        assert psi(ix, binop(lhs, rhs)) == binop(lhs, psi(ix, rhs));
    }
}

concept DNFRules = {
    use DNFGenericBinopRule[ E => Float
                           , binop => _+_
                           , binopArrayRule => binopArrayRulePlus
                           , binopScalarRule => binopScalarRulePlus
                           ];
    use DNFGenericBinopRule[ E => Float
                           , binop => _*_
                           , binopArrayRule => binopArrayRuleMul
                           , binopScalarRule => binopScalarRuleMul
                           ];
    use DNFGenericBinopRule[ E => Float
                           , binop => binary_sub
                           , binopArrayRule => binopArrayRuleSub
                           , binopScalarRule => binopScalarRuleSub
                           ];
    // TODO: R3
    type Axis;
    type Offset;
    function rotate(array: Array, axis: Axis, offset: Offset): Array;
    function rotate_ix(ix: Index, axis: Axis, offset: Offset): Index;

    axiom rotateRule(ix: Index, array: Array, axis: Axis,
                     offset: Offset) {
        assert psi(ix, rotate(array, axis, offset)) ==
               psi(rotate_ix(ix, axis, offset), array);
    }
}

/// End of DNF base, start of experiments

concept OFLiftCores = {
    type Array;
    type Float;
    type Axis;
    type Nat;

    function nbCores(): Nat;

    function forall_ix_snippet_threaded(u: Array, v: Array, u0: Array,
                                        u1: Array, u2: Array, c0: Float,
                                        c1: Float, c2: Float, c3: Float,
                                        c4: Float, nbThreads: Nat): Array;
    function forall_ix_snippet(u: Array, v: Array,
                               u0: Array, u1: Array, u2: Array,
                               c0: Float, c1: Float, c2: Float,
                               c3: Float, c4: Float): Array;

    axiom liftCoresRule(u: Array, v: Array, u0: Array, u1: Array, u2: Array,
                        c0: Float, c1: Float, c2: Float, c3: Float, c4: Float) {
        var d = nbCores();

        assert forall_ix_snippet(u, v, u0, u1, u2, c0, c1, c2, c3, c4) ==
               forall_ix_snippet_threaded(u, v, u0, u1, u2, c0, c1, c2, c3, c4, d);
    }
}

concept OFTile = {
    type Array;
    type Float;

    function forall_ix_snippet_tiled(u: Array, v: Array, u0: Array,
                                     u1: Array, u2: Array, c0: Float,
                                     c1: Float, c2: Float, c3: Float,
                                     c4: Float): Array;
    function forall_ix_snippet(u: Array, v: Array,
                               u0: Array, u1: Array, u2: Array,
                               c0: Float, c1: Float, c2: Float,
                               c3: Float, c4: Float): Array;

    axiom tileRule(u: Array, v: Array, u0: Array, u1: Array, u2: Array,
                   c0: Float, c1: Float, c2: Float, c3: Float, c4: Float) {
        assert forall_ix_snippet(u, v, u0, u1, u2, c0, c1, c2, c3, c4) ==
               forall_ix_snippet_tiled(u, v, u0, u1, u2, c0, c1, c2, c3, c4);
    }
}

// concept OFPad = {
//     type Array;
//     type PaddedArray;
//     type Float;
//     type PaddingAmount;
//     type Axis;

//     function cpadlr(a: Array, axis: Axis, n: PaddingAmount)
//         : PaddedArray;
//     function inner(pa: PaddedArray): Array;

//     function axis(): Axis;
//     function paddingAmount(): PaddingAmount;

//     function forall_ix_snippet_padded(u: PaddedArray, v: PaddedArray,
//         u0: PaddedArray, u1: PaddedArray, u2: PaddedArray, c0: Float,
//         c1: Float, c2: Float, c3: Float, c4: Float): PaddedArray;

//     function forall_ix_snippet(u: Array, v: Array,
//                                u0: Array, u1: Array, u2: Array,
//                                c0: Float, c1: Float, c2: Float,
//                                c3: Float, c4: Float): Array;


//     axiom padRule(u: Array, v: Array, u0: Array, u1: Array, u2: Array,
//                   c0: Float, c1: Float, c2: Float, c3: Float, c4: Float) {
//         var a = axis();
//         var p = paddingAmount();
//         assert forall_ix_snippet(u, v, u0, u1, u2, c0, c1, c2, c3, c4) ==
//                inner(forall_ix_snippet_padded(cpadlr(u, a, p), cpadlr(v, a, p),
//                     cpadlr(u0, a, p), cpadlr(u1, a, p), cpadlr(u2, a, p),
//                     c0, c1, c2, c3, c4));
//     }
// }

// For the below, we assume that the inputs are padded as needed.
concept OFPad = {
    type Array;
    type Float;

    procedure all_substeps(upd u0: Array, upd u1: Array, upd u2: Array,
                           obs c0: Float, obs c1: Float, obs c2: Float,
                           obs c3: Float, obs c4: Float);

    procedure refill_all_padding(upd a: Array);

    function forall_ix_snippet_padded(u: Array, v: Array,
        u0: Array, u1: Array, u2: Array, c0: Float,
        c1: Float, c2: Float, c3: Float, c4: Float): Array;

    function forall_ix_snippet(u: Array, v: Array,
                               u0: Array, u1: Array, u2: Array,
                               c0: Float, c1: Float, c2: Float,
                               c3: Float, c4: Float): Array;

    axiom padRule(u: Array, v: Array, u0: Array, u1: Array, u2: Array,
                  c0: Float, c1: Float, c2: Float, c3: Float, c4: Float) {
        assert forall_ix_snippet(u, v, u0, u1, u2, c0, c1, c2, c3, c4) ==
               { var result = forall_ix_snippet_padded(u, v, u0, u1, u2, c0,
                    c1, c2, c3, c4);
                 call refill_all_padding(result);
                 value result;
               };
    }

    type Index;
    type Axis;
    type Offset;
    function rotate_ix(ix: Index, axis: Axis, offset: Offset): Index;
    function rotate_ix_padded(ix: Index, axis: Axis, offset: Offset): Index;

    // We can replace rotate_ix by a more optimal rotation
    axiom rotateIxPadRule(ix: Index, axis: Axis, offset: Offset) {
        assert rotate_ix(ix, axis, offset) ==
               rotate_ix_padded(ix, axis, offset);
    }
}

// concept OFPad0 = OFPad[ axis => zero ];
// concept OFPad1 = OFPad[ axis => one ];
// concept OFPad2 = OFPad[ axis => two ];

/*concept OFTest = {
    type Array;
    type Float;
    type PaddingAmount;
    type Axis;

    procedure all_substeps(upd u0: Array, upd u1: Array, upd u2: Array,
                           obs c0: Float, obs c1: Float, obs c2: Float,
                           obs c3: Float, obs c4: Float);

    axiom test(u0: Array, u1: Array, u2: Array, c0: Float, c1: Float, c2: Float,
               c3: Float, c4: Float) {

        var c0u = c0;
        var c1u = c1;
        var c2u = c2;
        var c3u = c3;
        var c4u = c4;

        assert { var u0u = u0;
                 var u1u = u1;
                 var u2u = u2;
                 call all_substeps(u0u, u1u, u1u, c0u, c1u, c2u, c3u, c4u);
               } ==
               { var u0p = u0;
                 var u1p = u1;
                 var u2p = u2;
                 call all_substeps(u0p, u1p, u2p, c0u, c1u, c2u, c3u, c4u);
               };
    }
}*/