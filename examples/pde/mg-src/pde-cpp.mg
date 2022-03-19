package examples.pde.mg-src.pde-cpp;

signature Magma = {
    type E;
    function binop(lhs: E, rhs: E): E;
}

concept Forall_ix = {
    require type E;
    type Array;
    type Index;
    type Context;
    type State;

    /* Array-level op */
    procedure forall_ix(out result: Array, obs ctx: Context, upd st: State);
    /* Index-level op */
    require procedure op(obs lhs: Array, obs rhs: Array, obs ix: Index,
                         upd result: Array, obs ctx: Context);
}

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

        var v0 = u0;
        var v1 = u1;
        var v2 = u2;

        call snippet(v0, u0, u0, u1, u2, c0, c1, c2, c3, c4);
        call snippet(v1, u1, u0, u1, u2, c0, c1, c2, c3, c4);
        call snippet(v2, u2, u0, u1, u2, c0, c1, c2, c3, c4);
        call snippet(u0, v0, u0, u1, u2, c0, c1, c2, c3, c4);
        call snippet(u1, v1, u0, u1, u2, c0, c1, c2, c3, c4);
        call snippet(u2, v2, u0, u1, u2, c0, c1, c2, c3, c4);
    }

    procedure snippet(upd u: Array, obs v: Array,
                      obs u0: Array, obs u1: Array, obs u2: Array,
                      obs c0: Float, obs c1: Float, obs c2: Float,
                      obs c3: Float, obs c4: Float) {
        u = forall_ix_snippet(u, v, u0, u1, u2, c0, c1, c2, c3, c4);
    }

    /*procedure snippet(upd u: Array, obs v: Array,
                      obs u0: Array, obs u1: Array, obs u2: Array,
                      obs c0: Float, obs c1: Float, obs c2: Float,
                      obs c3: Float, obs c4: Float) {
        var zero = zero();
        var one = one(): Offset;
        var two = two(): Axis;

        u = u + c4 * (c3 * (c1 * (rotate(v, zero, -one) + rotate(v, zero, one) + rotate(v, one(): Axis, -one) + rotate(v, one(): Axis, one) + rotate(v, two, -one) + rotate(v, two, one)) - three() * c2 * u0) - c0 * ((rotate(v, zero, one) - rotate(v, zero, -one)) * u0 + (rotate(v, one(): Axis, one) - rotate(v, one(): Axis, -one)) * u1 + (rotate(v, two, one) - rotate(v, two, -one)) * u2));
    }*/

    require function forall_ix_snippet(u: Array, v: Array,
                                       u0: Array, u1: Array, u2: Array,
                                       c0: Float, c1: Float, c2: Float,
                                       c3: Float, c4: Float): Array;
    type Index;

    require function psi(ix: Index, array: Array): Array;
    //require procedure set(obs ix: Index, upd array: Array, obs v: Float);

    // procedure snippet_ix(upd u: Array, obs v: Array, obs u0: Array,
    //                      obs u1: Array, obs u2: Array, obs c0: Float,
    //                      obs c1: Float, obs c2: Float, obs c3: Float,
    //                      obs c4: Float, obs ix: Index) {
    //     var zero = zero();
    //     var one = one(): Offset;
    //     var two = two(): Axis;

    //     var result = psi(ix, u + c4 * (c3 * (c1 * (rotate(v, zero, -one) + rotate(v, zero, one) + rotate(v, one(): Axis, -one) + rotate(v, one(): Axis, one) + rotate(v, two, -one) + rotate(v, two, one)) - three() * c2 * u0) - c0 * ((rotate(v, zero, one) - rotate(v, zero, -one)) * u0 + (rotate(v, one(): Axis, one) - rotate(v, one(): Axis, -one)) * u1 + (rotate(v, two, one) - rotate(v, two, -one)) * u2)));

    //     call set(ix, u, result);
    // }

    function snippet_ix(u: Array, v: Array, u0: Array,
                        u1: Array, u2: Array, c0: Float,
                        c1: Float, c2: Float, c3: Float,
                        c4: Float, ix: Index): Array {
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
    use ExtHardwareInfo;
}

implementation ExtHardwareInfo = external C++ base.hardware_info {
    type Nat;
    function nbCores(): Nat;
    function one(): Nat;
}

// This is because snippet_ix can not be required in ExtArrayOps, because
// then the types could not be ordered properly. We need to impleement the
// extend mechanism in Magnolia to avoid having to require the types here.
implementation ExtExtendMissingBypass = external C++ base.forall_ops {
    require type Float;
    require type Array;
    require type PaddedArray;
    require type Offset;
    require type Axis;
    require type Index;
    require type Nat;

    // require procedure snippet_ix(upd u: Array, obs v: Array,
    //                              obs u0: Array, obs u1: Array, obs u2: Array,
    //                              obs c0: Float, obs c1: Float, obs c2: Float,
    //                              obs c3: Float, obs c4: Float, obs ix: Index);

    require function snippet_ix(u: Array, v: Array,
                                u0: Array, u1: Array, u2: Array,
                                c0: Float, c1: Float, c2: Float,
                                c3: Float, c4: Float, ix: Index): Array;


    function forall_ix_snippet(u: Array, v: Array,
                               u0: Array, u1: Array, u2: Array,
                               c0: Float, c1: Float, c2: Float,
                               c3: Float, c4: Float): Array;

    function forall_ix_snippet_padded(u: PaddedArray, v: PaddedArray,
                                      u0: PaddedArray, u1: PaddedArray,
                                      u2: PaddedArray,
                                      c0: Float, c1: Float, c2: Float,
                                      c3: Float, c4: Float): Array;

    function forall_ix_snippet_tiled(u: Array, v: Array, u0: Array,
                                     u1: Array, u2: Array, c0: Float,
                                     c1: Float, c2: Float, c3: Float,
                                     c4: Float): Array;

    // unlift(A[<s0 s1 s2...>, ...], 1) = A[<s0 (s1*s2)...>, ...]
    function unliftAndUnpad(array: Array, axis: Axis,
                            paddingAmount: Nat): Array;
    function padAndLift(array: Array, axis: Axis, d: Nat,
                        paddingAmount: Nat): Array;
    function forall_ix_snippet_threaded(u: Array, v: Array, u0: Array,
                                        u1: Array, u2: Array, c0: Float,
                                        c1: Float, c2: Float, c3: Float,
                                        c4: Float, nbThreads: Nat): Array;
}

implementation ExtArrayOps = external C++ base.array_ops {
    type Float;
    type Array;
    type Offset;
    type Axis;
    type Index;
    type Shape;
    type LinearArray;
    type LinearIndex;
    type Stride;
    type Range;

    function shape(array: Array): Shape;
    function subshape(ix: Index, shape: Shape): Shape;
    function psi(ix: Index, array: Array): Array;
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
    function rotate_ix(ix: Index, axis: Axis, o: Offset, shape: Shape): Index;

    /* Axis utils */
    function zero_axis(): Axis;
    function one_axis(): Axis;
    function two_axis(): Axis;

    /* Offset utils */
    function one_offset(): Offset;
    function unary_sub(o: Offset): Offset;

    /* Scalar-LinearArray ops */
    function _+_(lhs: Float, rhs: LinearArray): LinearArray;
    function binary_sub(lhs: Float, rhs: LinearArray): LinearArray;
    function _*_(lhs: Float, rhs: LinearArray): LinearArray;
    function _/_(num: Float, den: LinearArray): LinearArray;

    /* LinearArray-LinearArray ops */
    function _+_(lhs: LinearArray, rhs: LinearArray): LinearArray;
    function binary_sub(lhs: LinearArray, rhs: LinearArray): LinearArray;
    function _*_(lhs: LinearArray, rhs: LinearArray): LinearArray;

    /* OF utils */
    function uniqueShape(): Shape;
    function toArray(la: LinearArray, s: Shape): Array;

    function start(ix: Index, shape: Shape): LinearIndex;
    function stride(ix: Index, shape: Shape): Stride;
    function iota(s: Stride): Range;

    function _+_(lix: LinearIndex, range: Range): Range;
    function _*_(lix: LinearIndex, stride: Stride): LinearIndex;

    function rav(a: Array): LinearArray;
    function elementsAt(la: LinearArray, r: Range): LinearArray;

    /* DNF utils */
    function emptyIndex(): Index;

    /* Rewriting, padding utils */
    type PaddedArray;

    function asPadded(a: Array): PaddedArray;
    function cpadr(a: PaddedArray, axis: Axis): PaddedArray;
    function cpadl(a: PaddedArray, axis: Axis): PaddedArray;
    // function inner(a: PaddedArray): Array;
}

concept DNFIntroducePsi = {
    type Array;
    type Index;

    function psi(ix: Index, a: Array): Array;
    function emptyIndex(): Index;

    axiom introducePsiRule(a: Array) {
        assert a == psi(emptyIndex(), a);
    }
}

concept DNFCleanupPsiIntroduction = {
    use signature(DNFIntroducePsi);

    axiom cleanUpPsiRule(a: Array) {
        assert psi(emptyIndex(), psi(emptyIndex(), a)) ==
               psi(emptyIndex(), a);
    }
}

concept DNFGenericBinopRule = {
    type E;
    type Array;
    type Index;

    function binop(lhs: E, rhs: Array): Array;
    function binop(lhs: Array, rhs: Array): Array;
    function psi(ix: Index, array: Array): Array;

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
    type Shape;
    type Axis;
    type Offset;
    function shape(array: Array): Shape;
    function rotate(array: Array, axis: Axis, offset: Offset): Array;
    function rotate_ix(ix: Index, axis: Axis, offset: Offset, shape: Shape): Index;

    axiom rotateRule(ix: Index, array: Array, axis: Axis,
                     offset: Offset) {
        assert psi(ix, rotate(array, axis, offset)) ==
               psi(rotate_ix(ix, axis, offset, shape(array)), array);
    }
}

signature OFRewritingTypesAndOps = {
    type Array;
    type PaddedArray;
    type Index;

    type Axis;
    type Nat;

    function inner(pa: PaddedArray): Array;
    function asPadded(a: Array): PaddedArray;

    function cpadl(a: PaddedArray, axis: Axis): PaddedArray;
    function cpadr(a: PaddedArray, axis: Axis): PaddedArray;
    function unpadl(a: PaddedArray, axis: Axis): PaddedArray;
    function unpadr(a: PaddedArray, axis: Axis): PaddedArray;

    function axis(): Axis;

    type Float;
    function forall_ix_snippet_padded(u: PaddedArray, v: PaddedArray,
                                      u0: PaddedArray, u1: PaddedArray,
                                      u2: PaddedArray,
                                      c0: Float, c1: Float, c2: Float,
                                      c3: Float, c4: Float): Array;

    function forall_ix_snippet(u: Array, v: Array,
                               u0: Array, u1: Array, u2: Array,
                               c0: Float, c1: Float, c2: Float,
                               c3: Float, c4: Float): Array;
}

concept OFIntroducePaddingInArguments = {
    use OFRewritingTypesAndOps;

    axiom toPaddingOpsRule(u: Array, v: Array,
                           u0: Array, u1: Array, u2: Array,
                           c0: Float, c1: Float, c2: Float,
                           c3: Float, c4: Float) {
        assert forall_ix_snippet(u, v, u0, u1, u2, c0, c1, c2, c3, c4) ==
               forall_ix_snippet_padded(asPadded(u), asPadded(v),
                                        asPadded(u0), asPadded(u1),
                                        asPadded(u2), c0, c1, c2, c3, c4);
    }
}

concept OFAddLeftPaddingInArgumentsGeneralAxis = {
    use OFRewritingTypesAndOps;
    axiom addLeftPaddingInArgumentsGeneral(
            u: PaddedArray, v: PaddedArray, u0: PaddedArray,
            u1: PaddedArray, u2: PaddedArray,
            c0: Float, c1: Float, c2: Float, c3: Float, c4: Float) {
        assert forall_ix_snippet_padded(u, v, u0, u1, u2, c0, c1, c2, c3, c4) ==
               forall_ix_snippet_padded(cpadl(u, axis()), cpadl(v, axis()),
                    cpadl(u0, axis()), cpadl(u1, axis()), cpadl(u2, axis()),
                    c0, c1, c2, c3, c4);
    }
}

concept OFIntroducePaddingRule = {
    use OFRewritingTypesAndOps;
    axiom introducePaddingRule(a: Array) {
        assert a == inner(asPadded(a));
    }
}

concept OFAddLeftPaddingRuleGeneralAxis = {
    use OFRewritingTypesAndOps;
    axiom addLeftPaddingRule(a: PaddedArray) {
        assert inner(a) == inner(cpadl(a, axis()));
    }
}

concept OFAddLeftPadding0Axis =
    OFAddLeftPaddingInArgumentsGeneralAxis[ axis => zero ];
concept OFAddLeftPadding1Axis =
    OFAddLeftPaddingInArgumentsGeneralAxis[ axis => one ];
concept OFAddLeftPadding2Axis =
    OFAddLeftPaddingInArgumentsGeneralAxis[ axis => two ];

concept OFAddRightPadding0Axis = OFAddLeftPadding0Axis[ cpadl => cpadr ];
concept OFAddRightPadding1Axis = OFAddLeftPadding1Axis[ cpadl => cpadr ];
concept OFAddRightPadding2Axis = OFAddLeftPadding2Axis[ cpadl => cpadr ];

// TODO: enforce signature compatibility through satisfactions only
// TODO: do not break procedure argument modes when switching stuff (can not
// make rvalue lvalues)
concept OFRemoveLeftoverPadding = {
    use OFRewritingTypesAndOps;
    axiom removeLeftoverPaddingPaddedArrayRule(a: PaddedArray, axis: Axis) {
        //assert inner(unpadl(a, axis)) == inner(a);
        //assert inner(unpadr(a, axis)) == inner(a);
        assert inner(cpadr(a, axis)) == inner(a);
        assert inner(cpadl(a, axis)) == inner(a);
    }

    axiom removeLeftoverPaddingArrayRule(a: Array) {
        assert inner(asPadded(a)) == a;
    }
}

concept OFLiftCores = {
    use OFRewritingTypesAndOps[ axis => zero ];

    type Nat;

    function nbCores(): Nat;
    function one(): Nat;

    function padAndLift(array: Array, axis: Axis, d: Nat,
                        paddingAmount: Nat): Array;
    function unliftAndUnpad(array: Array, axis: Axis,
                            paddingAmount: Nat): Array;

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
        var i = zero(): Axis;
        var d = nbCores();
        var p = one(): Nat;

        // assert forall_ix_snippet(u, v, u0, u1, u2, c0, c1, c2, c3, c4) ==
        //        unliftAndUnpad(forall_ix_snippet_threaded(
        //            padAndLift(u, i, d, p), padAndLift(v, i, d, p),
        //            padAndLift(u0, i, d, p), padAndLift(u1, i, d, p),
        //            padAndLift(u2, i, d, p), c0, c1, c2, c3, c4, d), i, p);
        assert forall_ix_snippet(u, v, u0, u1, u2, c0, c1, c2, c3, c4) ==
               forall_ix_snippet_threaded(u, v, u0, u1, u2, c0, c1, c2, c3, c4, d);
    }
}

concept OFTile = {
    use OFRewritingTypesAndOps[ axis => zero ];

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

concept OFVectorize = {
    use OFRewritingTypesAndOps[ axis => last_axis ];
}[ last_axis => two ];

concept OFRavel = {
    type Array;
    type Shape;
    type LinearArray;
    type Index;

    function shape(arr: Array): Shape;
    function subshape(ix: Index, shape: Shape): Shape;
    function psi(ix: Index, arr: Array): Array;
    function rav(a: Array): LinearArray;

    function toArray(la: LinearArray, shape: Shape): Array;

    axiom pctRule(ix: Index, a: Array) {
        assert psi(ix, a) == toArray(rav(psi(ix, a)), subshape(ix, shape(a)));
    }
}

concept ShapeIsUnique = {
    type Shape;

    function uniqueShape(): Shape;

    axiom shapeIsUniqueRule(s: Shape) {
        assert s == uniqueShape();
    }
}

concept ONFToArrayGenericBinopRule = {
    type Array;
    type LinearArray;
    type E;
    type Shape;

    function binop(lhs: E, rhs: Array): Array;
    function binop(lhs: Array, rhs: Array): Array;
    function binop(lhs: E, rhs: LinearArray): LinearArray;
    function binop(lhs: LinearArray, rhs: LinearArray): LinearArray;

    function toArray(la: LinearArray, shape: Shape): Array;

    axiom binopArrayRule(lhs: LinearArray, rhs: LinearArray, s: Shape) {
        assert binop(toArray(lhs, s), toArray(rhs, s)) ==
               toArray(binop(lhs, rhs), s);
    }

    axiom binopScalarRule(lhs: E, rhs: LinearArray, s: Shape) {
        assert binop(lhs, toArray(rhs, s)) == toArray(binop(lhs, rhs), s);
    }
}

concept ONFToArrayRules = {
    use ONFToArrayGenericBinopRule[ E => Float
                                  , binop => _+_
                                  , binopArrayRule => binopArrayRulePlus
                                  , binopScalarRule => binopScalarRulePlus
                                  ];
    use ONFToArrayGenericBinopRule[ E => Float
                                  , binop => _*_
                                  , binopArrayRule => binopArrayRuleMul
                                  , binopScalarRule => binopScalarRuleMul
                                  ];
    use ONFToArrayGenericBinopRule[ E => Float
                                  , binop => binary_sub
                                  , binopArrayRule => binopArrayRuleSub
                                  , binopScalarRule => binopScalarRuleSub
                                  ];
}

// Ranges in the PCT are always contiguous
concept PsiCorrespondenceTheorem = {
    //use OFRewritingTypesAndOps;

    type Array;
    type Index;
    type LinearArray;
    type LinearIndex;
    type Range;
    type Shape;
    type Stride;

    function shape(arr: Array): Shape;
    function psi(ix: Index, arr: Array): Array;

    function start(ix: Index, shape: Shape): LinearIndex;
    function stride(ix: Index, shape: Shape): Stride;
    function iota(s: Stride): Range;

    function _+_(lix: LinearIndex, range: Range): Range;
    function _*_(lix: LinearIndex, stride: Stride): LinearIndex;

    function rav(a: Array): LinearArray;
    function elementsAt(la: LinearArray, r: Range): LinearArray;

    axiom pctRule(ix: Index, a: Array) {
        var start = start(ix, shape(a));
        var stride = stride(ix, shape(a));

        assert rav(psi(ix, a)) ==
               elementsAt(rav(a), start * stride + iota(stride));
    }
}

concept OFCleanupIdentity = {
    use signature(PsiCorrespondenceTheorem);
    use signature(ONFToArrayRules);

    function emptyIndex(): Index;
    function uniqueShape(): Shape;

    axiom cleanUpIdentityRule(a: Array) {
        var start = start(emptyIndex(), uniqueShape());
        var stride = stride(emptyIndex(), uniqueShape());

        assert toArray(rav(a), uniqueShape()) == a;
        assert toArray(elementsAt(rav(a), start * stride + iota(stride)),
                       uniqueShape()) == a;
    }
}

// concept OFExtractInnerRule = {
//     use OFRewritingTypesAndOps;
//     type Float;

//     function forall_ix_snippet_padded(u: PaddedArray, v: PaddedArray,
//                                       u0: PaddedArray, u1: PaddedArray,
//                                       u2: PaddedArray,
//                                       c0: Float, c1: Float, c2: Float,
//                                       c3: Float, c4: Float): PaddedArray;

//     function forall_ix_snippet(u: Array, v: Array,
//                                u0: Array, u1: Array, u2: Array,
//                                c0: Float, c1: Float, c2: Float,
//                                c3: Float, c4: Float): Array;

//     axiom extractInnerRule(u: PaddedArray, v: PaddedArray, u0: PaddedArray,
//                            u1: PaddedArray, u2: PaddedArray,
//                            c0: Float, c1: Float, c2: Float, c3: Float,
//                            c4: Float, axis: Axis) {
//         assert forall_ix_snippet(inner(u), inner(v), inner(u0), inner(u1),
//                                  inner(u2), c0, c1, c2, c3, c4) ==
//                inner(forall_ix_snippet_padded(u, v, u0, u1, u2,
//                                               c0, c1, c2, c3, c4));
//     }
// }

// concept OFDistributedPaddingRules = OFPaddingRules[ cpadl => dlcpadl
//                                                   , cpadr => dlcpadr
//                                                   , unpadl => dlunpadl
//                                                   , unpadr => dlunpadr
//                                                   ];

// concept OFExtractInnerRule = {
//     use signature(OFPaddingRules);

//     function inner(pa: PaddedArray): Array;
    // function forall_ix_snippet_padded(u: PaddedArray, v: PaddedArray,
    //                                   u0: PaddedArray, u1: PaddedArray,
    //                                   u2: PaddedArray,
    //                                   c0: Float, c1: Float, c2: Float,
    //                                   c3: Float, c4: Float): PaddedArray;

    // function forall_ix_snippet(u: Array, v: Array,
    //                            u0: Array, u1: Array, u2: Array,
    //                            c0: Float, c1: Float, c2: Float,
    //                            c3: Float, c4: Float): Array;

//     function zero(): Axis;

    // axiom extractInnerRule(u: PaddedArray, v: PaddedArray, u0: PaddedArray,
    //                        u1: PaddedArray,
    //                        u2: Array, c0: Float, c1: Float, c2: Float,
    //                        c3: Float, c4: Float, axis: Axis) {
    //     assert forall_ix_snippet(inner(u), inner(v), inner(u0), inner(u1),
    //                              inner(u2), c0, c1, c2, c3, c4) ==
    //            inner(forall_ix_snippet_padded(u, v, u0, u1, u2,
    //                                           c0, c1, c2, c3, c4));
    // }
// }

// // TODO: add mode to reverse equations
// concept OFUnpaddingRules = {
//     type E;
//     type Array;
//     type Index;

//     type Axis;
//     type Nat;

//     //function nbCores(): Nat;
//     // No PaddedArray, because we do not care about differentiating padding
//     // from content in rewriting rules.
//     function cpadl(a: Array): Array;
//     function cpadr(a: Array): Array;

//     function unpadl(a: Array): Array;
//     function unpadr(a: Array): Array;

//     axiom paddingRule(a: Array) {
//         // left
//         assert unpadl(cpadl(a)) == a;
//         // right
//         assert unpadr(cpadr(a)) == a;
//     }
// }

// concept OFDistributedUnpaddingRules = OFUnpaddingRules[ cpadl => dlcpadl
//                                                       , cpadr => dlcpadr
//                                                       , unpadl => dlunpadl
//                                                       , unpadr => dlunpadr
//                                                       ];

/*
implementation PDE = {
    type E;
    type Array;
    type PaddedArray;

    type Axis;
    type Offset;
    type PaddingAmount;

    require function zero(): Axis;
    require function one(): Offset;
    require function -_(o: Offset): Offset;
    require function rotate(axis: Axis, o: Offset, a: Array): Array;
    require function _+_(lhs: Array, rhs: Array): Array;

    function step(a: Array): Array =
        rotate(zero(), one(), a) + rotate(zero(), -one(), a);

    function inner(pa: PaddedArray): Array;
    function cpad_n(axis: Axis, n: PaddingAmount, a: Array): PaddedArray;
}[ PaddedArray => Array ];

concept PDERewritings = {
    require signature(PDE);

    require function binaryElementwiseOp(lhs: Array, rhs: Array): Array;
    require predicate _<=_(o: Offset, p: PaddingAmount);


}[ binaryElementwiseOp => _+_ ];

concept PDEPaddingRewritings = {
    require signature(PDE); //[ PaddedArray => Array ];

    require function binaryElementwiseOp(lhs: Array, rhs: Array): Array;
    require predicate _<=_(o: Offset, p: PaddingAmount);
    require function paddingAmount(): PaddingAmount; // problem! this is not part of the PDE API! What do we do about it?

    axiom paddingRewriteBinaryElementwiseOp(lhs: Array,
                                            rhs: Array) {
                                            //n: PaddingAmount, axis: Axis) {
        var n = paddingAmount(); var axis = zero();
        assert binaryElementwiseOp(lhs, rhs) == binaryElementwiseOp(inner(cpad_n(axis, n, lhs)), inner(cpad_n(axis, n, rhs)));
        assert binaryElementwiseOp(inner(lhs), inner(rhs)) == inner(binaryElementwiseOp(lhs, rhs));
    }

    axiom paddingRewriteRotate(array: Array, axis: Axis, offset: Offset) {
        var n = paddingAmount();
        assert offset <= n => inner(cpad_n(axis, n, rotate(axis, offset, array))) == inner(rotate(axis, offset, cpad_n(axis, n, array)));
    }
}[ binaryElementwiseOp => _+_ ];
*/
