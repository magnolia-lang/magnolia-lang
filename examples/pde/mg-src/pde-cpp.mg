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
        call forall_snippet_ix(u, v, u0, u1, u2, c0, c1, c2, c3, c4);
    }

    require procedure forall_snippet_ix(upd u: Array, obs v: Array,
                                        obs u0: Array, obs u1: Array, obs u2: Array,
                                        obs c0: Float, obs c1: Float, obs c2: Float,
                                        obs c3: Float, obs c4: Float);
    type Index;

    require function psi(ix: Index, array: Array): Float;
    require procedure set(obs ix: Index, upd array: Array, obs v: Float);

    procedure snippet_ix(upd u: Array, obs v: Array, obs u0: Array,
                         obs u1: Array, obs u2: Array, obs c0: Float,
                         obs c1: Float, obs c2: Float, obs c3: Float,
                         obs c4: Float, obs ix: Index) {
        var zero = zero();
        var one = one(): Offset;

        var result = psi(ix, u + c4 * ( c3 * (((c1 * rotate(v, zero, -one) - c2 * u0) + c1 * rotate(v, zero, one)) + ((c1 * rotate(v, one(): Axis, -one) - c2 * u0) + c1 * rotate(v, one(): Axis, one)) + c1 * rotate(v, one(): Axis, one) + ((c1 * rotate(v, two(): Axis, -one) - c2 * u0) + c1 * rotate(v, two(): Axis, one))) - (u0 * ((-c0 * rotate(v, zero, -one)) + c0 * rotate(v, zero, one)) + u1 * ((-c0 * rotate(v, one(): Axis, -one)) + c0 * rotate(v, one(): Axis, one)) + u2 * ((-c0 * rotate(v, two(): Axis, -one)) + c0 * rotate(v, two(): Axis, one)))));

        call set(ix, u, result);
    }
}

program PDEProgram = {
    use PDE[-_ => unary_sub, _-_ => binary_sub];
    use ExtArrayOps[ one_float => one
                   , two_float => two
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

    require procedure snippet_ix(upd u: Array, obs v: Array,
                                 obs u0: Array, obs u1: Array, obs u2: Array,
                                 obs c0: Float, obs c1: Float, obs c2: Float,
                                 obs c3: Float, obs c4: Float, obs ix: Index);

    procedure forall_snippet_ix(upd u: Array, obs v: Array,
                                obs u0: Array, obs u1: Array, obs u2: Array,
                                obs c0: Float, obs c1: Float, obs c2: Float,
                                obs c3: Float, obs c4: Float);

}

implementation ExtArrayOps = external C++ base.array_ops {
    type Float;
    type Array;
    type Offset;
    type Axis;
    type Index;

    function psi(ix: Index, array: Array): Float;
    procedure set(obs ix: Index, upd array: Array, obs v: Float);

    /* Float ops */
    function unary_sub(f: Float): Float;
    function _+_(lhs: Float, rhs: Float): Float;
    function binary_sub(lhs: Float, rhs: Float): Float; // TODO: look at bug here
    function _*_(lhs: Float, rhs: Float): Float;
    function _/_(num: Float, den: Float): Float;
    function one_float(): Float;
    function two_float(): Float;

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
    // TODO:
    // TODO: R3
}

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