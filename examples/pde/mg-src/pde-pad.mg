package examples.pde.mg-src.pde-pad;

concept OFPad = {
  type Array;
  type Float;

  procedure refillPadding(upd a: Array);

  function schedulePadded(u: Array, v: Array,
    u0: Array, u1: Array, u2: Array): Array;

  function schedule(u: Array, v: Array,
                    u0: Array, u1: Array, u2: Array): Array;

  axiom padRule(u: Array, v: Array, u0: Array, u1: Array, u2: Array) {
    assert schedule(u, v, u0, u1, u2) ==
      { var result = schedulePadded(u, v, u0, u1, u2);
        call refillPadding(result);
        value result;
      };
  }

  type Index;
  type Axis;
  type Offset;
  function rotateIx(ix: Index, axis: Axis, offset: Offset): Index;
  function rotateIxPadded(ix: Index, axis: Axis, offset: Offset)
    : Index;

  axiom rotateIxPadRule(ix: Index, axis: Axis, offset: Offset) {
    assert rotateIx(ix, axis, offset) ==
           rotateIxPadded(ix, axis, offset);
  }
}

concept OFSpecializeSubstepGenerator = {
  type Index;
  type Array;
  type Float;
  type ScalarIndex;

  function mkIx(i: ScalarIndex, j: ScalarIndex, k: ScalarIndex)
      : Index;

  function substepIx(u: Array, v: Array, u0: Array,
              u1: Array, u2: Array, ix: Index): Float;

  function substepIx3D(u: Array, v: Array, u0: Array,
              u1: Array, u2: Array, i: ScalarIndex, j: ScalarIndex,
              k: ScalarIndex): Float;

  axiom specializeSubstepRule(u: Array, v: Array, u0: Array,
              u1: Array, u2: Array, i: ScalarIndex, j: ScalarIndex,
              k: ScalarIndex) {
    assert substepIx3D(u, v, u0, u1, u2, i, j, k) ==
           substepIx(u, v, u0, u1, u2, mkIx(i, j, k));
  }
};

concept OFSpecializePsi = {
  type Index;
  type Array;
  type E;
  type ScalarIndex;

  /* 3D index projection functions */
  function ix0(ix: Index): ScalarIndex;
  function ix1(ix: Index): ScalarIndex;
  function ix2(ix: Index): ScalarIndex;

  /* 3D index constructor */
  function mkIx(i: ScalarIndex, j: ScalarIndex, k: ScalarIndex)
    : Index;

  function psi(ix: Index, array: Array): E;
  function psi(i: ScalarIndex, j: ScalarIndex, k: ScalarIndex,
               array: Array): E;

  axiom specializePsiRule(ix: Index, array: Array) {
    assert psi(ix, array) == psi(ix0(ix), ix1(ix), ix2(ix), array);
  }

  axiom reduceMakeIxRule(i: ScalarIndex, j: ScalarIndex,
                         k: ScalarIndex) {
    var ix = mkIx(i, j, k);
    assert ix0(ix) == i;
    assert ix1(ix) == j;
    assert ix2(ix) == k;
  }
}[ E => Float ];

concept OFReduceMakeIxRotate = {
  use signature(OFSpecializePsi);

  type Axis;
  type Offset;

  function zero(): Axis;
  function one(): Axis;
  function two(): Axis;

  function rotateIx(ix: Index, axis: Axis, offset: Offset): Index;

  type AxisLength;

  function shape0(): AxisLength;
  function shape1(): AxisLength;
  function shape2(): AxisLength;

  function _+_(six: ScalarIndex, o: Offset): ScalarIndex;
  function _%_(six: ScalarIndex, sc: AxisLength): ScalarIndex;

  axiom reduceMakeIxRotateRule(i: ScalarIndex, j: ScalarIndex,
      k: ScalarIndex, array: Array, o: Offset) {
    var ix = mkIx(i, j, k);
    var s0 = shape0();
    var s1 = shape1();
    var s2 = shape2();

    assert ix0(rotateIx(ix, zero(), o)) == (i + o) % s0;
    assert ix0(rotateIx(ix, one(), o)) == i;
    assert ix0(rotateIx(ix, two(), o)) == i;

    assert ix1(rotateIx(ix, zero(), o)) == j;
    assert ix1(rotateIx(ix, one(), o)) == (j + o) % s1;
    assert ix1(rotateIx(ix, two(), o)) == j;

    assert ix2(rotateIx(ix, zero(), o)) == k;
    assert ix2(rotateIx(ix, one(), o)) == k;
    assert ix2(rotateIx(ix, two(), o)) == (k + o) % s2;
  }
}

// We suppose here that the amount of padding is sufficient across each axis
// for every indexing operation.
concept OFEliminateModuloPadding = {
  use signature(OFReduceMakeIxRotate);

  type Array;
  type Float;

  function psi(i: ScalarIndex, j: ScalarIndex, k: ScalarIndex,
               a: Array): Float;

  axiom eliminateModuloPaddingRule(i: ScalarIndex, j: ScalarIndex,
      k: ScalarIndex, a: Array, o: Offset) {
    var s0 = shape0();
    var s1 = shape1();
    var s2 = shape2();

    assert psi((i + o) % s0, j, k, a) == psi(i + o, j, k, a);
    assert psi(i, (j + o) % s1, k, a) == psi(i, j + o, k, a);
    assert psi(i, j, (k + o) % s2, a) == psi(i, j, k + o, a);
  }
}