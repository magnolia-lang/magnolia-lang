package examples.pde.mg-src.array-api;

concept ArrayAPI = {
    type Array;
    type E;

    type Axis;
    type Index;
    type Offset;

    /* Scalar-Scalar operations */
    function _+_(lhs: E, rhs: E): E;
    function _-_(lhs: E, rhs: E): E;
    function _*_(lhs: E, rhs: E): E;
    function _/_(lhs: E, rhs: E): E;

    /* Scalar-Array operations */
    function _+_(lhs: E, rhs: Array): Array;
    function _-_(lhs: E, rhs: Array): Array;
    function _*_(lhs: E, rhs: Array): Array;
    function _/_(lhs: E, rhs: Array): Array;

    /* Array-Array operations */
    function _+_(lhs: Array, rhs: Array): Array;
    function _-_(lhs: Array, rhs: Array): Array;
    function _*_(lhs: Array, rhs: Array): Array;
    function _/_(lhs: Array, rhs: Array): Array;

    /* Rotation (theta) */
    function rotate(array: Array, axis: Axis, offset: Offset): Array;

    /* Indexing (psi) */
    function psi(ix: Index, array: Array): E;
}

concept ArrayAPI_ArithmeticAxioms = {
  require ArrayAPI;

  /* Scalar-Array Axioms */
  axiom scalarBinaryPlusAxiom(lhs: E, rhs: Array, ix: Index) {
    assert psi(ix, lhs + rhs) == lhs + psi(ix, rhs);
  }

  axiom scalarBinarySubAxiom(lhs: E, rhs: Array, ix: Index) {
    assert psi(ix, lhs - rhs) == lhs - psi(ix, rhs);
  }

  axiom scalarMulAxiom(lhs: E, rhs: Array, ix: Index) {
    assert psi(ix, lhs * rhs) == lhs * psi(ix, rhs);
  }

  axiom scalarDivAxiom(lhs: E, rhs: Array, ix: Index) {
    assert psi(ix, lhs / rhs) == lhs / psi(ix, rhs);
  }

  /* Array-Array Axioms */
  axiom arrayBinaryPlusAxiom(lhs: Array, rhs: Array, ix: Index) {
    assert psi(ix, lhs + rhs) == psi(ix, lhs) + psi(ix, rhs);
  }

  axiom arrayBinarySubAxiom(lhs: Array, rhs: Array, ix: Index) {
    assert psi(ix, lhs - rhs) == psi(ix, lhs) - psi(ix, rhs);
  }

  axiom arrayMulAxiom(lhs: Array, rhs: Array, ix: Index) {
    assert psi(ix, lhs * rhs) == psi(ix, lhs) * psi(ix, rhs);
  }

  axiom arrayDivAxiom(lhs: Array, rhs: Array, ix: Index) {
    assert psi(ix, lhs / rhs) == psi(ix, lhs) / psi(ix, rhs);
  }
}
