package examples.mmult.mg-src.mmult;

implementation ImplMatrixMult = external C++ base.matrix {

    // require type IntLike;
    require type Element;

    type Size;

    type Matrix;

    procedure printMatrix(obs a: Matrix);
    function create_matrix(row: Size, col: Size): Matrix;
    function zeros(row: Size, col: Size): Matrix;
    function rand_matrix(row: Size, col: Size, upper_bound: Size): Matrix;

    function mmult(a: Matrix, b: Matrix): Matrix;
}

signature IntLikeOps = {
    type IntLike;

    function zero(): IntLike;
    function one(): IntLike;
    function add(a: IntLike, b: IntLike):IntLike;
    function mult(a: IntLike, b: IntLike): IntLike;
    predicate isLowerThan(a: IntLike, b: IntLike);
}

implementation Int32Utils = external C++ base.int32_utils
    IntLikeOps[IntLike => Int32];

program MatrixProgram = {

    use Int32Utils;
    use ImplMatrixMult[Element => Int32];

}