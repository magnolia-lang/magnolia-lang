package examples.bgl.mg-src.v2.list;

concept List = {
    require type A;
    type List;

    function empty(): List;
    function cons(a: A, l: List): List;

    function head(l: List): A guard !isEmpty(l);
    function tail(l: List): List guard !isEmpty(l);

    predicate isEmpty(l: List);

    axiom isEmptyBehavior(a: A, l: List) {
        assert isEmpty(empty());
        assert !isEmpty(cons(a, l));
    }

    axiom headTailBehavior(a: A, l: List) {
        assert head(cons(a, l)) == a;
        assert tail(cons(a, l)) == l;
    }
}
