package examples.bgl_v2.mg-src.list;

concept List = {
    require type A;
    type List;

    function empty(): List;
    procedure cons(obs a: A, upd l: List);

    function head(l: List): A guard !isEmpty(l);
    procedure tail(upd l: List) guard !isEmpty(l);

    predicate isEmpty(l: List);

    /*
    axiom isEmptyBehavior(a: A, l: List) {
        var mutableList = l;
        assert isEmpty(empty());
        call cons(a, mutableList);
        assert !isEmpty(mutableList);
    }

    axiom headTailBehavior(a: A, l: List) {
        var mutableList = l;
        call cons(a, mutableList);
        assert head(mutableList) == a;
        call tail(mutableList);
        assert l == mutableList;
    }*/
}

concept Iterator = {
    type Iterable;
    type Iterator;
    require type A;

    function iterBegin(itb: Iterable): Iterator;
    procedure iterNext(upd itr: Iterator);
    function iterEnd(itb: Iterable): Iterator;

    function iterUnpack(itr: Iterator): A;
}

concept IterableList = {
    use Iterator[ Iterable => List
                , Iterator => ListIterator
                , A => A
                ];
    use List;
}
