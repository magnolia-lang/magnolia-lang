package examples.bgl_v2.mg-src.stack;

concept Stack = {
    require type A;
    type Stack;

    function empty(): Stack;
    predicate isEmpty(s: Stack);

    function pop(s: Stack): Stack guard !isEmpty(s);
    function push(a: A, s: Stack): Stack;
    function top(s: Stack): A guard !isEmpty(s);

    axiom pushPopTopBehavior(s: Stack, a: A) {
        assert pop(push(a, s)) == s;
        assert top(push(a, s)) == a;
    }

    axiom emptyStackIsEmpty() {
        assert isEmpty(empty());
    }
}
