package examples.bgl.mg-src.stack;

concept Stack = {
    require type A;
    type Stack;

    function empty(): Stack;
    predicate isEmpty(s: Stack);

    procedure pop(upd s: Stack) guard !isEmpty(s);
    procedure push(obs a: A, upd s: Stack);
    function top(s: Stack): A guard !isEmpty(s);

    axiom pushPopTopBehavior(s: Stack, a: A) {
        var mut_s = s;
        call push(a, mut_s);
        assert top(mut_s) == a;

        call pop(mut_s);
        assert mut_s == s;
    }

    axiom emptyStackIsEmpty() {
        assert isEmpty(empty());
    }
}
