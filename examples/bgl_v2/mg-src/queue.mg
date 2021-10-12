package examples.bgl_v2.mg-src.queue
    imports examples.bgl_v2.mg-src.property_map
          , examples.bgl_v2.mg-src.tuple;

concept Queue = {
    require type A;
    type Queue;

    predicate isEmpty(q: Queue);

    procedure push(obs a: A, upd q: Queue);
    procedure pop(upd q: Queue) guard !isEmpty(q);
    function front(q: Queue): A;
}

concept FIFOQueue = {
    use Queue[ Queue => FIFOQueue ];
    function empty(): FIFOQueue;

    axiom pushPopBehavior(a: A, inq: FIFOQueue) {
        var mut_inq = inq;
        call push(a, mut_inq);
        assert front(mut_inq) == a;
        call pop(mut_inq);
        assert inq == mut_inq;
    }
}

concept UpdateablePriorityQueue = {
    require type A;
    require type Priority;
    type PriorityQueue;
    
    require ReadPropertyMap[ Key => A
                           , Value => Priority
                           , PropertyMap => PriorityMap
                           ];

    use Queue[ Queue => PriorityQueue ];
    
    function empty(pm: PriorityMap): PriorityQueue;
    function update(pm: PriorityMap, a: A, pq: PriorityQueue): PriorityQueue;
}
