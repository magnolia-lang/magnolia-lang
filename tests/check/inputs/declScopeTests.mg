package tests.check.inputs.declScopeTests;

signature S = {
    /* A Magnolia signature can contain type declarations, as well as
       prototypes for functions, procedures, and predicates. Any other
       type of declaration is an error.
     */
    
    // Valid declarations
    type T;
    function funProto(): T;
    procedure procProto();
    predicate predProto();

    // Invalid declarations
    function funBody(): T = funProto();
    procedure procBody() = call procProto();
    predicate predBody() = predProto();
    axiom axiomBody() = {};
};

concept C = {
    /* A Magnolia concept can contain type declarations, axiom definitions,
       and prototypes for functions, procedures, and predicates. Any other
       type of declaration is an error.
     */

    // Valid declarations
    type T; 
    function funProto(): T;
    procedure procProto();
    predicate predProto();
    axiom axiomBody() = {};

    // Invalid declarations
    function funBody(): T = funProto();
    procedure procBody() = call procProto();
    predicate predBody() = predProto();
};

implementation I = {
    /* A Magnolia implementation can contain anything, except for axioms.
     */

    // Valid declarations
    type T;
    function funProto(): T;
    procedure procProto();
    predicate predProto();
    function funBody(): T = funProto();
    procedure procBody() = call procProto();
    predicate predBody() = predProto();
    
    // Invalid declarations
    axiom axiomBody() = {};
};

implementation IExt = external C++ Ext {
    type U;
}

program P = {
    /* A Magnolia program can contain anything, except for axioms.
     * However, callables *MUST* eventually be given a body.
     */

    // Valid declarations
    type T; // errors without external definition, but is valid in itself
    function funProto(): T; // errors without body, but is valid in itself
    procedure procProto();  // errors without body, but is valid in itself
    predicate predProto();  // errors without body, but is valid in itself
    function funBody(): T = funProto();
    procedure procBody() = call procProto();
    predicate predBody() = predProto();
    // Requiring a type is fine, as long as we provide an external
    // implementation (here, it comes from IExt).
    require type U;
    use IExt;

    // Invalid declarations
    axiom axiomBody() = {}; // TODO: unimplemented error should be removed
};
