package tests.parse.inputs.syntaxTests;

// line comment
/* block comment */
/*< nested block comment /*< /*< /*< >*/ >*/ >*/ >*/

implementation TestSymbolicOperators = {
    type T;
    
    // Definitions
    // unary ops
    function +_(t: T): T;
    function -_(t: T): T;
    function !_(t: T): T;

    // mult ops
    function _*_(t1: T, t2: T): T;
    function _/_(t1: T, t2: T): T;
    function _%_(t1: T, t2: T): T;
    
    // add ops
    function _+_(t1: T, t2: T): T;
    function _-_(t1: T, t2: T): T;
    
    // shift ops
    function _>>_(t1: T, t2: T): T;
    function _<<_(t1: T, t2: T): T;
    
    // range
    function _.._(t1: T, t2: T): T;
    
    // comparison ops
    function _<_(t1: T, t2: T): T;
    function _>_(t1: T, t2: T): T;
    function _<=_(t1: T, t2: T): T;
    function _>=_(t1: T, t2: T): T;
    function _==_(t1: T, t2: T): T;
    function _!=_(t1: T, t2: T): T;
    function _===_(t1: T, t2: T): T;
    function _!==_(t1: T, t2: T): T;

    // logical and
    function _&&_(t1: T, t2: T): T;
    
    // logical or
    function _||_(t1: T, t2: T): T;
    
    // logical implications
    function _=>_(t1: T, t2: T): T;
    function _<=>_(t1: T, t2: T): T;

    // Calls
    procedure use_symbols(obs t: T) = {
        +t; -t; !t; ~t;         // unary ops
        t * t; t / t; t % t;    // mult ops
        t + t; t - t;           // add ops
        t >> t; t << t;         // shift ops
        t .. t;                 // range
        t < t; t > t; t <= t;   // comparison ops
        t >= t; t == t; t != t;
        t === t; t !== t;
        t && t;                 // logical and
        t || t;                 // logical or
        t => t; t <=> t;        // logical implications
    }

    // TODO: perform precedence tests somewhere
}

// WIP satisfaction tests
// Satisfaction using the models keyword
satisfaction S1 = {} with {} models {};

// Satisfaction using the approximates keyword
satisfaction S2 = {} with {} approximates {};
