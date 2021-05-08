# The Magnolia Programming Language

Magnolia is a research programming language based on the theory of institutions.

## How to compile

The compiler has been tested with GHC 8.10.3 and is built using `cabal`.
A very simplistic Makefile is provided, and the following should be sufficient
to get going:

```bash
make build
# For convenience, you may define an alias for the compiler
alias magnolia='cabal exec magnolia --'
```

## How to build Magnolia programs

Right now, code generation is very much incomplete; the output is pretty much
useless. However, running

```bash
magnolia build <path/to/package.mg>
```

will go through the compilation steps, and will output typechecking/parsing
errors.

Another option to explore Magnolia source files is to explore them using the
`repl` option instead:

```bash
magnolia repl
mgn> help
Available commands are:
        help: show this help menu
	inspectm: inspect a module
	inspectp: inspect a package
	list: list the loaded packages
	load: load a package
	reload: reload a package
```

For example, given a file named `"example.mg"` at the root of the repo, one can
load it and inspect its contents like so:

```bash
mgn> load example
mgn> inspectp example
SomeSig
SomeConcept
SomeOtherConcept
```

## How to contribute

Feel free to open issues and pull requests! A brief overview of implemented and
unimplemented features is available below. If you submit a pull request for one
of the features marked as missing, please also update the README.

If more details are necessary, feel free to ping me (bchetioui) directly and
we can discuss them.

### Feature overview

#### Missing features

- [ ] Satisfactions handling
- [ ] Extensive testing of the compiler
- [ ] Code generation
- [ ] Documentation (*the code base is small, so it is not yet critical*)
- [ ] Distinction between "normal" blocks and value blocks
- [ ] Assignment out of the AST, and related type inference

#### Partially implemented features

Since the language design is still evolving, nothing is fully implemented. The
following features are partially supported:

- [ ] Error diagnostics (*can be improved by improving error messages, and error
  locs*)
- [ ] Type and consistency checking (*some edge cases are not handled, left as
  TODOs; error recovery can be added in some places*)
- [ ] Imports (*the boundary between file names and package names could be
  clarified; fully qualified name resolution is also not really supported yet*)
- [ ] References to modules, renamings (*some module casts are still
  unimplemented, e.g. "concept X = I" where I is an implementation is not yet
  handled*)

In addition, there are a number of TODOs left in the codebase. I have not done
a great job at writing them so as to be directly exploitable without reading
the code in more details, but they should give a fairly good idea of where we
are at.

Some invariants could also be enforced better across the codebase, e.g. by
consistently using NonEmpty lists in cases when we should always find a match.

### Style guide

Very few guidelines here at the moment. We can import an existing style guide
should the need arise. Otherwise:

* try to keep your lines under 80 characters;
* make sure that `hlint src` outputs "No hints" when run at the root of the
  repository.
