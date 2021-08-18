# The Magnolia Programming Language

![Continuous
integration](https://github.com/magnolia-lang/magnolia-lang/actions/workflows/ci.yml/badge.svg)

Magnolia is a research programming language based on the theory of institutions.

⚠️ The compiler is still at an experimental stage, and not yet fully functional.
Do not rely on anything to remain stable for the moment. ⚠️

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

⚠️  The repl feature, while functional, is outdated, and likely on the way to
being deprecated. ⚠️ 

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

Feel free to open issues and pull requests!

### Style guide

Very few guidelines here at the moment. We can import an existing style guide
should the need arise. Otherwise:

* try to keep your lines under 80 characters;
* make sure that `hlint src` outputs "No hints" when run at the root of the
  repository.
