# The Magnolia Programming Language

![Continuous
integration](https://github.com/magnolia-lang/magnolia-lang/actions/workflows/ci.yml/badge.svg)

Magnolia is a research programming language based on the theory of institutions.

⚠️  The compiler is still at an experimental stage. Expect bugs, sharp edges, and
changing APIs. ⚠️

## Getting started

### How to install

To install Magnolia, simply run the following command:

```bash
make install
```

`magnolia` will then be installed in the default directory for your cabal
binaries. By default, this should be `~/.cabal/bin`. Make sure that directory
is in your `PATH`.

### How to build Magnolia programs

To build a Magnolia program, use the `build` command as follows:

```bash
magnolia build <path/to/package.mg> --output-directory <path/to/generated/files>
```

By default, this command does **NOT** overwrite files. In order to re-generate
previously generated programs in the same output directory, you can explicitly
add the `--allow-overwrite` flag to your command.

### Using the REPL

⚠️  The repl feature, while functional, is outdated, and likely on the way to
being deprecated. ⚠️

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

## For developers

### How to compile

The compiler has been tested with GHC 8.10.3 and is built using `cabal`.
A very simplistic Makefile is provided, and the following should be sufficient
to get going with a development version of the compiler:

```bash
make build
# For convenience, you may define an alias for the compiler
alias magnolia='cabal exec magnolia --'
```

## How to contribute

Feel free to open issues and pull requests!

### Style guide

We roughly follow the same guidelines as specified in the [Futhark
project](https://github.com/diku-dk/futhark). Most importantly:

* lines should (as much as possible) be under 80 characters long;
* we use 2 spaces for indent.
