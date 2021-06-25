# Guidelines on testing

**Tests should target specific features and their behavior.**

Naming convension: `<descriptiveTestName>Tests.mg`

Current parts of the compiler that have related tests:

- `Check.hs`:
  - `declarationScopeTests.mg`

  - `modeTests.mg`

  - `stateHandlingTests.mg`

  - `stmtTests.mg`

All error messages with (some) test coverage :

- [ ] AmbiguousFunctionRefErr
- [ ] AmbiguousProcedureRefErr
- [ ] AmbiguousTopLevelRefErr
- [x] CompilerErr
- [ ] CyclicCallableErr
- [ ] CyclicModuleErr
- [ ] CyclicNamedRenamingErr
- [ ] CyclicPackageErr
- [ ] DeclContextErr
- [x] InvalidDeclErr
- [ ] MiscErr
- [x] ModeMismatchErr
- [ ] NotImplementedErr
- [ ] ParseErr
- [ ] TypeErr
- [ ] UnboundFunctionErr
- [ ] UnboundNameErr
- [ ] UnboundProcedureErr
- [ ] UnboundTopLevelErr
- [ ] UnboundTypeErr
- [ ] UnboundVarErr
