mgn := cabal exec -v0 magnolia --

build:
	cabal new-build

clean:
	cabal new-clean

lint:
	hlint src

test-targets = declScopeTests \
		externalTests \
		modeTests \
		packageTests \
		parsingTests \
		renamingTests \
		requirementTests \
	       	stateHandlingTests \
		stmtTests

tests: $(test-targets:%=check-output-%)

update-tests: $(test-targets:%=update-output-%)

check-output-%: tests/inputs/%.mg tests/outputs/%.mg.out build
	$(eval tempfile := $(shell mktemp))
	$(mgn) build $< > $(tempfile)
	@diff $(tempfile) $(word 2, $^) > /dev/null && echo "OK" || (echo "Output is not matching for test file $<"; rm $(tempfile); exit 1)
	@rm $(tempfile)

update-output-%: tests/inputs/%.mg build
	$(mgn) build $< > tests/outputs/$(notdir $<).out
