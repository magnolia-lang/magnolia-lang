mgn := cabal exec -v0 magnolia --
pass := check

build:
	cabal new-build

clean:
	cabal new-clean

lint:
	hlint src

parse-test-targets = package-name-with-hyphens

check-test-targets = declScopeTests \
		     externalTests \
		     modeTests \
		     packageTests \
		     parsingTests \
		     regressionTests \
		     renamingTests \
		     requirementTests \
	    	     stateHandlingTests \
		     stmtTests

self-contained-codegen-test-targets = basic

tests-all:
	for pass in parse check self-contained-codegen ; do \
		make tests pass=$$pass ; \
	done

update-tests-all:
	for pass in parse check self-contained-codegen ; do \
                make update-tests pass=$$pass ; \
        done

tests: $($(pass)-test-targets:%=check-output-%)

update-tests: $($(pass)-test-targets:%=update-output-%)

check-output-%: tests/$(pass)/inputs/%.mg tests/$(pass)/outputs/%.mg.out build
	$(eval tempfile := $(shell mktemp))
	$(mgn) test --pass $(pass) $< > $(tempfile)
	@diff $(tempfile) $(word 2, $^) > /dev/null && echo "OK" || (echo "Output is not matching for test file $<"; rm $(tempfile); exit 1)
	@rm $(tempfile)

update-output-%: tests/$(pass)/inputs/%.mg build
	$(mgn) test --pass $(pass) $< > tests/$(pass)/outputs/$(notdir $<).out
