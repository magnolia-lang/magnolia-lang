mgn := cabal exec -v0 magnolia --
pass := check
tests-out-dir := gen

build:
	cabal new-build

install:
	cabal new-install --overwrite-policy=always

clean:
	cabal new-clean

lint:
	hlint src

parse-test-targets = commentTests \
                     package-name-with-hyphens

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

self-contained-codegen-test-targets = externalTests \
                                      generalTranslationTests

tests: tests-all-passes # do not run python tests for convenience

tests-all: tests-all-passes tests-frontend

tests-frontend:
	pytest -v tests/frontend/test_frontend.py

tests-all-passes:
	for pass in parse check self-contained-codegen ; do \
		make tests-pass pass=$$pass ; \
	done

update-tests:
	for pass in parse check self-contained-codegen ; do \
		make update-tests-pass pass=$$pass ; \
	done

tests-pass: $($(pass)-test-targets:%=check-output-%)

update-tests-pass: $($(pass)-test-targets:%=update-output-%)

check-output-%: tests/$(pass)/inputs/%.mg tests/$(pass)/outputs/%.mg.out build
	$(eval tempfile := $(shell mktemp))
	$(mgn) test --pass $(pass) --output-directory=$(tests-out-dir) $< > $(tempfile)
	@diff $(tempfile) $(word 2, $^) > /dev/null && echo "OK" || (echo "Output is not matching for test file $<"; rm $(tempfile); exit 1)
	@rm $(tempfile)

update-output-%: tests/$(pass)/inputs/%.mg build
	$(mgn) test --pass $(pass) --output-directory=$(tests-out-dir) $< > tests/$(pass)/outputs/$(notdir $<).out

example-names = fizzbuzz

build-examples: $(example-names:%=build-example-%)

build-example-%: examples/% build
	$(eval example-name := $(notdir $<))
	$(mgn) build --output-directory $</cpp-src/gen --base-import-directory gen --allow-overwrite $</mg-src/$(example-name).mg
	make -C $<
	# TODO(bchetioui): add hashtree tool to check generation is a noop
	# TODO(bchetioui): add to CI
