mgn := cabal exec -v0 magnoliac --
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

parse-test-targets = package-name-with-hyphens \
		     syntaxTests

check-test-targets = declScopeTests \
		     externalTests \
		     modeTests \
		     moduleTests \
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

tmp-tests-py: $(self-contained-codegen-test-targets:%=tmp-test-py-%)

tmp-test-py-%: tests/self-contained-codegen/inputs/%.mg
	$(mgn) test --pass self-contained-codegen --backend python $<

tests-pass: $($(pass)-test-targets:%=check-output-%)

update-tests-pass: $($(pass)-test-targets:%=update-output-%)

check-output-%: tests/$(pass)/inputs/%.mg tests/$(pass)/outputs/%.mg.out build
	$(eval tempfile := $(shell mktemp))
	$(mgn) test --pass $(pass) --output-directory=$(tests-out-dir) $< > $(tempfile)
	@diff $(tempfile) $(word 2, $^) > /dev/null && echo "OK" || (echo "Output is not matching for test file $<"; rm $(tempfile); exit 1)
	@rm $(tempfile)

update-output-%: tests/$(pass)/inputs/%.mg build
	$(mgn) test --pass $(pass) --output-directory=$(tests-out-dir) $< > tests/$(pass)/outputs/$(notdir $<).out

example-names = bgl \
                containers \
                fizzbuzz \
                while_loop

# TODO: add also CUDA examples here
build-examples: build-examples-cpp build-examples-py

build-examples-cpp: $(example-names:%=build-example-cpp-%)
build-examples-py:  $(example-names:%=build-example-py-%)

build-example-cpp-%: examples/% build
	$(eval example-name := $(notdir $<))
	$(mgn) build --output-directory $</cpp-src/gen --backend cpp --base-import-directory gen --allow-overwrite $</mg-src/$(example-name)-cpp.mg
	make -C $<
	@# TODO(bchetioui): add hashtree tool to check generation is a noop
	@# TODO(bchetioui): add to CI

build-example-cuda-%: examples/% build
	$(eval example-name := $(notdir $<))
	$(mgn) build --output-directory $</cuda-src/gen --backend cuda --base-import-directory gen --allow-overwrite $</mg-src/$(example-name)-cuda.mg
	make cuda -C $<

build-example-py-%: examples/% build
	$(eval example-name := $(notdir $<))
	$(mgn) build --output-directory $</py-src/gen --backend python --base-import-directory gen --allow-overwrite $</mg-src/$(example-name)-py.mg
