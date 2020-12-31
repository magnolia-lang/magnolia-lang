build:
	cabal new-build

clean:
	cabal new-clean

lint:
	hlint src

add:
	find src -name "*.hs" | xargs git add
