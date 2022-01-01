.PHONY: help test docs clean distclean devrepl
.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
    match = re.match(r'^([a-z0-9A-Z_-]+):.*?## (.*)$$', line)
    if match:
        target, help = match.groups()
        print("%-20s %s" % (target, help))
print("""
Instead of "make test", consider "make devrepl" if you want to run the test
suite or generate the docs repeatedly.

Make sure you have Revise.jl installed in your standard Julia environment
""")
endef
export PRINT_HELP_PYSCRIPT

help:  ## show this help
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)


# We want to test against a checkout of QuantumPropagators
QUANTUMPROPAGATORS ?= ../QuantumPropagators.jl

define DEV_PACKAGES
using Pkg;
Pkg.develop(path="$(QUANTUMPROPAGATORS)");
endef
export DEV_PACKAGES

define ENV_PACKAGES
$(DEV_PACKAGES)
Pkg.develop(PackageSpec(path=pwd()));
Pkg.instantiate()
endef
export ENV_PACKAGES

JULIA ?= julia


Manifest.toml: Project.toml $(QUANTUMPROPAGATORS)/Project.toml
	$(JULIA) --project=. -e "$$DEV_PACKAGES;Pkg.instantiate()"


test:  test/Manifest.toml  ## Run the test suite
	$(JULIA) --project=test --threads auto --color=auto --startup-file=yes --code-coverage="user" --depwarn="yes" --check-bounds="yes" -e 'include("test/runtests.jl")'
	@echo "Done. Consider using 'make devrepl'"


test/Manifest.toml: test/Project.toml
	$(JULIA) --project=test -e "$$ENV_PACKAGES"


devrepl: test/Manifest.toml ## Start an interactive REPL for testing and building documentation
	@$(JULIA) --project=test --banner=no --startup-file=yes -e 'include("test/init.jl")' -i


docs: test/Manifest.toml ## Build the documentation
	$(JULIA) --project=test docs/make.jl
	@echo "Done. Consider using 'make devrepl'"


clean: ## Clean up build/doc/testing artifacts
	rm -f src/*.cov test/*.cov
	rm -rf docs/build


distclean: clean ## Restore to a clean checkout state
	rm -f Manifest.toml docs/Manifest.toml test/Manifest.toml
