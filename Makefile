
# maintenance
.PHONY: isort flake black test type interrogate darglint \
		clean cleanall style docs check

# installation
.PHONY: install installplus install.e install.all
		install.base install.dev install.docs

# uninstallation
.PHONY: uninstall uninstallplus uninstall.e uninstall.all \
		uninstall.base uninstall.dev uninstall.docs

# documentation
.PHONY: pregendocs.doc pregendocs.examples pregendocs.local pregendocs.remote \
		gendocs \
		postgendocs.doc postgendocs.local postgendocs.remote gendocsall.local

# generate examples
.PHONY: intro squares hanoi escher_square lattice lenet logo \
		hilbert koch tensor latex hex_variation images \
                tree serve
####------------------------------------------------------------####

# libname is either same as PACKAGE_NAME or
#             as on PYPI (replace - with _)
LIBNAME := chalk_diagrams
PACKAGE_NAME := chalk
TESTPYPI_DOWNLOAD_URL := "https://test.pypi.org/simple/"
PYPIPINSTALL := "python -m pip install -U --index-url"
PIPINSTALL_PYPITEST := "$(PYPIPINSTALL) $(TESTPYPI_DOWNLOAD_URL)"
PKG_INFO := "import pkginfo; dev = pkginfo.Develop('.'); print((dev.$${FIELD}))"

# This is where you store the eggfile
# and other generated archives
ARCHIVES_DIR := ".archives"

# Folder path for tests
TESTS_DIR := "tests"

# Interrogate will flag the test as FAILED if
# % success threshold is under the following value
INTERROGATE_FAIL_UNDER := 0  # ideally this should be 100

# Specify paths of various dependency files
REQ_FOLDER := "requirements"
# location: requirements.txt
REQ_FILE := "requirements.txt"
# location: requirements/dev.txt
DEV_REQ_FILE := "dev.txt"
# location: requirements/docs.txt
DOCS_REQ_FILE := "docs.txt"

####------------------------------------------------------------####

### Code maintenance

## Run isort

isort:
	@ echo "✨ Applying import sorter: isort ... ⏳"
	# The settings are maintained in setup.cfg file.
	isort $(PACKAGE_NAME) setup.py \
		 tests \

## Run black

black:
	@ echo "✨ Applying formatter: black ... ⏳"
	black --target-version py38 --line-length 79 $(PACKAGE_NAME) setup.py \
		 tests \

## Run flake8

flake:
	@ echo "✨ Applying formatter: flake8 ... ⏳"
	flake8 --show-source $(PACKAGE_NAME) setup.py \
		 tests \

## Run pytest

test:
	@ echo "✨ Run tests: pytest ... ⏳"
	@if [ -d "$(TESTS_DIR)" ]; then pytest $(TESTS_DIR); else echo "\n\t🔥 No tests configured yet. Skipping tests.\n"; fi

## Run mypy

type:
	@ echo "✨ Applying type checker: mypy ... ⏳"
	mypy --strict --ignore-missing-imports --no-warn-unused-ignores $(PACKAGE_NAME) \
		 tests \

## Run darglint

darglint:
	@ echo "✨ Applying docstring type checker: darglint ... ⏳"
	# The settings are maintained in setup.cfg file.
	darglint -v 2 $(PACKAGE_NAME) --ignore-properties

## Run interrogate

interrogate:
	@ echo "✨ Applying doctest checker: interrogate ... ⏳"
	$(eval INTERROGATE_CONFIG := -vv --ignore-nested-functions --ignore-semiprivate --ignore-private --ignore-magic --ignore-module --ignore-init-method --fail-under $(INTERROGATE_FAIL_UNDER))
	$(eval INTERROGATE_COMMAND := interrogate $(INTERROGATE_CONFIG))
	# Check tests folder
	@if [ -d "$(TESTS_DIR)" ]; then $(INTERROGATE_COMMAND) $(TESTS_DIR); else echo "\n\t🔥 No tests configured yet. Skipping tests.\n"; fi
	# Check package folder
	@$(INTERROGATE_COMMAND) $(PACKAGE_NAME)

## Cleanup
#
# Instruction:
#
# make clean    : if only cleaning artifacts created after running,
#                 code, tests, etc.
# make cleanall : if cleaning all artifacts (including the ones
#                 generated after creating dist and wheels).
#
# Note: archives created (dist and wheels) are stored in
#       $(ARCHIVES_DIR). This is defined at the top of this Makefile.
#--------------------------------------------------------------------

clean:
	@ echo "🪣 Cleaning repository ... ⏳"
	rm -rf \
		.ipynb_checkpoints **/.ipynb_checkpoints \
		.pytest_cache **/.pytest_cache \
		**/__pycache__ **/**/__pycache__

cleanall: clean
	@ echo "🪣 Cleaning dist/archive files ... ⏳"
	rm -rf build/* dist/* $(PACKAGE_NAME).egg-info/* $(ARCHIVES_DIR)/*

## Style Checks and Unit Tests

style: clean isort black flake clean

docs: clean darglint interrogate clean

check: style docs type test clean

####------------------------------------------------------------####

### Code Installation

## Install for development (from local repository)
#
# Instruction: Contributors will need to run...
#
# - "make installplus": if installing for the first time or want to
#                         update to the latest dev-requirements or
#                         other extra dependencies.
# - "make install.e"  : if only installing the local source (after
#                         making some changes) to the source code.
#--------------------------------------------------------------------

# .PHONY: install.e
install.e: clean
	@echo "📀🟢🔵 Installing $(PACKAGE_NAME) from local source ... ⏳"
	python -m pip install -Ue .[tikz,latex,png,svg]

# .PHONY: install
install: clean install.base install.e
	@echo "📀🟢🟡🔵 Installing $(PACKAGE_NAME) and base-dependencies from PyPI ... ⏳"

# .PHONY: installplus
installplus: install.all install.e
	@echo "📀🟢🟡🔵🟠 Installing $(PACKAGE_NAME) and all-dependencies from PyPI ... ⏳"

# .PHONY: install.all
install.all: clean install.base install.dev install.docs
	@echo "📀🟢🟡 Installing $(PACKAGE_NAME)'s all-dependencies from PyPI ... ⏳"

# .PHONY: install.base
install.base:
	@echo "📀🟢🟡 Installing from: $(DEV_REQ_FILE) ... ⏳"
	if [ -f $(REQ_FILE) ]; then python -m pip install -U -r $(REQ_FILE); fi

# .PHONY: install.dev
install.dev:
	@echo "📀🟢🟡 Installing from: $(DEV_REQ_FILE) ... ⏳"
	if [ -f $(REQ_FOLDER)/$(DEV_REQ_FILE) ]; then python -m pip install -U -r $(REQ_FOLDER)/$(DEV_REQ_FILE); fi

# .PHONY: install.docs
install.docs:
	@echo "📀🟢🟡 Installing from: $(DOCS_REQ_FILE) ... ⏳"
	@if [ -f $(REQ_FOLDER)/$(DOCS_REQ_FILE) ]; then python -m pip install -U -r $(REQ_FOLDER)/$(DOCS_REQ_FILE); fi

## Uninstall from dev-environment

# .PHONY: uninstall.e
uninstall.e: clean
	@echo "📀🟢🔵 Uninstalling $(PACKAGE_NAME)' local editable version ... ⏳"
	@# https://stackoverflow.com/questions/48826015/uninstall-a-package-installed-with-pip-install
	rm -rf "$(LIBNAME).egg-info"

# .PHONY: uninstall
uninstall: clean uninstall.base uninstall.e
	@echo "📀🔴🟡🔵 Uninstalling $(PACKAGE_NAME) and base-dependencies from PyPI ... ⏳"

# .PHONY: uninstallplus
uninstallplus: uninstall.all uninstall.e
	@echo "📀🔴🟡🔵🟠 Uninstalling $(PACKAGE_NAME) and all-dependencies from PyPI ... ⏳"

# .PHONY: uninstall.all
uninstall.all: clean uninstall.base uninstall.dev uninstall.docs clean
	@echo "📀🔴🟡 Uninstalling $(PACKAGE_NAME)'s all-dependencies from PyPI ... ⏳"

# .PHONY: uninstall.base
uninstall.base:
	@echo "📀🔴🟡 Uninstalling from: $(DEV_REQ_FILE) ... ⏳"
	if [ -f $(REQ_FILE) ]; then python -m pip uninstall -r $(REQ_FILE); fi

# .PHONY: uninstall.dev
uninstall.dev:
	@echo "📀🔴🟡 Uninstalling from: $(DEV_REQ_FILE) ... ⏳"
	if [ -f $(REQ_FOLDER)/$(DEV_REQ_FILE) ]; then python -m pip uninstall -r $(REQ_FOLDER)/$(DEV_REQ_FILE); fi

# .PHONY: uninstall.docs
uninstall.docs:
	@echo "📀🔴🟡 Uninstalling from: $(DEV_REQ_FILE) ... ⏳"
	@if [ -f $(REQ_FOLDER)/$(DOCS_REQ_FILE) ]; then python -m pip uninstall -r $(REQ_FOLDER)/$(DOCS_REQ_FILE); fi


## Install from test.pypi.org
#
# Instruction:
#
# 🔥 This is useful if you want to test the latest released package
#    from the TestPyPI, before you push the release to PyPI.
#--------------------------------------------------------------------

pipinstalltest:
	@echo "💿 Installing $(PACKAGE_NAME) from TestPyPI ($(TESTPYPI_DOWNLOAD_URL)) ... ⏳"
	# Example Usage:
	#   👉 To run a command like:
	#   > python -m pip install -U --index-url https://test.pypi.org/simple/ $(PACKAGE_NAME)==$(VERSION)
	#   👉 Run the following command:
	#   > make pipinstalltest VERSION="0.1.0"
	#   👉 Specifying VERSION="#.#.#" installs a specific version.
	#      If no version is specified, the latest version is installed from TestPyPI.
	@if [ $(VERSION) ]; then $(PIPINSTALL_PYPITEST) $(PACKAGE_NAME)==$(VERSION); else $(PIPINSTALL_PYPITEST) $(PACKAGE_NAME); fi;

## Gendocs

# .PHONY: gendocs
gendocs:
	@echo "🔥 Generate documentation with MkDocs ... ⏳"
	# generate documentation
	mkdocs serve --dirtyreload

## Postgendocs

# .PHONY: postgendocs.doc
postgendocs.doc:
	#echo "Cleanup docs... ⏳"
	rm -rf docs/doc

# .PHONY: postgendocs.local
postgendocs.local: postgendocs.doc

# .PHONY: postgendocs.remote
postgendocs.remote: postgendocs.doc

# .PHONY: gendocsall.local
gendocsall.local: pregendocs.local gendocs postgendocs.local

# .PHONY: gendocsall.remote
# gendocsall.remote: pregendocs.remote gendocs postgendocs.remote
# 	@ # Use mkdocs-publish-ghpages.yml action instead of this make command


####------------------------------------------------------------####



# Define the examples directory
EXAMPLES_DIR := examples

# List of images to be generated
IMAGES := squares bigben hanoi intro hilbert koch tensor hex_variation tree tournament parade arrows lenet escher_square logo

# Generate the images
images: $(IMAGES)
	@echo "🎁 Generate all examples ... ⏳"

# Rule to process each image
$(IMAGES):
	python $(EXAMPLES_DIR)/$@.py
	jupytext --execute --set-kernel chalk --to ipynb -o $(EXAMPLES_DIR)/$@.ipynb $(EXAMPLES_DIR)/$@.py
	jupyter nbconvert --to html $(EXAMPLES_DIR)/$@.ipynb 

# List of images to be generated
VISTESTS := alignment arc broadcast combinators envelope names path rendering shapes style subdiagram traces trails transformations text

VT_DIR := api

vis: $(VISTESTS)
	@echo "🎁 Generate all vis tests ... ⏳"

$(VISTESTS):
	python $(VT_DIR)/$@.py
	jupytext --execute --set-kernel chalk --to ipynb -o $(VT_DIR)/$@.ipynb $(VT_DIR)/$@.py
	jupyter nbconvert --to html $(VT_DIR)/$@.ipynb 


.PHONY: images $(IMAGES) vis $(VISTESTS)


serve:
	python -m http.server 8080 -d examples/output/

docsapi:
	python docs/api/*.py



