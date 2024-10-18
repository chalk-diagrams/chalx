# Convert examples to executed ipynb
EXAMPLES := $(wildcard examples/*.py)
EXAMPLES_IPYNB := $(patsubst examples/%.py,docs/examples/%.ipynb,$(EXAMPLES))

$(EXAMPLES_IPYNB): docs/examples/%.ipynb: examples/%.py
	python $<
	jupytext --run-path . --execute --to ipynb -o $@ $<


examples: $(EXAMPLES_IPYNB)

# Convert walkthrough to executed ipynb
WALKTHROUGH := $(wildcard walkthrough/*.py)
WALKTHROUGH_IPYNB := $(patsubst walkthrough/%.py,notebooks/walkthrough/%.ipynb,$(WALKTHROUGH))
WALKTHROUGH_MD := $(patsubst walkthrough/%.py,docs/notebooks/walkthrough/%.md,$(WALKTHROUGH))

$(WALKTHROUGH_IPYNB): notebooks/walkthrough/%.ipynb: walkthrough/%.py
	mkdir -p $(dir $@)
	python $<
	jupytext --run-path . --execute --to ipynb -o $@ $<

$(WALKTHROUGH_MD): docs/notebooks/walkthrough/%.md: notebooks/walkthrough/%.ipynb
	jupyter nbconvert --to markdown --output-dir docs/$< $<

walkthrough: $(WALKTHROUGH_IPYNB)
walkthrough_docs: $(WALKTHROUGH_MD)

 # Run pre-commit
style:
	pre-commit run --all-files



# Set up chalk kernel
setup-chalk-kernel:
	python -m ipykernel install --user --name=chalk --display-name="Chalk"

.PHONY: examples walkthrough setup-chalk-kernel style
