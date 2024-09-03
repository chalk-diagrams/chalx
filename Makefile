# Convert examples to executed ipynb
EXAMPLES := $(wildcard examples/*.py)
EXAMPLES_IPYNB := $(patsubst examples/%.py,output/examples/%.ipynb,$(EXAMPLES))

$(EXAMPLES_IPYNB): output/examples/%.ipynb: examples/%.py
	@mkdir -p output/examples
	jupytext --to ipynb -o $@ $<
	jupyter nbconvert --execute --inplace --to ipynb --ExecutePreprocessor.kernel_name=chalk --ExecutePreprocessor.cwd=. $@

examples: $(EXAMPLES_IPYNB)

# Convert walkthrough to executed ipynb
WALKTHROUGH := $(wildcard walkthrough/*.py)
WALKTHROUGH_IPYNB := $(patsubst walkthrough/%.py,output/walkthrough/%.ipynb,$(WALKTHROUGH))

$(WALKTHROUGH_IPYNB): output/walkthrough/%.ipynb: walkthrough/%.py
	@mkdir -p output/walkthrough
	jupytext --to ipynb -o $@ $<
	jupyter nbconvert --execute --inplace --to ipynb --ExecutePreprocessor.kernel_name=chalk --ExecutePreprocessor.cwd=. $@

walkthrough: $(WALKTHROUGH_IPYNB)

 # Run pre-commit
style:
	pre-commit run --all-files



# Set up chalk kernel
setup-chalk-kernel:
	python -m ipykernel install --user --name=chalk --display-name="Chalk"

.PHONY: examples walkthrough setup-chalk-kernel style
