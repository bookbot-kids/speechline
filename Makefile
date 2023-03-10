.PHONY: quality style cov test

# use local checkout instead of pre-installed version
export PYTHONPATH = speechline

check_dirs := examples speechline tests
notebooks := $(wildcard notebooks/*/*.ipynb)

# runs checks on all files

quality:
		black --check $(check_dirs)
		isort --check-only $(check_dirs)
		flake8 $(check_dirs)
		mkdocs build

# runs checks on all files and potentially modifies some of them

style:
		black $(check_dirs)
		isort $(check_dirs)

# runs coverage tests for the library

cov:
		python -m pytest --cov-report term-missing --cov -v

# runs all tests for the library

test:
		tox -r

# execute and convert notebooks to markdown guides

nb2md:
		for nb in $(notebooks) ; do \
			jupyter nbconvert --to markdown $$nb --execute --RegexRemovePreprocessor.patterns="['\\s*\\Z']" --output-dir docs/guides/; \
		done