# Contributing to LorenzPy

... 🚧 More will come soon ...

### 🛠️ Setting up the development environment:

1. Fork and Clone the repository.
2. Install the package with dev-dependencies in an editable way
(from your local clone of the repository):
    ````
    pip install -e .[dev,plot]
    ````
   This will install:
    - `pytest` and `pytest-cov` for testing with coverage.
    - `black` and `ruff` for linting.
    - `mypy` for type-hinting.
    - `mkdocs`, `mkdocstrings[python]` and `mkdocs-material` for
   documentation generation.
    - `pre-commit` to add a pre-commit hook that includes:
      - `trailing-whitespace` and `check-added-large-files`
      - `black`
      - `ruff`
      - `mypy`
3. Install the pre-commit hooks with `pre-commit install`

After making changes to the code you can:
- (optional) run `pre-commit run --all-files` to manually run the pre-commit hooks.
  Otherwise, pre-commit is always run when committing a file.
- Test the code with coverage by running `pytest`
- To see the local docs in your browser, run `mkdocs serve`.
Changes in the code and docs files will be automatically implemented.

### ✈️ Manual deployment:
⚠️Steps are not fixed yet.
1. Make changes ready for deployment (on seperate branch)
   - Modify Documentation according to the code changes.
   Observe changes with `mkdocs serve`.
   - Run pre-commit (runs automatically when commiting)
   - Bump version number in `pyproject.toml` and `lorenzpy/__init__`
   - Test everything with `pytest`
2. Create merge request. See if all GitHub actions pass. Merge.
3. (Me) Pull code on `main` branch after MR.
4. (Me) Reinstall package in a fresh venv using `pip install -e .[dev,plot]`
5. (Me) Tag git? Create GH release?
6. (Me) Upload to TestPyPI and PyPI and GH docs:
   - Run `python -m pip install --upgrade build twine`
   - To build run `python -m build`
   - To upload to Test PyPI: run `python -m twine upload --repository testpypi dist/*`
     - Test the package installation from Test PyPI by running:
        ```
        pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple lorenzpy
        ```
   - To upload to PyPI run `twine upload dist/*`
   - Upload new docs by running: ``mkdocs gh-deploy``

### 📚 Some resources:

| Element                                              | Resources                                                                                                                                                                                                                                                                |
|------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| General Python Packaging                             | [Python Packages E-book](https://py-pkgs.org/), [Creating an open source Python project from scratch](https://jacobtomlinson.dev/series/creating-an-open-source-python-project-from-scratch/), [PyPA Packaging Python Projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/) |
| Documentation with MKdocs                            | [Real Python Tutorial](https://realpython.com/python-project-documentation-with-mkdocs/), [Latex in MKdocs](https://squidfunk.github.io/mkdocs-material/reference/mathjax/#docsjavascriptsmathjaxjs)                                                                     |
| Other repos using `setuptools` with `pyproject.toml` | [modern-python-package](https://github.com/dkmiller/modern-python-package), [py-tiny-pkg](https://github.com/denkiwakame/py-tiny-pkg)                                                                                                                                    |
| Markdown                                             | [The Markdown Guide](https://www.markdownguide.org/), [GitHub Basic writing and formatting syntax](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)                   |
