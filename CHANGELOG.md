# Changelog:

---
### Pre-release status (13.03.2023):
- Now using [pre-commit](https://pre-commit.com/) as an automatic CI tool.
- pre-commit runs some standard hooks, black, ruff and mypy.
- The config for black, ruff and mypy is still in `pyproject.toml`
- To run the pre-commit manual, run `pre-commit run --all-files`
- To reinstall the pre-commits, run `pre-commit install`
- My pre-commit GH action is running on ubuntu on python3.9
- **Note:** There is some redundancy, as I have to specify the versions of
    black, ruff and mypy twice. Once in the `.pre-commit-config.yaml` and once in the
    `pyproject.toml`.
  Maybe I am only specifying which *hook* to use in the `.pre-commit-config.yaml`.
- Q:
  - Should I add the tests also to pre-commit?
  - When to test?


### Pre-release status (12.03.2023):

**Development features:**

1. Use [Black](https://github.com/psf/black) for formatting the code.
    - Local terminal usage: `black .`
    - Config in `pyproject.toml`
      - Set line-length to `88`, set python versions to `py8`to `py11`
    - GitHub Actions:
      - Just check with black by running: `black --check .`

2. Use [Ruff](https://github.com/charliermarsh/ruff) as the Python linter.
   - Local terminal usage: `ruff check .`
   - Config in `pyproject.toml`
     - set line-length, the Rules (*pycodestyle*, *pyflakes*, *pydocstyle*, *isort*),
      the source files, and to ignore rule *F401* to not complain about unused imports
      in `__init__` files.
   - GitHub Actions:
     - run `ruff check --format=github .`

3. Use [MyPy](https://github.com/python/mypy) as for type checking.
   - Local terminal usage: `mypy`
   - Config in `pyproject.toml`
     - Specify files: `src/lorenzpy/`
     - Ignore missing imports for `plotly`
   - GitHub Actions:
     - Just running `mypy`

4. Testing with [pytest-cov](https://github.com/pytest-dev/pytest-cov)
   - Local terminal usage: `pytest`
   - Config in `pyproject.toml`
     - automatically add the following options when `pytest` is run:
       - `--verbose --cov-config=pyproject.toml --cov-report term-missing --cov=lorenzpy`
       - Note: This is somehow important for GitHub actions to work fine...
     - Omit the `plot` folder for the coverage.
   - GitHub Actions:
     - simply running `pytest` and then uploading the coverage reports to [Codecov](https://about.codecov.io/)
     using the GitHub action: [codecov-action](https://github.com/codecov/codecov-action)

5. Generating docs with [mkdocs](https://github.com/mkdocs/mkdocs):
   - Following [this tutorial](https://realpython.com/python-project-documentation-with-mkdocs/)
   - Generate the docs with `mkdocs gh-deploy`
   - Use the plugin [mkdocstrings](https://github.com/mkdocstrings/mkdocstrings) to
   automatically use the code-docstrings in the documentation.

**Deployment on PyPI with GitHub Actions**:
- Following the [PyPA tutorial](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
- Also use [this PyPA tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/) to manually upload the package to test PyPI.
    - This uses `build` and `twine`

**Resources**:
Taking inspiration from:
 https://github.com/dkmiller/modern-python-package, and
https://github.com/denkiwakame/py-tiny-pkg.
