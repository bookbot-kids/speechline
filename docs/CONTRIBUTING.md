# Contributing to SpeechLine

!!! info "Credits"

    This document was adapted from the open-source contribution guidelines for [Facebook's Draft](https://github.com/facebook/draft-js/blob/a9316a723f9e918afde44dea68b5f9f39b7d9b00/CONTRIBUTING.md) and [HuggingFace's Contributing Guidelines](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md).


Hi there! Thanks for taking your time to contribute to SpeechLine!

We welcome everyone to contribute and we value each contribution, even the smallest ones! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

!!! tip

    Please be mindful to respect our [Code of Conduct](https://github.com/bookbot-kids/speechline/blob/main/CODE_OF_CONDUCT.md).

## Create a Pull Request

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests. Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests!

1. Fork the [repository](https://github.com/bookbot-kids/speechline) by clicking on the [Fork](https://github.com/bookbot-kids/speechline/fork) button on the repository's page. This creates a copy of the code under your GitHub user account.
   
2. Clone your fork to your local disk, and add the base repository as a remote:

    ```sh
    git clone git@github.com:<your Github handle>/speechline.git
    cd speechline
    git remote add upstream https://github.com/bookbot-kids/speechline.git
    ```

3. Create a new branch to hold your development changes:

    ```sh
    git checkout -b a-descriptive-name-for-my-changes
    ```

4. Set up a development environment by running the following command in a virtual environment:

    ```sh
    pip install .
    pip install -r requirements_test.txt
    ```

5. Install Linux package dependencies

    ```sh
    sudo apt install ffmpeg
    sudo apt-get install libsndfile1-dev
    ```

6. Develop the features on your branch, add tests and documentation.

    As you work on your code, you should make sure the test suite passes. Run the tests impacted by your changes like this:

    ```sh
    pytest tests/<TEST_TO_RUN>.py
    ```

    SpeechLine relies on `black` and `isort` to format its source code consistently. After you make changes, apply automatic style corrections and code verifications that can't be automated in one go with:

    ```sh
    make style
    ```

    SpeechLine also uses `flake8` to check for coding mistakes. Quality controls are run by the CI, but you can run the same checks with:

    ```sh
    make quality
    ```

    This will also ensure that the documentation can still be built.

    Once you're happy with your changes, add changed files with git add and record your changes locally with git commit:

    ```sh
    git add modified_file.py
    git commit -m "YOUR_COMMIT_MESSAGE_HERE"
    ```

    To keep your copy of the code up to date with the original repository, rebase your branch on upstream/branch before you open a pull request or if requested by a maintainer:

    ```sh
    git fetch upstream
    git rebase upstream/main
    ```

    Push your changes to your branch:

    ```sh
    git push -u origin a-descriptive-name-for-my-changes
    ```

    If you've already opened a pull request, you'll need to force push with the --force flag. Otherwise, if the pull request hasn't been opened yet, you can just push your changes normally.

7. Now you can go to your fork of the repository on GitHub and click on Pull request to open a pull request. When you're ready, you can send your changes to the project maintainers for review.

8. It's ok if maintainers request changes, it happens to our core contributors too! So everyone can see the changes in the pull request, work in your local branch and push the changes to your fork. They will automatically appear in the pull request.

## Tests

An extensive test suite is included to test the library behavior and several examples. Library tests can be found in the [tests](https://github.com/bookbot-kids/speechline/tree/main/tests) folder.

We use `pytest` to run our tests. From the root of the repository, specify a path to a subfolder or a test file to run the test.

```sh
python -m pytest --cov-report term-missing --cov -v ./tests/<TEST_TO_RUN>.py
```

Alternatively, you can run `make cov`.

Moreover, for our CI, we utilize `tox` that runs on GitHub Actions. You can similarly run this set of extensive tests locally by running:

```sh
tox -r
```

or equivalently, `make test`. 

## Style guide

For documentation strings, SpeechLine follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

## Report bugs using Github's [issues](https://github.com/bookbot-kids/speechline/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/bookbot-kids/speechline/issues/new). [This is an example](http://stackoverflow.com/q/12488905/180626) of a good and thorough bug report.

Great Bug Reports tend to have:

- A quick summary and/or background
- Steps to reproduce
    - Be specific!
    - Give sample code if you can.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## License
By contributing, you agree that your contributions will be licensed under its Apache 2.0 License. In short, when you submit code changes, your submissions are understood to be under the same [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0) that covers the project. Feel free to contact the maintainers if that's a concern.