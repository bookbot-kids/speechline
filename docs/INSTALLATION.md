# Installation

## Install with pip

The easiest way to install SpeechLine is via `pip`. 

### Latest PyPI Release

```sh
pip install speechline
```

This will install the latest **released** version of SpeechLine from PyPI. However, it might not be up to date with the one found in git. We recommend installing via `pip` but cloning the latest `main` branch.

### Latest Git Main Branch

```sh
pip install git+https://github.com/bookbot-kids/speechline.git
```

## Editable Install

You will need an editable install if you’d like to:

- Use the `main` version of the source code.
- Contribute to SpeechLine and need to test changes in the code.

Clone the repository and install SpeechLine with the following commands:

```sh
git clone https://github.com/bookbot-kids/speechline.git
cd speechline
pip install -e .
```

Also install several other dependencies for testing purposes:

```sh
pip install -r requirements_test.txt
```

Now you can easily update your clone to the latest version of SpeechLine with the following command:

```sh
git pull
```

Your Python environment will find the main version of SpeechLine on the next run.

## Linux Packages

SpeechLine relies on several Linux packages in order to run properly. On Linux, you can easily install dependencies by running:

```sh
sudo apt install ffmpeg
sudo apt-get install libsndfile1-dev
```