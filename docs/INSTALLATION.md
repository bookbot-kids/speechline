# Installation

## Install with pip

Coming soon!

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