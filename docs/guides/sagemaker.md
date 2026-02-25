# SpeechLine on AWS SageMaker

SpeechLine is a platform-independent framework. You should be able to run SpeechLine on any platform, even without GPU. However, we developed SpeechLine with the AWS ecosystem in mind, as SpeechLine is an actual framework we use in Bookbot.

In this guide, we are going to walk you through all the steps of running SpeehLine on AWS SageMaker, with additional utility scripts involving other services such as S3, and AirTable for loggings. While they are readily available in SpeechLine, they are not required to run the end-to-end pipeline.

## Create SageMaker Instance

Because we are only going to perform inference on GPU, we don't really need a GPU with high VRAM. Because of that, we prefer to run the cheapest GPU instance on SageMaker, `ml.g4dn.xlarge`. It comes with 1 NVIDIA T4, which has up to 16GB of VRAM. This is sufficient to run both classification and transcription with a reasonable batch size. Simply spin one up via the AWS Web Console, and enter Jupyter Lab.

## Install Linux Packages

We need to install all dependencies of SpeechLine. We'll begin with ffmpeg and ffprobe. Since the VM runs on AWS Linux, it does not have access to `apt`. For this, we have to resort to download static builds of ffmpeg and ffprobe. You can follow this [guide](https://www.maskaravivek.com/post/how-to-install-ffmpeg-on-ec2-running-amazon-linux/) which explains how to install ffmpeg on AWS Linux. We're going to modify several of their steps.

First, let's move to `/usr/local/bin`, download the latest static build of ffmpeg (with ffprobe), unzip the file, and create a symbolic link of the runnables to `/usr/bin`:

```sh
cd /usr/local/bin
sudo wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
sudo tar -xf ffmpeg-release-amd64-static.tar.xz
sudo ln -s /usr/local/bin/ffmpeg-5.1.1-amd64-static/ffmpeg /usr/bin/ffmpeg # (1)
sudo ln -s /usr/local/bin/ffmpeg-5.1.1-amd64-static/ffprobe /usr/bin/ffprobe # (2)
```

1. The latest version of ffmpeg might be different from this guide's. Replace `ffmpeg-5.1.1-amd64-static` with the latest version that you downloaded.
2. The latest version of ffmpeg might be different from this guide's. Replace `ffmpeg-5.1.1-amd64-static` with the latest version that you downloaded.

For cleanliness, let's delete the downloaded compressed file and move back to the working directory.

```sh
sudo rm ffmpeg-release-amd64-static.tar.xz
cd /home/ec2-user/SageMaker
```

You can verify that the runnable is now accessible from anywhere by running

```sh
ffmpeg -version
ffprobe -version
```

## Install SpeechLine

At the time of writing, SpeechLine is yet to be availabe on PyPI. We have to manually clone the source code and install it from there. Also, for convenience, we'll be using the latest PyTorch Conda environment provided for us out-of-the-box.

```sh
source activate amazonei_pytorch_latest_p37
git clone https://github.com/bookbot-kids/speechline.git
cd speechline
pip install .
```

## (Optional) Download Audio Data from S3

If your files are located on AWS S3 and you need to bulk download an entire folder on S3, you can run the `download_s3_bucket` script located under `scripts`. Simply specify the bucket's name, prefix to the folder in the bucket, and the output directory. For ease, let's export some variables which we can reuse later down the line.

```sh
export BUCKET_NAME="my-bucket"
export INPUT_DIR="dropbox/"
export OUTPUT_DIR="train/"
```

```sh
python scripts/download_s3_bucket.py --bucket=$BUCKET_NAME --prefix=$INPUT_DIR --output_dir=$INPUT_DIR
```

Moreover, SpeechLine has a specific requirement on how the folder structure should be. Because we would want to group the files into their specific languages to access the correct models, it should be of the following structure:

```
path_to_files
├── langX
│   ├── a.wav
│   ├── a.txt
│   ├── b.wav
│   └── b.txt
└── langY
    ├── c.wav
    └── c.txt
```

where `*.txt` files are the ground truth transcripts of the utterance `*.wav` files. Note that the `*.txt` files are optional -- the script will still run just fine if you don't have them.

For instance, your directory should look like this:

```
dropbox/
└── en-us
    ├── utt_0.wav
    ├── utt_0.txt
    ...
├── id-id
    ├── utt_1.wav
    ├── utt_1.txt
    ...
...
```

## (Optional) Convert aac Audios to wav

We like to store our audios in aac format due to its smaller size. However, we do need the audio files to be of `*.wav` format for data loading. Hence, we also provided a conversion script that handles this. Simply specify your input directory containing the audios, in the folder structure shown above, and run the script.

```sh
python scripts/aac_to_wav.py --input_dir=$INPUT_DIR
```

## Run SpeechLine

Finally, you should be able to run the end-to-end SpeechLine pipeline. All you need to specify are the input and output directories, and the configuration file. You can use the pre-configured file in `examples/config.json`, or modify based on your requirements.

```sh
python speechline/run.py --input_dir=$INPUT_DIR --output_dir=$OUTPUT_DIR --config="examples/config.json"
```

This should generate the resultant classified, transcribed, and chunked audios in the specified `$OUTPUT_DIR`. An example would be:

```
train/
└── en-us
    ├── utt_0.wav
    ├── utt_0.tsv
    ...
├── id-id
    ├── utt_1.wav
    ├── utt_1.tsv
    ...
```

A structure like shown above, with the right formats, should be easier to be ingested to a different training framework like Kaldi -- which was the main purpose of developing SpeechLine.

## (Optional) Upload Audio Data to S3

Continuing our requirement of storing files in S3, we have similarly provided another script to upload files back to S3 in parallel. Like downloading, uploading simply requires the bucket name, the input directory, and a folder prefix under which the files will be stored.

```sh
python scripts/upload_s3_bucket.py --bucket=$BUCKET_NAME --prefix=$OUTPUT_DIR --input_dir=$OUTPUT_DIR
```

This will upload files with the following path:

```
s3://{BUCKET_NAME}/{PREFIX}/{LANGUAGE}/*.{wav,tsv}
```

## (Optional) Log Data to AirTable

To help monitor the trend of our data collection and training data generation, we like to log our results to AirTable. We are going to write a separate guide on how to set the correct table up, but once that's done and you've logged in, you can simply run the following scripts to log the durations of both raw and generated data.

```sh
export AIRTABLE_URL="https://api.airtable.com/v0/URL/TABLE"
export AIRTABLE_API_KEY="MY_API_KEY"
python scripts/data_logger.py --url=$AIRTABLE_URL --input_dir=$OUTPUT_DIR --label="training"
python scripts/data_logger.py --url=$AIRTABLE_URL --input_dir=$INPUT_DIR --label="archive"
```

With the logged data, you can analyze how much raw data came in, and how much training data is generated, on a per-language basis.