# SpeechLine on Google Cloud VM

SpeechLine is a platform-independent framework. You should be able to run SpeechLine on any platform, even without GPU. 

In this guide, we are going to walk you through all the steps of running SpeehLine on Google Cloud Platform (GCP), with additional utility scripts involving other services such as AirTable for loggings. While they are readily available in SpeechLine, they are not required to run the end-to-end pipeline.

## Create GCP VM Instance

Because we are only going to perform inference on GPU, we don't really need a GPU with high VRAM. Because of that, we prefer to run the instance `n1-standard-16` with 1 NVIDIA T4, which has up to 16GB of VRAM. This is sufficient to run both classification and transcription with a reasonable batch size. Simply spin one up via the Google Cloud Compute Engine page, and enter the VM via SSH.

## Install Linux Packages

We need to install all dependencies of SpeechLine. We'll begin with ffmpeg and ffprobe. 

```sh
sudo apt-get install -y ffmpeg
```

You can verify that installation is successful by running

```sh
ffmpeg -version
ffprobe -version
```

## Install SpeechLine

We have to manually clone the source code and install SpeechLine from there. 

```sh
git clone https://github.com/bookbot-kids/speechline.git
cd speechline
pip install .
```

Ensure that your PyTorch installation has access to GPU by running:

```sh
python -c "import torch; print(torch.cuda.is_available())"
```

If you're getting an error like the following,

<details>
<summary>Error Log</summary>

```py
Traceback (most recent call last):
  File "/opt/conda/lib/python3.7/site-packages/torch/__init__.py", line 172, in _load_global_deps
    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
  File "/opt/conda/lib/python3.7/ctypes/__init__.py", line 364, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: /opt/conda/lib/python3.7/site-packages/torch/lib/../../nvidia/cublas/lib/libcublas.so.11: symbol cublasLtHSHMatmulAlgoInit version libcublasLt.so.11 not defined in file libcublasLt.so.11 with link time reference

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/opt/conda/lib/python3.7/site-packages/torch/__init__.py", line 217, in <module>
    _load_global_deps()
  File "/opt/conda/lib/python3.7/site-packages/torch/__init__.py", line 178, in _load_global_deps
    _preload_cuda_deps()
  File "/opt/conda/lib/python3.7/site-packages/torch/__init__.py", line 158, in _preload_cuda_deps
    ctypes.CDLL(cublas_path)
  File "/opt/conda/lib/python3.7/ctypes/__init__.py", line 364, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: /opt/conda/lib/python3.7/site-packages/nvidia/cublas/lib/libcublas.so.11: symbol cublasLtHSHMatmulAlgoInit version libcublasLt.so.11 not defined in file libcublasLt.so.11 with link time reference
```
</details>

Export the following variable to solve the issue above.

```sh
export LD_LIBRARY_PATH=/opt/conda/lib/python3.7/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH
```

and then run the same line of code to recheck if your PyTorch has access to your GPU.

## (Optional) Download Audio Data from Google Cloud Storage

If your files are located on Google Cloud Storage as a zipped file, you can simply use the `gcloud` command to copy the file from Cloud Storage into your VM.

```sh
gcloud alpha storage cp gs://{BUCKET}/{FILENAME.zip} {PATH/TO/OUTPUT}
```

which you can then simply unzip as per normal.

```sh
unzip -q {PATH/TO/OUTPUT/FILENAME.zip}
```

SpeechLine has a specific requirement on how the folder structure should be. Because we would want to group the files into their specific languages to access the correct models, it should be of the following structure:

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
export INPUT_DIR="dropbox/"
export OUTPUT_DIR="training/"
```

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

## (Optional) Log Data to AirTable

To help monitor the trend of our data collection and training data generation, we like to log our results to AirTable. We are going to write a separate guide on how to set the correct table up, but once that's done and you've logged in, you can simply run the following scripts to log the durations of both raw and generated data.

```sh
export AIRTABLE_URL="https://api.airtable.com/v0/URL/TABLE"
export AIRTABLE_API_KEY="MY_API_KEY"
python scripts/data_logger.py --url=$AIRTABLE_URL --input_dir=$OUTPUT_DIR --label="training"
python scripts/data_logger.py --url=$AIRTABLE_URL --input_dir=$INPUT_DIR --label="archive"
```

With the logged data, you can analyze how much raw data came in, and how much training data is generated, on a per-language basis.