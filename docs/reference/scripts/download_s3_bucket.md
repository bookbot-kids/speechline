# S3 Bucket Downloader

## Usage

```sh title="example_download_s3_bucket.sh"
python scripts/download_s3_bucket.py [-h] -b BUCKET -p PREFIX -o OUTPUT_DIR [-r REGION]
```

```
Download an S3 bucket with folder prefix.

optional arguments:
  -h, --help            show this help message and exit
  -b BUCKET, --bucket BUCKET
                        S3 bucket name.
  -p PREFIX, --prefix PREFIX
                        S3 folder prefix.
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Path to local output directory.
  -r REGION, --region REGION
                        AWS region name.
```

## Example

```sh title="example_download_s3_bucket.sh"
python scripts/download_s3_bucket.py --bucket="my_bucket" --prefix="recordings/" --output_dir="downloads/"
```

---

::: scripts.download_s3_bucket