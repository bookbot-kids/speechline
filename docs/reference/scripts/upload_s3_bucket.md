# S3 Bucket Uploader

## Usage

```sh title="example_upload_s3_bucket.sh"
python scripts/upload_s3_bucket.py [-h] -b BUCKET -p PREFIX -i INPUT_DIR [-r REGION]
```

```
Upload a directory to S3 bucket.

optional arguments:
  -h, --help            show this help message and exit
  -b BUCKET, --bucket BUCKET
                        S3 bucket name.
  -p PREFIX, --prefix PREFIX
                        S3 folder prefix.
  -i INPUT_DIR, --input_dir INPUT_DIR
                        Path to local directory to upload.
  -r REGION, --region REGION
                        AWS region name.
```

## Example

```sh title="example_upload_s3_bucket.sh"
python scripts/upload_s3_bucket.py --bucket="my_bucket" --prefix="recordings/" --input_dir="uploads/"
```

---

::: scripts.upload_s3_bucket