import pytest
from speechline.utils import s3


@pytest.fixture
def bucket_name():
    return "my-test-bucket"


@pytest.fixture
def s3_test(s3_client, bucket_name):
    s3_client.create_bucket(Bucket=bucket_name)
    yield


def test_s3_utils(tmp_path, s3_client, s3_test, bucket_name):
    my_client = s3.S3Client()
    testbox = tmp_path / "testbox"
    foo = testbox / "foo.txt"
    bar = testbox / "subbox/bar.txt"
    my_client.put_object(bucket_name, key=str(testbox / "baz/"), value="sb")
    my_client.put_object(bucket_name, key=str(foo), value="foo")
    my_client.put_object(bucket_name, key=str(bar), value="bar")
    my_client.download_s3_folder(bucket_name, s3_folder=str(testbox))
    my_client.download_s3_folder(
        bucket_name, s3_folder=str(testbox), local_dir=tmp_path
    )
    assert foo.read_text() == "foo"
    assert bar.read_text() == "bar"
