from distutils import dir_util
import pytest
import os

from speechline.ml.dataset import prepare_dataframe
from speechline.ml.classifier import AudioClassifier


@pytest.fixture
def datadir(tmpdir, request):
    """
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.
    Source: https://stackoverflow.com/a/29631801
    """
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    return tmpdir


def test_prepare_dataframe(datadir):
    df = prepare_dataframe(datadir)
    assert df.shape == (3, 5)


def test_audio_classifier(datadir):
    model_checkpoint = "bookbot/distil-wav2vec2-adult-child-cls-52m"
    classifier = AudioClassifier(model_checkpoint)
    df = prepare_dataframe(datadir)
    dataset = classifier.format_audio_dataset(df)
    predictions = classifier.predict(dataset)
    assert predictions == ["child", "child", "child"]
