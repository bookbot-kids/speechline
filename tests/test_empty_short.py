from speechline.transcribers import Wav2Vec2Transcriber
from speechline.utils.dataset import format_audio_dataset, prepare_dataframe


def test_wav2vec2_transcriber(datadir):
    model_checkpoint = "bookbot/wav2vec2-ljspeech-gruut"
    transcriber = Wav2Vec2Transcriber(model_checkpoint)
    df = prepare_dataframe(datadir)
    dataset = format_audio_dataset(df, sampling_rate=32000)
    transcriptions = transcriber.predict(dataset, chunk_length_s=5)
    assert transcriptions == [""]


def test_audio_url(datadir):
    model_checkpoint = "bookbot/wav2vec2-ljspeech-gruut"
    transcriber = Wav2Vec2Transcriber(model_checkpoint)
    transcriber.pipeline(
        "https://datasets-server.huggingface.co/assets/common_voice/--/en/train/0/audio"
        "/audio.mp3"
    )
    transcriber.pipeline(f"{datadir}/en-us/test_audio.mp3")
