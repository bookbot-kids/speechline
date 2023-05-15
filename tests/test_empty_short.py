from speechline.transcribers import Wav2Vec2Transcriber
from speechline.utils.dataset import format_audio_dataset, prepare_dataframe

def test_wav2vec2_transcriber(datadir):
    model_checkpoint = "bookbot/wav2vec2-ljspeech-gruut"
    transcriber = Wav2Vec2Transcriber(model_checkpoint)
    df = prepare_dataframe(datadir)
    dataset = format_audio_dataset(df, sampling_rate=transcriber.sampling_rate)
    transcriptions = transcriber.predict(dataset, return_timestamps="char")
    assert transcriptions ==  ['']
