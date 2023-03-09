# Force Aligning Punctuations

This guide will show the steps on how to align (or recover) punctuation using a Punctuation Alinger from SpeechLine. 

As you may or may not know, phoneme transcription results from Wav2Vec 2.0 does not include punctuations. You can restore and align punctuations by simply passing in the ground truth text to [`speechline.aligners.PunctuationForcedAligner`](../../reference/aligners/punctuation_forced_aligner)

The first step is, of course, to transcribe your text by loading in the transcription model


```python
from speechline.transcribers import Wav2Vec2Transcriber

transcriber = Wav2Vec2Transcriber("bookbot/wav2vec2-ljspeech-gruut")
```

Load the audio file into a `Dataset` format and pass it into the model


```python
from datasets import Dataset, Audio

dataset = Dataset.from_dict({"audio": ["sample.wav"]})
dataset = dataset.cast_column("audio", Audio(sampling_rate=transcriber.sampling_rate))
```


```python
phoneme_offsets = transcriber.predict(dataset, output_offsets=True)
```


    Transcribing Audios:   0%|          | 0/1 [00:00<?, ?ex/s]


Now we will need utilize `gruut`, a grapheme-to-phoneme library that can help transform our ground truth text (given in `sample.txt`) into phonemes. Note that `gruut` retains punctuations during the g2p conversion. This information will be exploited by SpeechLine's `PunctuationForcedAlinger` to restore the punctuations from the Wav2Vec 2.0 output.

Simply use the following g2p function to convert any text string into phonemes. You can, of course, provide your own g2p function if you wish to do so.


```python
from gruut import sentences


def g2p(text):
    phonemes = []
    for words in sentences(text):
        for word in words:
            if word.is_major_break or word.is_minor_break:
                phonemes += word.text
            elif word.phonemes:
                phonemes += word.phonemes
    return phonemes
```


```python
text = open("sample.txt").readline()
text
```




    'Her red umbrella, is just the best!'



Instantiate `PunctuationForcedAlinger` by passing into it your g2p function. Finally, you can perform punctuation restoration by feeding in the offsets from the transcription model and the ground truth text.


```python
from speechline.aligners import PunctuationForcedAligner

pfa = PunctuationForcedAligner(g2p)
pfa(phoneme_offsets[0], text)
```




    [{'end_time': 0.04, 'phoneme': 'h', 'start_time': 0.0},
     {'end_time': 0.2, 'phoneme': 'h', 'start_time': 0.14},
     {'end_time': 0.28, 'phoneme': 'ɚ', 'start_time': 0.24},
     {'end_time': 0.44, 'phoneme': 'i', 'start_time': 0.42},
     {'end_time': 0.54, 'phoneme': 'd', 'start_time': 0.5},
     {'end_time': 0.66, 'phoneme': 'ʌ', 'start_time': 0.64},
     {'end_time': 0.74, 'phoneme': 'm', 'start_time': 0.7},
     {'end_time': 0.82, 'phoneme': 'b', 'start_time': 0.78},
     {'end_time': 0.9, 'phoneme': 'ɹ', 'start_time': 0.84},
     {'end_time': 0.94, 'phoneme': 'ɛ', 'start_time': 0.92},
     {'end_time': 1.04, 'phoneme': 'l', 'start_time': 1.0},
     {'end_time': 1.12, 'phoneme': 'ə', 'start_time': 1.08},
     {'phoneme': ',', 'start_time': 1.12, 'end_time': 1.36},
     {'end_time': 1.38, 'phoneme': 'ɪ', 'start_time': 1.36},
     {'end_time': 1.58, 'phoneme': 'z', 'start_time': 1.54},
     {'end_time': 1.62, 'phoneme': 'd͡ʒ', 'start_time': 1.58},
     {'end_time': 1.66, 'phoneme': 'ʌ', 'start_time': 1.62},
     {'end_time': 1.76, 'phoneme': 's', 'start_time': 1.72},
     {'end_time': 1.82, 'phoneme': 't', 'start_time': 1.78},
     {'end_time': 1.88, 'phoneme': 'ð', 'start_time': 1.86},
     {'end_time': 1.94, 'phoneme': 'ə', 'start_time': 1.92},
     {'end_time': 2.0, 'phoneme': 'b', 'start_time': 1.98},
     {'end_time': 2.06, 'phoneme': 'ɛ', 'start_time': 2.04},
     {'end_time': 2.26, 'phoneme': 's', 'start_time': 2.22},
     {'end_time': 2.4, 'phoneme': 't', 'start_time': 2.38},
     {'phoneme': '!', 'start_time': 2.4, 'end_time': 2.4}]



And there we go, notice how there are now punctuations in `phoneme_offsets`.
