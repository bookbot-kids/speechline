# Force Aligning Punctuations

This guide will show the steps on how to align (or recover) punctuation using a Punctuation Forced Aligner from SpeechLine. 

As you may or may not know, transcription results from a Wav2Vec 2.0 usually do not include punctuations, unless it was trained to do so. In certain cases, however, you might want to have punctuations. An example would be transcribing phonemes of an audio as text-to-speech data, where punctuations are necessary.

## Use Case

For instance, you know the corresponding utterance text of an audio, but you're unsure about the phonemes that were actually uttered by the speaker. Moreover, you want timestamps for each phoneme, such that you could segment them. With a phoneme-level Transcriber and Punctuation Forced Aligner, you could "recover" punctuations and add them to the string of phonemes, as follows:

- Text: `Her red umbrella, is just the best!`
- Transcript: `h h ɚ i d ʌ m b ɹ ɛ l ə ɪ z d͡ʒ ʌ s t ð ə b ɛ s t`
- Recovered Transcript: `h h ɚ i d ʌ m b ɹ ɛ l ə , ɪ z d͡ʒ ʌ s t ð ə b ɛ s t !`

You can restore and align punctuations by simply passing in the ground truth text to a [`PunctuationForcedAligner`](../../reference/aligners/punctuation_forced_aligner). More details will be discuss in the following example.

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
phoneme_offsets = transcriber.predict(dataset, output_offsets=True, return_timestamps="char")
```


    Transcribing Audios:   0%|          | 0/1 [00:00<?, ?ex/s]


Now we will need utilize `gruut`, a grapheme-to-phoneme library that can help transform our ground truth text (given in `sample.txt`) into phonemes. Note that `gruut` retains punctuations during the g2p conversion. This information will be exploited by SpeechLine's `PunctuationForcedAligner` to restore the punctuations from the Wav2Vec 2.0 output and to estimate the right location to insert each punctuation.

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




    [{'end_time': 0.04, 'start_time': 0.0, 'text': 'h'},
     {'end_time': 0.2, 'start_time': 0.14, 'text': 'h'},
     {'end_time': 0.28, 'start_time': 0.24, 'text': 'ɚ'},
     {'end_time': 0.44, 'start_time': 0.42, 'text': 'i'},
     {'end_time': 0.54, 'start_time': 0.5, 'text': 'd'},
     {'end_time': 0.66, 'start_time': 0.64, 'text': 'ʌ'},
     {'end_time': 0.74, 'start_time': 0.7, 'text': 'm'},
     {'end_time': 0.82, 'start_time': 0.78, 'text': 'b'},
     {'end_time': 0.9, 'start_time': 0.84, 'text': 'ɹ'},
     {'end_time': 0.94, 'start_time': 0.92, 'text': 'ɛ'},
     {'end_time': 1.04, 'start_time': 1.0, 'text': 'l'},
     {'end_time': 1.12, 'start_time': 1.08, 'text': 'ə'},
     {'text': ',', 'start_time': 1.12, 'end_time': 1.36},
     {'end_time': 1.38, 'start_time': 1.36, 'text': 'ɪ'},
     {'end_time': 1.58, 'start_time': 1.54, 'text': 'z'},
     {'end_time': 1.62, 'start_time': 1.58, 'text': 'd͡ʒ'},
     {'end_time': 1.66, 'start_time': 1.62, 'text': 'ʌ'},
     {'end_time': 1.76, 'start_time': 1.72, 'text': 's'},
     {'end_time': 1.82, 'start_time': 1.78, 'text': 't'},
     {'end_time': 1.88, 'start_time': 1.86, 'text': 'ð'},
     {'end_time': 1.94, 'start_time': 1.92, 'text': 'ə'},
     {'end_time': 2.0, 'start_time': 1.98, 'text': 'b'},
     {'end_time': 2.06, 'start_time': 2.04, 'text': 'ɛ'},
     {'end_time': 2.26, 'start_time': 2.22, 'text': 's'},
     {'end_time': 2.4, 'start_time': 2.38, 'text': 't'},
     {'text': '!', 'start_time': 2.4, 'end_time': 2.4}]


