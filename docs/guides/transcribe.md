# Performing Audio Transcription

This guide will explain how to transcribe audio files using SpeechLine.

First, load in the your transcription model by passing its Hugging Face model checkpoint into [`Wav2Vec2Transcriber`](../../reference/transcribers/wav2vec2).


```python
from speechline.transcribers import Wav2Vec2Transcriber

transcriber = Wav2Vec2Transcriber("bookbot/wav2vec2-ljspeech-gruut")
```

Next, you will need to transform your input audio file (given by `sample.wav`) into a `Dataset` format like the following


```python
from datasets import Dataset, Audio

dataset = Dataset.from_dict({"audio": ["sample.wav"]})
dataset = dataset.cast_column("audio", Audio(sampling_rate=transcriber.sampling_rate))
```

Once preprocessing is finished, simply pass the input data into the transcriber.


```python
phoneme_offsets = transcriber.predict(dataset, output_offsets=True, return_timestamps="char")
```


    Transcribing Audios:   0%|          | 0/1 [00:00<?, ?ex/s]


The output format of the transcription model is shown below. It is a list of dictionary containing the transcribed `text`, `start_time` and `end_time` stamps of the corresponding phoneme token.


```python
phoneme_offsets
```




    [[{'end_time': 0.02, 'start_time': 0.0, 'text': 'ɪ'},
      {'end_time': 0.3, 'start_time': 0.26, 'text': 't'},
      {'end_time': 0.36, 'start_time': 0.34, 'text': 'ɪ'},
      {'end_time': 0.44, 'start_time': 0.42, 'text': 'z'},
      {'end_time': 0.54, 'start_time': 0.5, 'text': 'n'},
      {'end_time': 0.58, 'start_time': 0.54, 'text': 'oʊ'},
      {'end_time': 0.62, 'start_time': 0.58, 'text': 't'},
      {'end_time': 0.78, 'start_time': 0.76, 'text': 'ʌ'},
      {'end_time': 0.94, 'start_time': 0.92, 'text': 'p'}]]



You can manually check the model output by playing a segment (using the start and end timestamps) of your input audio file. 

First, load your audio file.


```python
from pydub import AudioSegment

audio = AudioSegment.from_file("sample.wav")
audio
```





<audio controls>
    <source src="data:audio/mpeg;base64,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" type="audio/mpeg"/>
    Your browser does not support the audio element.
</audio>




You can use the following function to play a segment of your audio from a given offset


```python
def play_segment(offsets, index: int):
    start = offsets[index]["start_time"]
    end = offsets[index]["end_time"]
    print(offsets[index]["text"])
    return audio[start * 1000 : end * 1000]
```

Here are some examples of the phoneme segments


```python
play_segment(phoneme_offsets[0], 0)
```

    ɪ






<audio controls>
    <source src="data:audio/mpeg;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU5LjI3LjEwMAAAAAAAAAAAAAAA//NYwAAAAAAAAAAAAEluZm8AAAAPAAAAAwAAAfgAkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSycnJycnJycnJycnJycnJycnJycnJycnJycnJycnJycnJ////////////////////////////////////////////AAAAAExhdmM1OS4zNwAAAAAAAAAAAAAAACQDQAAAAAAAAAH4RCJYDwAAAAAAAAAAAAAA//M4xAAAAANIAAAAAExBTUUzLjEwMFVVVVVVVVVVVVVVVVVVVVVVVVVVVUxBTUUzLjEwMFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV//M4xF8AAANIAAAAAFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV//M4xKAAAANIAAAAAFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV" type="audio/mpeg"/>
    Your browser does not support the audio element.
</audio>





```python
play_segment(phoneme_offsets[0], 1)
```

    t






<audio controls>
    <source src="data:audio/mpeg;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU5LjI3LjEwMAAAAAAAAAAAAAAA//NYwAAAAAAAAAAAAEluZm8AAAAPAAAABAAAAmQAeHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4paWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpdLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tL/////////////////////////////////AAAAAExhdmM1OS4zNwAAAAAAAAAAAAAAACQEQAAAAAAAAAJkz4h9eAAAAAAAAAAAAAAA//M4xAATsH6oBUYQABI/YwgAAAAAgLJ20Q0EAGAwsmTJkyZMAAAAQCAIAmD4Pg+D4IAgCAIAmD4Pg+D4IAhgMHz/+Jz8YCAIHP/8EMoAwfB993T/lwfB9/0icH0AAaqqqoEVBZTX/ri50ziU//M4xBAYCJbFk5h4AG7xUDQy56MGxXIgxsr08/AIfdeiY3nWNePxQNKfWJz4Z2VCQmp5dgJ4hasFAuA2DYrYLjgcU4e8I/CTDBmfGXlZ3W/3b9YxJ0Ui4Pf+Lf8MKpAAHAUHA4IAAA5ILp0s//M4xA4XohayUZJoAAIsL/LXg4hOrJYlo7Bla+EWGDG0LatlfHgRxLiXI1er44S4WkkXyj/+YsskjZFH//OF5JZqio2S///OmqKCSS0Vqcy//LgqCRUAmQ6Cv//BKkxBTUUzLjEwMKqqqqqq//M4xA4AAANIAcAAAKqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq" type="audio/mpeg"/>
    Your browser does not support the audio element.
</audio>





```python
play_segment(phoneme_offsets[0], 2)
```

    ɪ






<audio controls>
    <source src="data:audio/mpeg;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU5LjI3LjEwMAAAAAAAAAAAAAAA//NYwAAAAAAAAAAAAEluZm8AAAAPAAAAAwAAAfgAkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSycnJycnJycnJycnJycnJycnJycnJycnJycnJycnJycnJ////////////////////////////////////////////AAAAAExhdmM1OS4zNwAAAAAAAAAAAAAAACQDQAAAAAAAAAH4XMX/qwAAAAAAAAAAAAAA//M4xAAU4Ha0pUwQAMBYmcpSlFgkATBuT38pSk07W169evXgAACAIAgCYPg+D4PggCAIAgZrB8Pu5cP/5cHwfD9QIAgn+oEwfB8/BAMSgDB8Hz8Z0/EAIBjB8Hz/lwfB9a0NQ+9FWyloI8JT//M4xAsXKhq4AZiAABkbEq0PQM5DuXaAEOO01EelZjJ6icNlpGLJeyKkkO7fomlRhr0vz6pElF4V1JZSZaJS/myUrrUTdjZkDbMjf/zAtt6+xSPqer/yiyh4azSr/1IbSCug5BqB0RYlxKEo//M4xA0PQG3sN8wYAJyYEgPLwRABHUegJFkrCUqZMT3mg12+qV/LPK/qPfkSx4r1HhKs7/9YaUHf/U//6ExBTUUzLjEwMKqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq" type="audio/mpeg"/>
    Your browser does not support the audio element.
</audio>





```python
play_segment(phoneme_offsets[0], 3)
```

    z






<audio controls>
    <source src="data:audio/mpeg;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU5LjI3LjEwMAAAAAAAAAAAAAAA//NYwAAAAAAAAAAAAEluZm8AAAAPAAAAAwAAAfgAkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSycnJycnJycnJycnJycnJycnJycnJycnJycnJycnJycnJ////////////////////////////////////////////AAAAAExhdmM1OS4zNwAAAAAAAAAAAAAAACQDQAAAAAAAAAH4PFSuoQAAAAAAAAAAAAAA//M4xAAPCC6NE0MYAIQIna2Hd/iIAAYGH8HHLB+6UEAJ8HwQdgmHy4Ph+IHfg+PB8HzQgd/wfD/plw//y4Y//KO4IAgCAIAmD6q2W3W3DYbDYSiUSBsKBuTByE7tsm4RSEH9Xl7Wm2BjgChp//M4xCIdYp7eX49AAsFoRCn7AcPsoBFL/xMbRQjCpQdXf/nDkKRKdKxv/94yDW5gsGwsWN//xXRt6M0fafQ3//9u1H9c1Q247/duW////ImbeEmsacAjQKVNKaj/B//hOgCkAsAYDRMmKRSz//M4xAsSy0HIUckQAEqhQ4sia2MpS8Y55NzPl+bLVqGfv2///9cxnUtSt///+Z6lab/+j5f/zGyllKyGNUxnlKyGL/9SlZDFaYxnKFNVTEFNRTMuMTAwVVVVVVVVVVVVVVVVVVVVVVVVVVVV" type="audio/mpeg"/>
    Your browser does not support the audio element.
</audio>





```python
play_segment(phoneme_offsets[0], 4)
```

    n






<audio controls>
    <source src="data:audio/mpeg;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU5LjI3LjEwMAAAAAAAAAAAAAAA//NYwAAAAAAAAAAAAEluZm8AAAAPAAAABAAAAmQAeHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4paWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpdLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tL/////////////////////////////////AAAAAExhdmM1OS4zNwAAAAAAAAAAAAAAACQEQAAAAAAAAAJkjLMIhgAAAAAAAAAAAAAA//M4xAASWFK0DUkYAHh4jRoyMVitHtIyMEwBgDBMNk6NAwCAIAgCYPg+D4PggCAIAgCYPg+D4fBBwIAgCYPh/GAgCAY/6f9P////wfD7uXB8HwfjAQOKSgALZSkcjiCgkEYTQAprX6q0zShK//M4xBUaUWbSX5hYACzEMVGhEmHq0V9T/RFLrKSPZ4fD2QtQ5esvKA0c+9vsuqYbmjI7f/fL30hNm6+v/XmkMlt66hYC4PcRy5qKBAyXu7NAfegmsLg4eb63/PqZqEjQouqwAAEABQQCBDAA//M4xAoX2na6V41oABAgAFyGClj+J94jQwVeEmAF4OL8J8McC5LyLV+HJGBRJpc/8do5SYXRJhKv/0iRHqkskf/+XTQxLrLJElUf//1GRiXSRMjY6ktkf///y6dLHRb//8S1TEFNRTMuMTAw//M4xAkAAANIAcAAAFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV" type="audio/mpeg"/>
    Your browser does not support the audio element.
</audio>





```python
play_segment(phoneme_offsets[0], 5)
```

    oʊ






<audio controls>
    <source src="data:audio/mpeg;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU5LjI3LjEwMAAAAAAAAAAAAAAA//NYwAAAAAAAAAAAAEluZm8AAAAPAAAABAAAAmQAeHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4paWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpdLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tL/////////////////////////////////AAAAAExhdmM1OS4zNwAAAAAAAAAAAAAAACQEQAAAAAAAAAJk6KKSvgAAAAAAAAAAAAAA//M4xAARuJ6wBUYYAhsu7u7uyYQQcmAwGAwGFgAAAAAABDLJgAAAAAAiYcAABAYlwQBAEJeIAxBAEAwsHAQBAMf5z/wfP/4Pg+n/o/E5//lAQBA4owHPQIxb/0jDTyNnmClMIDBDNKCTjIUF//M4xBgaovLQy49oAAh4UO0BQRYkR0oD4KA90HExRMSkaruLEoT7MEZ3GHTrKB0lSGzKTUJkfpLrN9ST0X0ei9Pp6kL/V/+dVoPqtWSKGcQau2dTepafW//1d//OlB66sAAACAoFAQFAwABt//M4xAwYUpa5/41oAAH1aOSgJ74FBIPmRWDp18IsPMFNC3L/hziTEmBaiH/44ieCmmT//1DiLSSNjAkf/8wHskdPF0nG5Jf//rakVkkPY2pF5H///9AcUyTSOmqI0id//4dqTEFNRTMuMTAw//M4xAkAAANIAcAAAKqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq" type="audio/mpeg"/>
    Your browser does not support the audio element.
</audio>





```python
play_segment(phoneme_offsets[0], 6)
```

    t






<audio controls>
    <source src="data:audio/mpeg;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU5LjI3LjEwMAAAAAAAAAAAAAAA//NYwAAAAAAAAAAAAEluZm8AAAAPAAAABAAAAmQAeHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4paWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpdLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tL/////////////////////////////////AAAAAExhdmM1OS4zNwAAAAAAAAAAAAAAACQEQAAAAAAAAAJk07YsqwAAAAAAAAAAAAAA//M4xAARsNqwF0hIAAnBtCIiIiECguLn3BAoiaECgoKUWCsAAAAQFBIgQEAoA2CYbJ25+5whDJo0aNsEAQBBxR3xOH/n/BAEDnLg/9//zn/R//B9nH/CEF7/+wsfMHtgL/Y5KgqcVBaysoA2//M4xBga4vLEq4+IAJJGINeOSkWAyAIOdghAsBq4kREi8Yh7I30GSH8fAeu9YpFkTEvvPsgYY3mlhttTvV7tdWj/V/rv+ilnFJInk6k09R7pPWc1Fg9spa7qv/97f/mR5YAQCQoGxQIBBAAJ//M4xAsX0n7GHY1oAujCemDe8ABIJevHEM4LatlcOSMkvAdv8FeIaSIxv/ADSJYMGEWHUlUnR/4RUdCTHsJKHNMP/8cRTJI6XVJGQkv//5qpReKy8YkwkTyS////yRLviJVMQU1FMy4xMDBV//M4xAoAAANIAcAAAFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV" type="audio/mpeg"/>
    Your browser does not support the audio element.
</audio>





```python
play_segment(phoneme_offsets[0], 7)
```

    ʌ






<audio controls>
    <source src="data:audio/mpeg;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU5LjI3LjEwMAAAAAAAAAAAAAAA//NYwAAAAAAAAAAAAEluZm8AAAAPAAAAAwAAAfgAkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSycnJycnJycnJycnJycnJycnJycnJycnJycnJycnJycnJ////////////////////////////////////////////AAAAAExhdmM1OS4zNwAAAAAAAAAAAAAAACQDQAAAAAAAAAH4y4HsugAAAAAAAAAAAAAA//M4xAAS0Q68pUEQAAysHwAAoAYxzGAHAAAAMDAwMDAwMDOQhPzvzn/nOcWAAAggIAgCYPg+D4PggCDpd//0ggCAIAh/lAGD4Pg+HwQBAEJcHwfB9///B9WyfJMEAHyzzzBIUmjd4WEUpJMY//M4xBMZOh7VlYtYAOggyb8j+8ZpKmXpFQ0DqD2aAnFBNiYg6mbPJy4eUTAf/+SgNU+bIpGIGYttf8GXfEuOu5r//5qnbds8+7ayP//zyNOr5+96mz/oLALVcpzv/xyquQQVJJaAVDUGlA1B//M4xA0MgAYln8AYAlO2xEWeJXHkrDSgVclZ0Swa+z8l//pLB3gr/+p5X5b4ld+5Z2pMQU1FMy4xMDCqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq" type="audio/mpeg"/>
    Your browser does not support the audio element.
</audio>





```python
play_segment(phoneme_offsets[0], 8)
```

    p






<audio controls>
    <source src="data:audio/mpeg;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU5LjI3LjEwMAAAAAAAAAAAAAAA//NYwAAAAAAAAAAAAEluZm8AAAAPAAAAAwAAAfgAkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSycnJycnJycnJycnJycnJycnJycnJycnJycnJycnJycnJ////////////////////////////////////////////AAAAAExhdmM1OS4zNwAAAAAAAAAAAAAAACQDQAAAAAAAAAH4oFBr1wAAAAAAAAAAAAAA//M4xAARYHI8ABPSAAggO3iviZmnEYBBIIwNk6N6ZOgQYkFAwgFAoeGBA7AgIAgCYPyZA5BAoCYPgc+D4IAgGBBLn4fcCHWH8MZxP//++p3/+TUGmnA0UzwvhSHQfRKlQInp68erirRtYGHI//M4xBkOSKJUFmGGqNc8BO32deHByKfFXvPfJf1+tn9an872f/yNeK7P7j3KqgGkUscoDVGhEThJZgWFhYXdAoVEh+aFKun/3s/vZ/Aop/s/rFP/1+zireritUxBTUUzLjEwMFVVVVVVVVVV//M4xD4LGC4YXhhMAFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV" type="audio/mpeg"/>
    Your browser does not support the audio element.
</audio>



