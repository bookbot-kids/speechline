# Load Canary model
from nemo.collections.asr.models import EncDecMultiTaskModel
canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')
import os

def get_wav_files(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.wav')]

wav_files = get_wav_files('/home/s44504/3b01c699-3670-469b-801f-13880b9cac56/speechline/bookbot_en/en-ar')

# Transcribe
transcript = canary_model.transcribe(audio=["path_to_audio_file.wav"])
# By default, Canary assumes that input audio is in English and transcribes it.
 
# To transcribe in a different language, such as Spanish
transcript = canary_model.transcribe(
     audio=["path_to_spanish_audio_file.wav"],
     batch_size=1,
     task='asr')
 