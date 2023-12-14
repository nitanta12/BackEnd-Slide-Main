import os
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

load_dotenv()

speech_key = os.getenv('SPEECH_KEY')
service_region = os.getenv('SERVICE_REGION')

speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat["Riff24Khz16BitMonoPcm"])

audio_loc = 'output/audio'
speech_config.speech_synthesis_voice_name = "en-US-AriaNeural"
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

def generate_audio_from_text(text, audio_path):
    result = speech_synthesizer.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        # Saves the audio data to a WAV file
        with open(audio_path, "wb") as wav_file:
            wav_file.write(result.audio_data)
        print("Speech synthesized to wave file successfully")
    else:
        print("Error synthesizing speech: {}".format(result.reason))

def generate_for_home(document):
    speaker_notes = document["title"] + ' the author of the article is:' + document["author"][0]
    generate_audio_from_text(speaker_notes, os.path.join(audio_loc, 'frame_0.wav'))
    speaker_notes = 'Here is the image for the article .                   .          .'
    generate_audio_from_text(speaker_notes, os.path.join(audio_loc, 'frame_1.wav'))

def synthesize_audio(document):
    os.mkdir(audio_loc)
    generate_for_home(document)
    k=0                             
    for topic, contents in document['slides'].items():
        for num, sentences in contents.items():
            if num!=-1:
                speaker_notes = ''.join(sentences)
                audio_path = os.path.join(audio_loc, 'frame_{}.wav'.format(k+2))
                generate_audio_from_text(speaker_notes, audio_path)
                k+=1
