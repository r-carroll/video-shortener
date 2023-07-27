import spacy
import json
import os
import ssl
import datetime
import srt
from googletrans import Translator
from google.cloud import speech, storage
import openai
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import moviepy.video.fx.all as vfx
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader, JSONLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI

ssl._create_default_https_context = ssl._create_unverified_context

def load_api_key():
    with open('creds/celtic-guru-247118-9e8cccc27c4a.json', 'r') as json_file:
        data = json.load(json_file)
        api_key = data.get('open_ai_key')
    return api_key

def is_end_sentence(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return doc[-1].is_punct

def concatenate_sentences(sentences):
    complete_sentences = []
    current_sentence = sentences[0]["text"]
    start_times = [sentences[0]["start_time"]]
    end_times = [sentences[0]["end_time"]]
    for sentence in sentences:
        if len(current_sentence) == 0:
            current_sentence = sentence["text"]
            start_times.append(sentence["start_time"])
            end_times.append(sentence["end_time"])
        
        if is_end_sentence(current_sentence):
            complete_sentences.append({"text": current_sentence, "start_time": min(start_times), "end_time": max(end_times)})
            current_sentence = ""
            start_times = []
            end_times = []
        elif current_sentence != sentence["text"]:
            current_sentence += " " + sentence["text"]
            start_times.append(sentence["start_time"])
            end_times.append(sentence["end_time"])

    return complete_sentences

def whisper_concatenate_sentences(sentences):
    complete_sentences = []
    current_sentence = sentences[0]["text"]
    start_times = [sentences[0]["start"]]
    end_times = [sentences[0]["end"]]
    for sentence in sentences:
        if len(current_sentence) == 0:
            current_sentence = sentence["text"]
            start_times.append(sentence["start"])
            end_times.append(sentence["end"])
        
        if is_end_sentence(current_sentence):
            complete_sentences.append({"text": current_sentence, "start": min(start_times), "end": max(end_times)})
            current_sentence = ""
            start_times = []
            end_times = []
        elif current_sentence != sentence["text"]:
            current_sentence += " " + sentence["text"]
            start_times.append(sentence["start"])
            end_times.append(sentence["end"])

    return complete_sentences

def upload_to_bucket(bucket_name, file_name):
    print("uploading audio to bucket")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    if blob.exists():
        blob.delete()

    blob.chunk_size = 5 * 1024 * 1024 # Set 5 MB blob size so it doesn't timeout

    blob.upload_from_filename(file_name)

    print(f"File {file_name} uploaded to {file_name}.")

    return blob.public_url

def get_transcription(bucket_name, audio_path, folder_name):
    transcript_file = f"{folder_name}/{folder_name}.json"
    client = speech.SpeechClient()
    speech_audio = speech.RecognitionAudio(uri='gs://sermon-speech-audio/temp_audio.wav')
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code='en-US',
        audio_channel_count = 2,
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True,
        model='video',
        use_enhanced=True)
    
    if os.path.exists(transcript_file):
        with open(transcript_file, "r") as f:
            sentences = json.load(f)
    else:
        upload_to_bucket(bucket_name, audio_path)
        operation = client.long_running_recognize(config=config, audio=speech_audio)
        print("Waiting for operation to complete...")
        response = operation.result(timeout=5400)
        sentences = []
        for result in response.results:
            alternatives = result.alternatives
            
            for alternative in alternatives:
                sentence = alternative.transcript.strip()
                if sentence:
                    sentences.append({
                        'text': sentence,
                        'start_time': alternative.words[0].start_time.total_seconds(),
                        'end_time': alternative.words[-1].end_time.total_seconds()
                    })
        sentences = concatenate_sentences(sentences)
        with open(transcript_file, 'w') as f:
            json.dump(sentences, f)

    return sentences

def translate_srt(sentences, target_language, folder):
  translator = Translator()
  translated_sentences = []
  translated_texts = [sentence['text'] for sentence in sentences]
  if target_language != 'en':
    translated_texts = translator.translate(translated_texts , dest=target_language)

  for index, sentence in enumerate(sentences):
    target_text = ""
    if target_language == 'en':
        target_text = translated_texts[index]
    else:
        target_text = translated_texts[index].text
    
    start_timedelta = datetime.timedelta(seconds=sentence["start"])
    end_timedelta = datetime.timedelta(seconds=sentence["end"])
    translated_sentences.append(
        srt.Subtitle(
            index=index,
            content=target_text,
            start=start_timedelta,
            end=end_timedelta))

  translated_srt = srt.compose(translated_sentences)

  with open(f"{folder}/{target_language}.srt", "w") as f:
    f.write(translated_srt)

  return translated_srt

def generate_subtitles(sentences, folder, video):
    target_languages = ['en', 'es', 'pt']

    for language in target_languages:
        translate_srt(sentences, language, folder)

    generator = lambda text: TextClip(text, font='Helvetica-bold', method='caption',
                                      fontsize=36, color='white', size=(video.w * 0.8, None), bg_color='black')
    sub = SubtitlesClip(f"{folder}/en.srt", generator)
    # sub = vfx.fadein(sub, 2)
    final = CompositeVideoClip([video, sub.set_position((.1, .9), relative=True)])
    final = final.set_audio(video.audio)
    final.write_videofile(f"{folder}/subbed.mp4", fps=video.fps, audio_codec='aac')

def gpt_viral_segments(sentences, folder):
    openai.api_key = load_api_key()
    return
