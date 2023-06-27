import spacy
import json
import os
from google.cloud import speech, storage

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

def upload_to_bucket(bucket_name, file_name):
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
