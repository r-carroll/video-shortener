import os
import io
import re
import json
import math
import numpy as np
import moviepy.editor as mp
from moviepy.audio.fx import volumex
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import nltk
from google.cloud import speech
from google.cloud import storage
from google.cloud import language_v1
from nltk.corpus import stopwords
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
import nltk
import pdb


nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creds/celtic-guru-247118-9e8cccc27c4a.json"
audio_path = 'temp_audio.wav'
bucket_name = 'sermon-speech-audio'
video_file_name = 'By-Faith.mp4'
original_video = VideoFileClip(video_file_name)
segment_duration = 60

def upload_to_bucket(bucket_name, file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    # if blob.exists():
    #     blob.delete()

    blob.chunk_size = 5 * 1024 * 1024 # Set 5 MB blob size so it doesn't timeout

    # blob.upload_from_filename(file_name)

    print(f"File {file_name} uploaded to {file_name}.")

    return blob.public_url

def get_transcription():
    transcript_file = 'sentences.json'
    client = speech.SpeechClient()
    speech_audio = speech.RecognitionAudio(uri='gs://sermon-speech-audio/temp_audio.wav')
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code='en-US',
        audio_channel_count = 2,
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True)
    
    if os.path.exists(transcript_file):
        with open(transcript_file, "r") as f:
            sentences = json.load(f)
    else:
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
        with open('sentences.json', 'w') as f:
            json.dump(sentences, f)

    return sentences

def compute_lsa_scores(sentences):
    texts = [sentence['text'] for sentence in sentences]
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts).astype(float)
    lsa = TruncatedSVD(n_components=1, algorithm='arpack', random_state=42)
    lsa_scores = np.abs(lsa.fit_transform(X))
    return lsa_scores

def compute_sentiment_scores(sentences):
    texts = [sentence['text'] for sentence in sentences]
    client = language_v1.LanguageServiceClient()
    sentiment_scores = []
    for text in texts:
        document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT, language='en')
        sentiment = client.analyze_sentiment(request={'document': document}).document_sentiment.score
        sentiment_scores.append(sentiment)
    return sentiment_scores

def extract_important_segments():
    sentences = get_transcription()
    lsa_scores = compute_lsa_scores(sentences)
    sentiment_scores = compute_sentiment_scores(sentences)
    features = np.column_stack((lsa_scores, sentiment_scores))
    kmeans = KMeans(n_clusters=int(original_video.duration/segment_duration), random_state=42)
    kmeans.fit(features)
    cluster_centers = kmeans.cluster_centers_
    cluster_indices = np.argsort(np.linalg.norm(features - cluster_centers[:, np.newaxis], axis=2), axis=1)
    important_segments = []
    for i in range(len(cluster_centers)):
        start_index = cluster_indices[i, 0]
        end_index = cluster_indices[i, -1]
        start_time = sentences[start_index]['start_time']
        end_time = sentences[end_index]['end_time']
        important_segments.append({'start_time': start_time, 'end_time': end_time})
    return important_segments


def condense_segment(segment, duration):
    start_time = segment['start_time']
    end_time = segment['end_time']
    video_segment = original_video.subclip(start_time, end_time)
    # condensed_segment = video_segment.set_duration(duration)
    return video_segment

def concatenate_segments(segments, duration):
    clips = []
    for segment in segments:
        condensed_segment = condense_segment(segment, duration)
        if condensed_segment.duration <= 0:
            continue
        clips.append(condensed_segment)
    if not clips:
        return None
    final_clip = concatenate_videoclips(clips, method='compose')
    return final_clip

video = mp.VideoFileClip(video_file_name)
audio = video.audio.write_audiofile(audio_path)

# with io.open(audio_path, 'rb') as audio_file:
#     content = audio_file.read()

# #upload_to_bucket(bucket_name, audio_path)
# summarize_video(input_path)

important_segments = extract_important_segments()
condensed_segments = []
# for segment in important_segments:
#     condensed_segment = condense_segment(segment, segment_duration)
#     condensed_segments.append(condensed_segment)
# pdb.set_trace()
summary_video = concatenate_segments(important_segments, 60)
summary_video.write_videofile('summary_video.mp4', fps=24, codec='libx264', audio_codec='aac')