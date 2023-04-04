import os
import io
import re
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
    # client = speech.SpeechClient()
    # speech_audio = speech.RecognitionAudio(uri='gs://sermon-speech-audio/temp_audio.wav')
    # config = speech.RecognitionConfig(
    #     encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    #     sample_rate_hertz=44100,
    #     language_code='en-US',
    #     audio_channel_count = 2,
    #     enable_automatic_punctuation=True,)
    # operation = client.long_running_recognize(config=config, audio=speech_audio)
    # print("Waiting for operation to complete...")
    # response = operation.result(timeout=5400)

    # transcript = ""
    # for result in response.results:
    #     transcript += result.alternatives[0].transcript.strip() + " "

    # transcript_file = open("transcript.txt", "w")
    transcript_file = open("transcript.txt", "r")
    text = transcript_file.read()
    # transcript_file.write(transcript)
    transcript_file.close()

    return text
    # return transcript

def compute_lsa_scores(sentences):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences).astype(float)
    lsa = TruncatedSVD(n_components=1, algorithm='arpack', random_state=42)
    lsa_scores = np.abs(lsa.fit_transform(X))
    return lsa_scores

def compute_sentiment_scores(sentences):
    client = language_v1.LanguageServiceClient()
    sentiment_scores = []
    for sentence in sentences:
        document = language_v1.Document(content=sentence, type_=language_v1.Document.Type.PLAIN_TEXT, language='en')
        sentiment = client.analyze_sentiment(request={'document': document}).document_sentiment.score
        sentiment_scores.append(sentiment)
    return sentiment_scores

def extract_important_segments():
    #upload_to_bucket(bucket_name, audio_path)
    transcript = get_transcription()
    sentences = re.split('[.!?]', transcript)
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
        end_index = start_index + next((i for i in range(start_index, len(sentences) - 1) if len(sentences[i]) > 1 and len(sentences[i+1]) > 1), len(sentences) - 1) - start_index
        start_time = sum(len(sentences[j]) + 1 for j in range(start_index))
        end_time = sum(len(sentences[j]) + 1 for j in range(end_index + 1))
        important_segments.append({'start_time': start_time, 'end_time': end_time})
    return important_segments



def condense_segment(segment, duration):
    start_time = segment['start_time']
    end_time = segment['end_time']
    video_segment = original_video.subclip(start_time, end_time)
    condensed_segment = video_segment.fx(volumex, 0.5).set_duration(duration)
    return condensed_segment

video = mp.VideoFileClip(video_file_name)
audio = video.audio.write_audiofile(audio_path)

# with io.open(audio_path, 'rb') as audio_file:
#     content = audio_file.read()

# #upload_to_bucket(bucket_name, audio_path)
# summarize_video(input_path)

important_segments = extract_important_segments()
condensed_segments = []
for segment in important_segments:
    condensed_segment = condense_segment(segment, segment_duration)
    condensed_segments.append(condensed_segment)
summary_video = concatenate_videoclips(condensed_segments)
summary_video.write_videofile('summary_video.mp4')