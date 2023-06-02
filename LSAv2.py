import os
import sys
import io
import json
import numpy as np
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
from sklearn.preprocessing import StandardScaler
import spacy
import moviepy.editor as mp
from datetime import datetime, timedelta


nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creds/celtic-guru-247118-9e8cccc27c4a.json"
audio_path = 'temp_audio.wav'
bucket_name = 'sermon-speech-audio'
video_file_name = sys.argv[1]
file_without_path = os.path.basename(video_file_name)
file_name_without_extension = os.path.splitext(file_without_path)[0]
original_video = VideoFileClip(video_file_name)
segment_duration = 60
TARGET_DURATION = 20 * 60;

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

def get_transcription(bucket_name, audio_path):
    transcript_file = file_name_without_extension + '.json'
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
        with open(transcript_file, 'w') as f:
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

def extract_important_segments(bucket_name, audio_path):
    sentences = get_transcription(bucket_name, audio_path)
    lsa_scores = compute_lsa_scores(sentences)
    sentiment_scores = compute_sentiment_scores(sentences)
    features = np.column_stack((lsa_scores, sentiment_scores))
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=int(original_video.duration/segment_duration), init='k-means++', random_state=42)
    kmeans.fit(features)
    cluster_centers = kmeans.cluster_centers_
    cluster_indices = np.argsort(np.linalg.norm(features - cluster_centers[:, np.newaxis], axis=2), axis=1)

    # plt.scatter(features[:, 0], features[:, 1], c=cluster_indices)
    # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, marker='*', c=matplotlib.colors.to_rgba('red'))
    # plt.xlabel('LSA score')
    # plt.ylabel('Sentiment score')
    # plt.show()

    important_segments = []
    for i in range(len(cluster_centers)):
        timestamps = []
        start_index = cluster_indices[i, 0]
        end_index = cluster_indices[i, -1]
        timestamps.append(sentences[start_index]['start_time'])
        timestamps.append(sentences[end_index]['end_time'])
        timestamps.sort()
        important_segments.append({'start_time': timestamps[0], 'end_time': timestamps[1]})

    # print('the shape of the important segments is' + important_segments)
    pdb.set_trace()
    return important_segments

def summarize_whole_segments(bucket_name, audio_path, threshold=0.8):
    sentences = get_transcription(bucket_name, audio_path)
    # Create a spacy model.
    nlp = spacy.load("en_core_web_sm")

    # Create a list of sentence embeddings.
    sentence_embeddings = []
    for sentence in sentences:
        sentence_text = sentence['text']  # Extract the text from the dictionary
        sentence_embeddings.append(nlp(sentence_text).vector)

    # Calculate the similarity between each pair of sentence embeddings.
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            similarity_matrix[i][j] = np.dot(sentence_embeddings[i], sentence_embeddings[j])

    # Find the most similar sentences for each sentence.
    most_similar_sentences = []
    for i in range(len(sentences)):
        most_similar_sentences.append(np.argsort(similarity_matrix[i])[::-1])

    # Initialize variables for tracking segments
    ordered_segments = []
    used_indices = set()

    # Process sentences in descending order of similarity
    for i in range(len(sentences)):
        sentence_index = most_similar_sentences[i][0]  # Get the most similar sentence index
        if sentence_index in used_indices:
            continue  # Skip if the sentence has already been used

        segment_start = sentence_index
        segment_end = sentence_index

        # Expand the segment to include adjacent similar sentences
        while segment_end + 1 < len(sentences) and similarity_matrix[sentence_index][segment_end + 1] > threshold:
            segment_end += 1

        # Add the segment to the result
        segment_data = {
            'text': '',
            'start_time': sentences[segment_start]['start_time'],
            'end_time': sentences[segment_end]['end_time']
        }

        for j in range(segment_start, segment_end + 1):
            used_indices.add(j)
            segment_data['text'] += sentences[j]['text'] + ' '

        ordered_segments.append(segment_data)

    return ordered_segments



def summarize_segments(bucket_name, audio_path):
    sentences = get_transcription(bucket_name, audio_path)
    # Create a spacy model.
    nlp = spacy.load("en_core_web_sm")

    # Create a list of sentence embeddings.
    sentence_embeddings = []
    for sentence in sentences:
        sentence_text = sentence['text']  # Extract the text from the dictionary
        sentence_embeddings.append(nlp(sentence_text).vector)

    # Calculate the similarity between each pair of sentence embeddings.
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            similarity_matrix[i][j] = np.dot(sentence_embeddings[i], sentence_embeddings[j])

    # Find the most similar sentences for each sentence.
    most_similar_sentences = []
    for i in range(len(sentences)):
        most_similar_sentences.append(np.argsort(similarity_matrix[i])[::-1][:2])

    # Order the sentences so that the sentences that are most similar to each other are grouped together.
    ordered_sentences = []
    for i in range(len(sentences)):
        ordered_sentence_indices = most_similar_sentences[i]
        ordered_sentence_data = []
        for index in ordered_sentence_indices:
            sentence_data = {
                'text': sentences[index]['text'],
                'start_time': sentences[index]['start_time'],
                'end_time': sentences[index]['end_time']
            }
            ordered_sentence_data.append(sentence_data)
        ordered_sentences.extend(ordered_sentence_data)

    return ordered_sentences




def condense_segment(segment):
    start_time = segment['start_time']
    end_time = segment['end_time']
    video_segment = original_video.subclip(start_time, end_time)
    return video_segment

def concatenate_segments(segments):
    clips = []
    for segment in segments:
        condensed_segment = condense_segment(segment)
        if condensed_segment.duration <= 0:
            continue
        clips.append(condensed_segment)
    if not clips:
        return None
    final_clip = concatenate_videoclips(clips, method='compose')
    return final_clip


def create_summary_video(segments, target_duration):
    summary_segments = []
    total_duration = timedelta(seconds=0)
    target_duration = timedelta(seconds=target_duration)
    index = 0

    while index < len(segments) and total_duration < target_duration:
        segment = segments[index]
        if total_duration + calculate_segment_duration(segment) <= target_duration:
            # Add the segment to the summary video
            summary_segments.append(segment)
            total_duration += calculate_segment_duration(segment)
        else:
            pdb.set_trace()
            remaining_duration = target_duration - total_duration
            trimmed_segment = trim_segment(segment, remaining_duration)
            summary_segments.append(trimmed_segment)
            total_duration += calculate_segment_duration(trimmed_segment)
        index += 1

    sorted_segments = sorted(summary_segments, key=lambda s: s["start_time"])
    summary_video = concatenate_segments(sorted_segments)

    return summary_video

def calculate_segment_duration(segment):
    start_time = timedelta(seconds=segment["start_time"])
    end_time = timedelta(seconds=segment["end_time"])
    return end_time - start_time

def get_longest_subsegment(segment, max_duration):
    if calculate_segment_duration(segment) <= max_duration:
        return segment
    else:
        # Find the longest subsegment that fits in the remaining time
        subsegment = segment.copy()
        subsegment["end_time"] = segment["start_time"] + max_duration
        return subsegment
    
def trim_segment(segment, max_duration):
    start_time = segment["start_time"]
    end_time = segment["end_time"]
    duration = calculate_segment_duration(segment)

    if duration <= max_duration:
        return segment

    # Calculate the proportion of the segment to keep
    proportion = max_duration / duration

    # Calculate the new end time
    new_end_time = start_time + (proportion * (end_time - start_time))

    # Create a new segment with the trimmed time range
    trimmed_segment = {
        "start_time": start_time,
        "end_time": new_end_time,
    }

    return trimmed_segment

def trim_segments(segments):
    # Create a dictionary to map start times to segments
    segment_dict = {segment["start_time"]: segment for segment in segments}

    # Sort the start times in ascending order
    start_times = sorted(segment_dict.keys())

    # Iterate over the start times and trim each segment
    trimmed_segments = []
    previous_end_time = 0
    for start_time in start_times:
        segment = segment_dict[start_time]

        # If the segment overlaps with the previous one, trim it
        if segment["start_time"] < previous_end_time:
            segment["start_time"] = previous_end_time

        # Add the trimmed segment to the list
        trimmed_segments.append(segment)
        previous_end_time = segment["end_time"]

    # Restore the original order of the segments
    sorted_segments = sorted(trimmed_segments, key=lambda s: segments.index(s))

    return sorted_segments


video = mp.VideoFileClip(video_file_name)
audio = video.audio.write_audiofile(audio_path)

# Load audio file
with io.open(audio_path, 'rb') as audio_file:
    content = audio_file.read()


important_segments = summarize_whole_segments(bucket_name, audio_path)
condensed_segments = [] # might remove this in a bit
trimmed_segments = trim_segments(important_segments)
summary_video = create_summary_video(trimmed_segments, TARGET_DURATION)
# summary_video = concatenate_segments(important_segments, 60)
summary_video.write_videofile('summary_video.mp4', fps=24, codec='libx264', audio_codec='aac')