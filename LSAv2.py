import os
import sys
import io
import json
import operator
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

import spacy
import moviepy.editor as mp
from datetime import timedelta
import moviepy.editor as mp
from pydub import AudioSegment, silence


nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creds/celtic-guru-247118-9e8cccc27c4a.json"
audio_path = 'temp_audio.wav'
bucket_name = 'sermon-speech-audio'
video_file_name = sys.argv[2]
command = sys.argv[1]
file_without_path = os.path.basename(video_file_name)
file_name_without_extension = os.path.splitext(file_without_path)[0]
if not os.path.exists(file_name_without_extension):
    os.mkdir(file_name_without_extension)
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

def compute_sentiment_scores(sentences):
    texts = [sentence['text'] for sentence in sentences]
    client = language_v1.LanguageServiceClient()
    sentiment_scores = []
    for text in texts:
        document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT, language='en')
        sentiment = client.analyze_sentiment(request={'document': document}).document_sentiment.score
        sentiment_scores.append(sentiment)
    return sentiment_scores


def calculate_relevance(text, sentences):
    # Load the spaCy model for English language processing
    nlp = spacy.load("en_core_web_sm")

    # Tokenize and process each segment in the transcript
    transcript_docs = [nlp(segment['text']) for segment in sentences]

    # Extract the text from transcript docs
    transcript_texts = [doc.text for doc in transcript_docs]

    # Pad the transcript texts to the length of the longest segment
    np_texts = np.array(transcript_texts)
    corpus = transcript_texts + [text]

    # Apply TF-IDF vectorization to the corpus
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Calculate cosine similarity between the text and the whole transcript
    text_tfidf = tfidf_matrix[-1]
    transcript_tfidf = tfidf_matrix[:-1]
    similarity = cosine_similarity(text_tfidf, transcript_tfidf)

    average_similarity = np.mean(similarity)

    return average_similarity


def order_by_relevance(bucket_name, audio_path):
    segments = []
    sentences = get_transcription(bucket_name, audio_path)

    for sentence in sentences:
        relevance_score = calculate_relevance(sentence['text'], sentences)
        segments.append({
            'text': sentence['text'],
            'relevance': relevance_score,
            'start_time': sentence['start_time'],
            'end_time': sentence['end_time']
        })

    segments.sort(key=operator.itemgetter('relevance'), reverse=True)
    return segments

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




def meme_segments(bucket_name, audio_path):
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
            remaining_duration = target_duration - total_duration
            trimmed_segment = trim_segment(segment, remaining_duration)
            summary_segments.append(trimmed_segment)
            total_duration += calculate_segment_duration(trimmed_segment)
        index += 1

    sorted_segments = sorted(summary_segments, key=lambda s: s["start_time"])
    summary_video = concatenate_segments(sorted_segments)

    return summary_video

def write_segment_files(segments, video):
    index = 0
    while index < 5 or index >= len(segments):
        start_time = segments[index]['start_time']
        end_time = segments[index]['end_time']
        segment_clip = video.subclip(start_time, end_time)
        segment_filename = f"segment_{start_time}_{end_time}.mp4"
        segment_clip.write_videofile(file_name_without_extension + '/' + segment_filename, codec="libx264", audio_codec='aac')

        # Close the segment clip
        segment_clip.close()

        index+=1

    # Close the original video
    video.close()


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

def remove_empty_space(temp_audio_path):
    audio = AudioSegment.from_wav(temp_audio_path)

    # Split the audio into chunks based on silence
    chunks = []
    silence_thresh = -50
    min_silence_duration = 1200
    audio_chunks = silence.split_on_silence(
        audio,
        min_silence_len=min_silence_duration,
        silence_thresh=silence_thresh
    )

    # Iterate through the audio chunks and determine their duration
    for chunk in audio_chunks:
        chunk_duration = len(chunk) / 1000  # Convert to seconds

        # Add the chunk to the list if it is not considered empty space
        if chunk_duration > min_silence_duration / 1000:
            chunks.append(chunk)

    # Concatenate the non-empty audio chunks with half a second of silence in between
    modified_audio = chunks[0]
    for chunk in chunks[1:]:
        silence_segment = AudioSegment.silent(duration=500)  # Half a second of silence
        modified_audio += silence_segment + chunk

    modified_audio.export('summary.wav', format="wav")



video = mp.VideoFileClip(video_file_name)
audio = video.audio.write_audiofile(audio_path)

# Load audio file
with io.open(audio_path, 'rb') as audio_file:
    content = audio_file.read()

if command == "summarize":
    important_segments = summarize_whole_segments(bucket_name, audio_path)
    trimmed_segments = trim_segments(important_segments)
    summary_video = create_summary_video(trimmed_segments, TARGET_DURATION)
    summary_video.write_videofile('summary_video.mp4', fps=24, codec='libx264', audio_codec='aac')
elif command == "clean":
    remove_empty_space(audio_path)
elif command == "shorts":
    segments = order_by_relevance(bucket_name, audio_path)
    write_segment_files(segments, video)

elif command == "meme":
    important_segments = meme_segments(bucket_name, audio_path)
    trimmed_segments = trim_segments(important_segments)
    summary_video = create_summary_video(trimmed_segments, TARGET_DURATION)
    summary_video.write_videofile('summary_video.mp4', fps=24, codec='libx264', audio_codec='aac')
