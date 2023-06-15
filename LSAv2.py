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
from nltk.sentiment import SentimentIntensityAnalyzer

import spacy
import moviepy.editor as mp
from datetime import timedelta
import moviepy.editor as mp
from pydub import AudioSegment, silence
import translation

from pytube import YouTube


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nlp = spacy.load("en_core_web_sm")
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

def clean_sentences(sentences):
    new_sentences = []

    # Iterate over the sentences
    i = 0
    while i + 1 < len(sentences):
        new_sentence = {}
        current_sentence = sentences[i]
        current_text = current_sentence['text']
        current_start_time = current_sentence['start_time']
        current_end_time = current_sentence['end_time']

        next_sentence = sentences[i + 1]
        next_text = next_sentence['text']
        next_start_time = next_sentence['start_time']
        next_end_time = next_sentence['end_time']

        # Process the current sentence text with spaCy
        current_doc = nlp(current_text)
        next_doc = nlp(next_text)

        # Check if the last token is the end of a sentence
        #print(doc[-1])
        if not current_doc[-1].text.endswith(('.', '!', '?')):
            # Concatenate with the next sentence if it exists
            if i + 1 < len(sentences):
                # Concatenate the texts, update the end time
                current_text += ' ' + next_text
                new_sentence = {'text': current_text, 'start_time': current_start_time, 'end_time': next_end_time}
                i += 1 # skip next sentence
                new_sentences.append(new_sentence)
        else:
            new_sentences.append(current_sentence)

        # Move to the next sentence
        i += 1

    return new_sentences


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
        sentences = clean_sentences(sentences)
        with open(transcript_file, 'w') as f:
            json.dump(sentences, f)

    return sentences

def calculate_relevance(text, sentences):
    transcript_docs = [nlp(segment['text']) for segment in sentences]
    transcript_texts = [doc.text for doc in transcript_docs]
    corpus = transcript_texts + [text]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    text_tfidf = tfidf_matrix[-1]
    transcript_tfidf = tfidf_matrix[:-1]
    similarity = cosine_similarity(text_tfidf, transcript_tfidf)

    average_similarity = np.mean(similarity)

    return average_similarity

def compute_sentiment_score(text):
    sentiment_analyzer = SentimentIntensityAnalyzer()
    return sentiment_analyzer.polarity_scores(text)['compound']

def order_by_relevance(bucket_name, audio_path):
    segments = []
    sentences = get_transcription(bucket_name, audio_path)

    for index, sentence in enumerate(sentences):
        relevance_score = calculate_relevance(sentence['text'], sentences)
        sentiment_score = abs(compute_sentiment_score(sentence['text']))
        combined_score = relevance_score * sentiment_score
        segments.append({
            'text': sentence['text'],
            'relevance': combined_score,
            'start_time': sentence['start_time'],
            'end_time': sentence['end_time']
        })

    segments.sort(key=operator.itemgetter('relevance'), reverse=True)
    return segments

def summarize_whole_segments(bucket_name, audio_path, threshold=0.8):
    sentences = get_transcription(bucket_name, audio_path)

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




def meme_segments(bucket_name, audio_path, chunk_duration=1.0):
    sentences = get_transcription(bucket_name, audio_path)

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

    # Split sentences into smaller chunks based on chunk_duration.
    ordered_sentences = []
    for i in range(len(sentences)):
        ordered_sentence_indices = most_similar_sentences[i]
        ordered_sentence_data = []
        for index in ordered_sentence_indices:
            sentence_data = sentences[index]
            start_time = sentence_data['start_time']
            end_time = sentence_data['end_time']
            duration = end_time - start_time
            num_chunks = int(np.ceil(duration / chunk_duration))

            if num_chunks > 1:
                chunk_duration_actual = duration / num_chunks
                for chunk_index in range(num_chunks):
                    chunk_start_time = start_time + chunk_index * chunk_duration_actual
                    chunk_end_time = chunk_start_time + chunk_duration_actual
                    chunk_data = {
                        'text': sentence_data['text'],
                        'start_time': chunk_start_time,
                        'end_time': chunk_end_time
                    }
                    ordered_sentence_data.append(chunk_data)
            else:
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
        segment_clip.write_videofile(f"{file_name_without_extension}/{segment_filename}", codec="libx264", audio_codec='aac')

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
    #sorted_segments = sorted(trimmed_segments, key=lambda s: segments.index(s))

    return trimmed_segments

def remove_empty_space(temp_audio_path):
    audio = AudioSegment.from_wav(temp_audio_path)
    chunks = []
    silence_thresh = -50
    min_silence_duration = 1200
    audio_chunks = silence.split_on_silence(
        audio,
        min_silence_len=min_silence_duration,
        silence_thresh=silence_thresh
    )

    for chunk in audio_chunks:
        chunk_duration = len(chunk) / 1000  # Convert to seconds
        if chunk_duration > min_silence_duration / 1000:
            chunks.append(chunk)

    modified_audio = chunks[0]
    for chunk in chunks[1:]:
        silence_segment = AudioSegment.silent(duration=500)  # Half a second of silence
        modified_audio += silence_segment + chunk

    modified_audio.export('summary.wav', format="wav")

def all_silent(temp_audio_path):
    audio = AudioSegment.from_wav(temp_audio_path)
    silence_thresh = -50
    min_silence_duration = 1200
    audio_chunks = silence.detect_silence(
        audio,
        min_silence_len=min_silence_duration,
        silence_thresh=silence_thresh
    )

    silent_chunks = []
    for start, end in audio_chunks:
        chunk_start_time = start / 1000  # Convert to seconds
        chunk_end_time = end / 1000  # Convert to seconds
        silent_chunks.append((chunk_start_time, chunk_end_time))

    return silent_chunks

def generate_silent_video(video, silent_chunks):
    clips = []

    for start_time, end_time in silent_chunks:
        silent_clip = video.subclip(start_time, end_time)
        clips.append(silent_clip)

    silent_video = concatenate_videoclips(clips)
    silent_video.write_videofile("silent_video.mp4", codec="libx264", audio_codec='aac')

def download_video(url):
    print("in the method")
    if ("youtube.com" not in url):
        url = input("Enter YouTube URL: ")
    yt = YouTube(url,use_oauth=True,allow_oauth_cache=True)
    filename = yt.title.replace(" ","_")
    print("Downloading YouTube File: " + yt.title)
    yt.streams.first().download(filename=filename + ".mp4")



video = mp.VideoFileClip(video_file_name)
audio = video.audio.write_audiofile(audio_path)

# Load audio file
with io.open(audio_path, 'rb') as audio_file:
    content = audio_file.read()

important_segments = []

if command == "shorts" or command == "summarize":
    important_segments = order_by_relevance(bucket_name, audio_path)

if command == "summarize":
    trimmed_segments = trim_segments(important_segments)
    summary_video = create_summary_video(trimmed_segments, TARGET_DURATION)
    summary_video.write_videofile('summary_video.mp4', fps=24, codec='libx264', audio_codec='aac')
elif command == "clean":
    remove_empty_space(audio_path)
elif command == "unclean":
    chunks = all_silent(audio_path)
    generate_silent_video(video, chunks)
elif command == "shorts":
    write_segment_files(important_segments, video)
elif command == "meme":
    important_segments = meme_segments(bucket_name, audio_path)
    trimmed_segments = trim_segments(important_segments)
    summary_video = create_summary_video(trimmed_segments, TARGET_DURATION)
    summary_video.write_videofile('summary_video.mp4', fps=24, codec='libx264', audio_codec='aac')
elif command == "caption":
    sentences = get_transcription(bucket_name, audio_path)
    translation.generate_srt_files(sentences, file_name_without_extension)
elif command == "download":
    download_video(video_file_name)
else:
    print(f"Available commands: \n shorts: generate 5 short-format videos" +
          f"\n summarize: generate a 20 minute summary video" +
          f"\n clean: remove pauses and empty space" +
          f"\n caption: generate SRT files for English, Spanish, and Portuguese" +
          f"\n meme: just make a disaster")
