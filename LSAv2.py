import os
import sys
import io
import json
import operator
import random
import ssl
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import nltk
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
import argparse

import spacy
from datetime import timedelta
import moviepy.editor as mp

# local imports
import AudioProcessing
import TextProcessing


ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creds/celtic-guru-247118-9e8cccc27c4a.json"
audio_path = 'temp_audio.wav'
parser = argparse.ArgumentParser()
parser.add_argument("command")
parser.add_argument("video_file_name")
parser.add_argument("--skip-insights", action="store_true")
parser.add_argument("--skip-video", action="store_true")
parser.add_argument("--skip-srt", action="store_true")
args = parser.parse_args()
command = args.command
video_file_name = args.video_file_name
# video_file_name = sys.argv[2]
# command = sys.argv[1]
file_without_path = os.path.basename(video_file_name)
folder_name = os.path.splitext(file_without_path)[0]
if not os.path.exists(folder_name):
    os.mkdir(folder_name)
original_video = VideoFileClip(video_file_name)
segment_duration = 60
TARGET_DURATION = 20 * 60;


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

def order_by_relevance(sentences):
    segments = []

    print("determining major topics and ordering sentences")
    for sentence in sentences:
        relevance_score = calculate_relevance(sentence['text'], sentences)
        sentiment_score = abs(compute_sentiment_score(sentence['text']))
        combined_score = relevance_score * (sentiment_score * 0.75)
        segments.append({
            'text': sentence['text'],
            'relevance': combined_score,
            'start': sentence['start'],
            'end': sentence['end']
        })

    segments.sort(key=operator.itemgetter('relevance'), reverse=True)
    ordered_transcript = f"{folder_name}/ordered_transcript.json"
    with open(ordered_transcript, 'w') as f:
            json.dump(segments, f)
    return segments


def meme_segments(audio_path, chunk_duration=1.0):
    sentences = AudioProcessing.get_whisper_transcription(audio_path, folder_name)

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
            start_time = sentence_data['start']
            end_time = sentence_data['end']
            duration = end_time - start_time
            num_chunks = int(np.ceil(duration / chunk_duration))

            if num_chunks > 1:
                chunk_duration_actual = duration / num_chunks
                for chunk_index in range(num_chunks):
                    chunk_start_time = start_time + chunk_index * chunk_duration_actual
                    chunk_end_time = chunk_start_time + chunk_duration_actual
                    chunk_data = {
                        'text': sentence_data['text'],
                        'start': chunk_start_time,
                        'end': chunk_end_time
                    }
                    ordered_sentence_data.append(chunk_data)
            else:
                ordered_sentence_data.append(sentence_data)

        ordered_sentences.extend(ordered_sentence_data)

    return ordered_sentences



def condense_segment(segment, video):
    start_time = segment['start']
    end_time = segment['end']
    video_segment = video.subclip(start_time, end_time)
    return video_segment

def concatenate_segments(segments, video):
    clips = []
    for segment in segments:
        condensed_segment = condense_segment(segment, video)
        if condensed_segment.duration <= 0:
            continue
        clips.append(condensed_segment)
    if not clips:
        return None
    final_clip = concatenate_videoclips(clips, method='compose')
    return final_clip


def create_summary_video(segments, target_duration, video, shuffle=False):
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

    sorted_segments = sorted(summary_segments, key=lambda s: s["start"])

    if shuffle:
        random.shuffle(sorted_segments)

    summary_video = concatenate_segments(sorted_segments, video)

    return summary_video

def write_segment_files(segments, video):
    index = 0

    if not os.path.exists(f"{folder_name}/shorts"):
        os.mkdir(f"{folder_name}/shorts")

    while index < 5 or index >= len(segments):
        start_time = segments[index]['start']
        end_time = segments[index]['end']
        segment_clip = video.subclip(start_time, end_time)
        segment_filename = f"segment_{start_time}_{end_time}.mp4"
        segment_clip.write_videofile(f"{folder_name}/shorts/{segment_filename}", codec="libx264", audio_codec='aac')

        # Close the segment clip
        segment_clip.close()

        index+=1

    # Close the original video
    video.close()


def calculate_segment_duration(segment):
    start_time = timedelta(seconds=segment["start"])
    end_time = timedelta(seconds=segment["end"])
    return end_time - start_time

def get_longest_subsegment(segment, max_duration):
    if calculate_segment_duration(segment) <= max_duration:
        return segment
    else:
        # Find the longest subsegment that fits in the remaining time
        subsegment = segment.copy()
        subsegment["end"] = segment["start"] + max_duration
        return subsegment
    
def trim_segment(segment, max_duration):
    start_time = segment["start"]
    end_time = segment["end"]
    duration = calculate_segment_duration(segment)

    if duration <= max_duration:
        return segment

    # Calculate the proportion of the segment to keep
    proportion = max_duration / duration

    # Calculate the new end time
    new_end_time = start_time + (proportion * (end_time - start_time))

    # Create a new segment with the trimmed time range
    trimmed_segment = {
        "start": start_time,
        "end": new_end_time,
    }

    return trimmed_segment

def trim_segments(segments):
    # Create a dictionary to map start times to segments
    segment_dict = {segment["start"]: segment for segment in segments}

    # Sort the start times in ascending order
    start_times = sorted(segment_dict.keys())

    # Iterate over the start times and trim each segment
    trimmed_segments = []
    previous_end_time = 0
    for start_time in start_times:
        segment = segment_dict[start_time]

        # If the segment overlaps with the previous one, trim it
        if segment["start"] < previous_end_time:
            segment["start"] = previous_end_time

        # Add the trimmed segment to the list
        trimmed_segments.append(segment)
        previous_end_time = segment["end"]

    # Restore the original order of the segments
    sorted_segments = sorted(trimmed_segments, key=lambda s: segments.index(s))

    return sorted_segments

original_audio = original_video.audio.write_audiofile(audio_path)

# Load audio file
with io.open(audio_path, 'rb') as audio_file:
    content = audio_file.read()

important_segments = []

# Operations needed for all flows
transcripts = AudioProcessing.get_whisper_transcription(audio_path, folder_name)
sentences_withtime = transcripts[0]
text_only = transcripts[1]
if not args.skip_insights:
    TextProcessing.get_insights(sentences_withtime, text_only, folder_name)


    

if command == "summarize":
    print("building summary manifest")
    important_segments = order_by_relevance(sentences_withtime)
    trimmed_segments = trim_segments(important_segments)
    summary_video = create_summary_video(trimmed_segments, TARGET_DURATION, original_video)
    summary_video.write_videofile(f"{folder_name}/summary_video.mp4", fps=24, codec='libx264', audio_codec='aac')
if command == "clean" or command == "all":
    print("starting audio cleaning workflow")
    AudioProcessing.remove_empty_space(audio_path, folder_name)
if command == "unclean":
    chunks = AudioProcessing.all_silent(audio_path)
    silent_video = AudioProcessing.clips_to_video(original_video, chunks)
    silent_video.write_videofile(f"{folder_name}/silent_video.mp4", codec="libx264", audio_codec='aac')
if command == "shorts" or command == "all":
    print("starting shorts workflow")
    # with open(f'{folder_name}/shorts.json') as f:
    #     viral_segments = json.load(f)
    important_segments = order_by_relevance(sentences_withtime)
    write_segment_files(important_segments, original_video)
if command == "meme":
    important_segments = meme_segments(audio_path)
    trimmed_segments = trim_segments(important_segments)
    summary_video = create_summary_video(trimmed_segments, TARGET_DURATION, original_video)
    summary_video.write_videofile(f"{folder_name}/meme_video.mp4", fps=24, codec='libx264', audio_codec='aac')
if command == "caption" or command == "all":
    if not args.skip_video:
        print("starting caption workflow")
        include_srt = not args.skip_srt
        TextProcessing.generate_subtitles(sentences_withtime, folder_name, original_video, include_srt)
else:
    print(f"Available commands: \n shorts: generate 5 short-format videos" +
          f"\n summarize: generate a 20 minute summary video" +
          f"\n clean: remove pauses and empty space, cleans up audio" +
          f"\n unclean: only the pauses" +
          f"\n caption: generate SRT files for English, Spanish, and Portuguese" +
          f"\n all: cleans, shorts, and captions" +
          f"\n meme: just make a disaster")
