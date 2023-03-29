import os
import io
import math
import numpy as np
import moviepy.editor as mp
import smile
import smile.audio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk
from google.cloud import speech
from google.cloud import storage
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

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
    #     audio_channel_count = 2)
    # operation = client.long_running_recognize(config=config, audio=speech_audio)
    # print("Waiting for operation to complete...")
    # response = operation.result(timeout=5400)

    # transcript = ""
    # for result in response.results:
    #     transcript += result.alternatives[0].transcript.strip() + " "

    transcript_file = open("transcript.txt", "r")
    text = transcript_file.read()
    # transcript_file.write(transcript)
    transcript_file.close()

    return text
    # return transcript

def smile_split(audio, window_size=3, hop_size=1.5):
    """
    window_size (float): Size of the window in seconds. Default is 3 seconds.
    hop_size (float): Hop size between consecutive windows in seconds. Default is 1.5 seconds.
    """

    samples = audio.to_soundarray()
    framesize = int(window_size * audio.fps)
    hopsize = int(hop_size * audio.fps)

    smiles = smile.audio.Smiler(
        frameSize=framesize, hopSize=hopsize, rmsThresh=0.01
    )

    smiles.processAudio(samples)
    timeslots = smiles.getResults()

    segments = []
    for slot in timeslots:
        start = slot[0] / audio.fps
        end = slot[1] / audio.fps
        segment = audio.subclip(start, end)
        segments.append(segment)

    return segments

def summarize_video(video_file, summary_length=20, window_size=60, hop_size=30):
    # Load the video file and extract the audio
    video = mp.VideoFileClip(video_file)
    audio = video.audio

    # Convert the summary length in minutes to the corresponding number of frames
    summary_duration = summary_length * 60 * 60
    frame_rate = audio.fps
    summary_frames = int(summary_duration * frame_rate)

    # Segment the audio and generate summaries for each segment
    audio_segments = segment_audio(audio)
    summaries = []
    for segment in audio_segments:
        summary = summarize_segment(segment, summary_frames)
        summaries.append(summary)

    summary_text = ' '.join(summaries)

    # Use LSA to get a sentence embedding for each sentence in the summary
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(summaries)
    svd = TruncatedSVD(n_components=1)
    X_lsa = svd.fit_transform(X)

    # Scale the sentence embeddings to the range [0, 1]
    X_lsa = (X_lsa - np.min(X_lsa)) / (np.max(X_lsa) - np.min(X_lsa))

    # Calculate the duration of each summary sentence as a fraction of the total summary duration
    durations = []
    for sentence in summaries:
        sentence_duration = len(sentence) / len(summary_text)
        durations.append(sentence_duration)

    # Get the original video's frames per second
    fps = video.fps

    # Create a clip for each summary sentence
    clips = []
    for i in range(len(summaries)):
        # Get the start and end time of the sentence in the original video
        start_time = i * hop_size
        end_time = start_time + window_size

        # Create a clip for the sentence
        sentence_clip = video.subclip(start_time, end_time)

        # Set the clip's duration based on the duration of the summary sentence
        sentence_duration = durations[i] * window_size
        sentence_clip = sentence_clip.set_duration(sentence_duration)

        # Add the clip to the list of clips
        clips.append(sentence_clip)

    # Concatenate the clips and write the summary video file
    summary_video = concatenate_videoclips(clips)
    summary_video.write_videofile('summary.mp4')



os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creds/celtic-guru-247118-9e8cccc27c4a.json"
input_path = 'By-Faith.mp4'
audio_path = 'temp_audio.wav'
bucket_name = 'sermon-speech-audio'

video = mp.VideoFileClip(input_path)
audio = video.audio.write_audiofile(audio_path)

with io.open(audio_path, 'rb') as audio_file:
    content = audio_file.read()

#upload_to_bucket(bucket_name, audio_path)
summarize_video(input_path)