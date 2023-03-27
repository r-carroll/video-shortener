import os
import io
import numpy as np
import moviepy.editor as mp
import smile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from google.cloud import speech
from google.cloud import storage

def upload_to_bucket(bucket_name, file_name):
    """Uploads a file to the given Cloud Storage bucket."""
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

def summarize_video(video_file, summary_length=20):
    window_size = 4
    hop_size = 1

    # Get transcription of speech in video
    transcript = get_transcription()

    # Preprocess transcript text
    sentences = sent_tokenize(transcript)
    stop_words = set(stopwords.words('english'))
    preprocessed_text = []
    for sentence in sentences:
        words = sentence.lower().split()
        words = [w for w in words if not w in stop_words]
        preprocessed_text.append(' '.join(words))

    # Create term-document matrix
    vectorizer = TfidfVectorizer()
    term_doc_matrix = vectorizer.fit_transform(preprocessed_text)

    # Compress matrix using SVD
    svd = TruncatedSVD(n_components=min(10, len(sentences)-1))
    svd.fit(term_doc_matrix)
    compressed_matrix = svd.transform(term_doc_matrix)

    # Calculate scores for each sentence
    scores = np.sum(compressed_matrix, axis=1)

    # Select sentences with highest scores
    summary_indices = np.argsort(scores)[-summary_length:]

    # Extract segments from original video and concatenate to create summary video
    audio = mp.AudioFileClip(video_file)
    summary_segments = smile.smile_split(audio.to_soundarray(), window_size=window_size, hop_size=hop_size)
    summary_video = summary_segments[0]
    for i in range(1, len(summary_segments)):
        summary_video = summary_video.append(summary_segments[i])

    # Write summary video to file
    summary_file = f"{os.path.splitext(video_file)[0]}_summary.mp4"
    summary_video.write_videofile(summary_file)
    return summary_file


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