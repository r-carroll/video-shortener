import os
import io
import re
import moviepy.editor as mp
from google.cloud import speech
from google.cloud import language_v1
from google.cloud import speech_v2
from google.cloud import storage
# from smile import EmotionDetector
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Set up Google Cloud API credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creds/celtic-guru-247118-9e8cccc27c4a.json"
nltk.download('stopwords')

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

def getSpeechText(client):
    speech_audio = speech.RecognitionAudio(uri='gs://sermon-speech-audio/temp_audio.wav')
    config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=44100,
    language_code='en-US',
    audio_channel_count = 2)

    operation = client.long_running_recognize(config=config, audio=speech_audio)

    print("Waiting for operation to complete...")
    response = operation.result(timeout=5400)

# response = client.recognize(config=config, audio=uploaded_uri)

# Perform speech-to-text transcription
# request = speech.RecognizeRequest(request={"config": config, "audio": content})
# response = client.list_voices(request=request)
    transcript = ''
    for result in response.results:
        transcript += result.alternatives[0].transcript
    return transcript

# Initialize Google Cloud APIs
client = speech.SpeechClient()
language_client = language_v1.LanguageServiceClient()

# Load stop words
stop_words = set(stopwords.words('english'))

# Set up video input and output paths
input_path = 'By-Faith.mp4'
output_path = 'condensed.mp4'
audio_path = 'temp_audio.wav'
bucket_name = 'sermon-speech-audio'

# Initialize emotion detector
# detector = EmotionDetector()

# Extract audio from input video
video = mp.VideoFileClip(input_path)
audio = video.audio.write_audiofile(audio_path)

# Load audio file
with io.open(audio_path, 'rb') as audio_file:
    content = audio_file.read()

uploaded_uri = upload_to_bucket(bucket_name, audio_path)

transcript = getSpeechText(client)

transcript_file = open("transcript.txt", "w")
transcript_file.write(transcript)
transcript_file.close()

# Analyze emotional tone of speaker
# emotions = detector.detect_emotions(uploaded_uri)

# Process transcript text
document = language_v1.Document(content=transcript, type_=language_v1.Document.Type.PLAIN_TEXT)
annotations = language_client.analyze_syntax(request={'document': document})
sentences = [sentence.text.content for sentence in annotations.sentences]
sentences = [re.sub(r'[^\w\s]','',sentence) for sentence in sentences]
sentences = [sentence for sentence in sentences if not any(word.lower() in sentence.lower() for word in stop_words)]
sentences = [sentence for sentence in sentences if len(sentence.split()) > 3]

# Extract important segments using Latent Semantic Analysis
vectorizer = TfidfVectorizer(stop_words='english', use_idf=True)
X = vectorizer.fit_transform(sentences)
svd = TruncatedSVD(n_components=1, algorithm='randomized', n_iter=100, random_state=42)
X = svd.fit_transform(X)
idx = X.argmax()
start_time = round(video.duration / len(sentences) * idx, 2)
end_time = start_time + 20

# Create summary video
summary_video = mp.VideoFileClip(input_path).subclip(start_time, end_time)
summary_video = summary_video.set_audio(audio)

# Write summary video to file
summary_video.write_videofile(output_path, fps=24)
