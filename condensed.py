import os
import pandas as pd
import moviepy.editor as mp
import speech_recognition
import opensmile
import numpy as np
import subprocess
from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creds/celtic-guru-247118-9e8cccc27c4a.json"

def upload_to_bucket(bucket_name, file_name):
    """Uploads a file to the given Cloud Storage bucket."""
    # Instantiate a client object
    storage_client = storage.Client()

    # Get the bucket object
    bucket = storage_client.bucket(bucket_name)

    # Create a blob object for the file to be uploaded
    blob = bucket.blob(file_name)

    if blob.exists():
        blob.delete()

    # Upload the file to Cloud Storage
    blob.upload_from_filename(file_name)

    print(f"File {file_name} uploaded to {file_name}.")

    return blob.public_url

#Extract audio from video
csv_file = 'opensmile.csv'
video_path = "By-Faith.mp4"
audio_path = "By-Faith.wav"
bucket_name = "sermon-speech-audio"
opensmile_path = "/Users/ryan.carroll/Documents/GitHub/opensmile/build/progsrc/smilextract/SMILExtract"
config_file = "/Users/ryan.carroll/Documents/GitHub/opensmile/config/gemaps/v01b/GeMAPSv01b.conf"
video = mp.VideoFileClip(video_path)
audio = video.audio
audio.write_audiofile(audio_path)

#Process the audio
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.GeMAPSv01b,
    feature_level=opensmile.FeatureLevel.Functionals,
)
smile_result = smile.process_file(audio_path)
print(smile_result)
features = np.asarray([smile_result.to_numpy()])

# Upload audio file to bucket
uploaded_uri = upload_to_bucket(bucket_name, audio_path)

# subprocess.run([
#     opensmile_path,
#     "-C", config_file,
#     "-I", audio_path,
#     "-O", csv_file
# ])
#os.system(f'./Users/ryan.carroll/Documents/GitHub/opensmile/build/progsrc/smilextract/SMILExtract -C /Users/ryan.carroll/Documents/GitHub/opensmile/config/gemaps/v01b/GeMAPSv01b.conf -I {audio_path} -O {csv_file}')
# features = pd.read_csv(csv_file, sep=';')

# Step 3: Use Google's intelligence API
recognizer = speech_recognition.Recognizer()
with speech_recognition.AudioFile(uploaded_uri) as source:
    audio = recognizer.record(source)
text = recognizer.recognize_google(audio)

# Step 4: Identify important parts
# Use acoustic features to detect emotional parts
emotional_indices = np.where(features[:, smile.feature_names.index('F0semitoneFrom27.5Hz_sma3nz_amean')] > threshold)[0]
# Search for theological concepts in text
theological_keywords = ["God", "Jesus", "Holy Spirit", "salvation", "faith"]
theological_indices = [i for i, word in enumerate(text.split()) if word in theological_keywords]
# Use both sets of indices to identify important parts
important_indices = sorted(list(set(emotional_indices) | set(theological_indices)))

# Step 5: Create summary video
summary_path = "summary.mp4"
clips = [video.subclip(start_time, end_time) for start_time, end_time in zip(timestamps[important_indices], timestamps[important_indices][1:])]
summary = mp.concatenate_videoclips(clips)
summary.write_videofile(summary_path)

