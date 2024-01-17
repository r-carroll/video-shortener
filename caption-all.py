import os
import sys
from moviepy.video.io.VideoFileClip import VideoFileClip
import AudioProcessing
import TextProcessing

audio_path = 'temp_audio.wav'
folder_path = sys.argv[1]
captioned_path = f"{folder_path}/captioned"

if not os.path.exists(captioned_path):
        os.makedirs(captioned_path)

for filename in os.listdir(folder_path):
       if filename.endswith(('.mp4', '.mov')):
           try:
            print(f"Processing {filename}")
            video_path = os.path.join(folder_path, filename)
            video = VideoFileClip(video_path)
            audio = video.audio.write_audiofile(audio_path)
            sentences = AudioProcessing.get_whisper_transcription(audio_path, captioned_path, filename)
            if len(sys.argv) == 2:
                TextProcessing.generate_subtitles(sentences, captioned_path, video, False, filename)
           except Exception as e:
               print(f"Error processing {filename}: {e}")  