from pydub import AudioSegment, silence
from moviepy.video.compositing.concatenate import concatenate_videoclips
import openai
import json

def load_api_key():
    with open('creds/celtic-guru-247118-9e8cccc27c4a.json', 'r') as json_file:
        data = json.load(json_file)
        api_key = data.get('open_ai_key')
    return api_key


def remove_empty_space(temp_audio_path, export_path):
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

    modified_audio.export(f"{export_path}cleaned_audio.wav", format="wav")
    print("cleaned audio saved")


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

def clips_to_video(video, chunks):
    clips = []

    for start_time, end_time in chunks:
        clip = video.subclip(start_time, end_time)
        clips.append(clip)

    return concatenate_videoclips(clips)

def get_viral_transcript(audio_path):
    openai.api_key = load_api_key()
    f = open(audio_path, "rb")
    transcript = openai.Audio.transcribe("whisper-1", f, prompt="Please transcribe only the first sentence")
    print(transcript)
