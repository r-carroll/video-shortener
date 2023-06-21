from pydub import AudioSegment, silence
from moviepy.video.compositing.concatenate import concatenate_videoclips
import openai

def remove_empty_space(temp_audio_path):
    audio = AudioSegment.from_wav(temp_audio_path)
    chunks = []
    silence_thresh = -50
    min_silence_duration = 1200
    audio_chunks = silence.detect_nonsilent(  
        audio,
        min_silence_len=min_silence_duration,
        silence_thresh=silence_thresh
    )

    for start, end in audio_chunks:
        chunk_start_time = start / 1000  # Convert to seconds
        chunk_end_time = end / 1000  # Convert to seconds
        chunk_duration = chunk_end_time - chunk_start_time
        if chunk_duration > min_silence_duration / 1000:
            chunks.append((chunk_start_time, chunk_end_time))

    return chunks

def add_silence_between_segments(chunks, audio_path):
    output_audio = AudioSegment.empty()
    silence_duration = 500

    for i, (start, end) in enumerate(chunks):
            output_audio += AudioSegment.from_wav(audio_path)[int(start * 1000):int(end * 1000)]
            # output_audio.append(segment, 500)
            if i < len(chunks) - 1:
                output_audio += AudioSegment.silent(duration=silence_duration)

    return output_audio


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
    openai.api_key = "sk-..."
    f = open(audio_path, "rb")
    transcript = openai.Audio.transcribe("whisper-1", f, params={"prompt"})
