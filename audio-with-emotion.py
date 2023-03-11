import os
import subprocess
import pandas as pd
import librosa
import pydub
import moviepy.editor as mp

# Set up labeling criteria
scripture_refs = ['john 3:16', 'psalm 23', 'romans 8']
personal_anecdotes = ['when I was a child', 'last week at the grocery store']
theological_concepts = ['salvation', 'grace', 'baptism']
emotion_threshold = 0.5  # threshold for detecting emotional moments (0 to 1)

# Set up file paths
video_file = 'sermon.mp4'
audio_file = 'sermon_audio.mp3'
csv_file = 'sermon.csv'
output_file = 'sermon_summary.mp4'

# Extract audio from video
clip = mp.VideoFileClip(video_file)
clip.audio.write_audiofile(audio_file)

# Extract features using OpenSMILE
smile_path = '/path/to/opensmile'
config_file = '/path/to/config/emobase.conf'
command = f'{smile_path}/bin/SMILExtract -C {config_file} -I {audio_file} -O {csv_file}'
subprocess.call(command, shell=True)

# Load features into a Pandas dataframe
df = pd.read_csv(csv_file, delimiter=';')

# Analyze the dataframe for interesting segments
segments = []
last_segment_end = 0
for i, row in df.iterrows():
    timestamp = row['frameTime']
    f0_mean = row['F0semitoneFrom27.5Hz_sma3nz_amean']
    energy_mean = row['loudness_sma3_amean']
    transcript = row['transcript']
    
    # Scripture references
    if any(ref in transcript.lower() for ref in scripture_refs):
        segment_start = max(timestamp - 5.0, 0)  # start 5 seconds before reference
        segment_end = min(timestamp + 15.0, df.iloc[-1]['frameTime'])  # end 15 seconds after reference
        segments.append((segment_start, segment_end))
    
    # Personal anecdotes
    if any(anecdote in transcript.lower() for anecdote in personal_anecdotes):
        segment_start = max(timestamp - 10.0, 0)  # start 10 seconds before anecdote
        segment_end = min(timestamp + 10.0, df.iloc[-1]['frameTime'])  # end 10 seconds after anecdote
        segments.append((segment_start, segment_end))
    
    # Theological concepts
    if any(concept in transcript.lower() for concept in theological_concepts):
        segment_start = max(timestamp - 3.0, 0)  # start 3 seconds before concept
        segment_end = min(timestamp + 12.0, df.iloc[-1]['frameTime'])  # end 12 seconds after concept
        segments.append((segment_start, segment_end))
    
    # Emotional moments
    if row['class'] == 'arousal' and row['class_emo'] == 'neutral' and row['class2_emo'] == 'neutral':
        continue  # skip neutral segments
    if row['class'] == 'arousal' and row['class_emo'] != 'neutral' and row['class2_emo'] != 'neutral':
        if row['emo_prob_1'] >= emotion_threshold:
            segment_start = max(last_segment_end, timestamp - 5.0)  # start 5
            segment_end = min(timestamp + 5.0, df.iloc[-1]['frameTime'])  # end 5 seconds after emotional moment
            segments.append((segment_start, segment_end))
            last_segment_end = segment_end

# Merge overlapping segments
merged_segments = []
for start, end in segments:
    if not merged_segments:
        merged_segments.append((start, end))
    else:
        last_start, last_end = merged_segments[-1]
        if start <= last_end:
            merged_segments[-1] = (last_start, max(end, last_end))
        else:
            merged_segments.append((start, end))

# Create summary video
summary = None
for start, end in merged_segments:
    clip = mp.VideoFileClip(video_file).subclip(start, end)
    if summary is None:
        summary = clip
    else:
        summary = mp.concatenate_videoclips([summary, clip])

# Save summary video
summary.write_videofile(output_file, fps=clip.fps, codec='libx264', audio_codec='aac')

# Clean up intermediate files
os.remove(audio_file)
os.remove(csv_file)
