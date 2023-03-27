import os
import pandas as pd
import moviepy.editor as mp

# Set input and output file names
video_file = 'input_video.mp4'
audio_file = 'audio.wav'
csv_file = 'opensmile.csv'
output_file = 'summary_video.mp4'

# Set parameters
time_allowed = 20  # allowed time in minutes
interesting_labels = ['arousal', 'valence', 'loudness', 'spectral_flux', 'mfcc_1']

# Extract audio from video file
clip = mp.VideoFileClip(video_file)
clip.audio.write_audiofile(audio_file)

# Extract features from audio using OpenSMILE
os.system(f'./SMILExtract -C opensmile.conf -I {audio_file} -O {csv_file}')

# Load feature data into pandas DataFrame
df = pd.read_csv(csv_file, sep=';')

# Identify interesting segments based on labels
segments = []
last_segment_end = 0
for index, row in df.iterrows():
    timestamp = row['frameTime']
    if timestamp >= last_segment_end:
        segment_start = timestamp
        interesting_score = 0
        for label in interesting_labels:
            interesting_score += abs(row[label])
        if interesting_score >= 2:
            segment_end = min(timestamp + 5.0, df.iloc[-1]['frameTime'])  # end 5 seconds after interesting moment
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

# Sort segments by interestingness
# need to figure out a better way to sort to keep things in order
# segment_scores = []
# for start, end in merged_segments:
#     score = 0
#     for i, row in df.loc[(df['frameTime'] >= start) & (df['frameTime'] <= end)].iterrows():
#         for label in interesting_labels:
#             score += abs(row[label])
#     segment_scores.append((start, end, score))
# sorted_segments = sorted(segment_scores, key=lambda x: x[2], reverse=True)

# Create summary video
summary_duration = 0
summary = None
for start, end, _ in merged_segments: # will replace this with sorted segments
    clip = mp.VideoFileClip(video_file).subclip(start, end)
    if summary_duration + clip.duration <= time_allowed*60:
        if summary is None:
            summary = clip
        else:
            summary = mp.concatenate_videoclips([summary, clip])
        summary_duration += clip.duration
    else:
        break

# Save summary video
summary.write_videofile(output_file, fps=clip.fps, codec='libx264', audio_codec='aac')

# Clean up intermediate files
os.remove(audio_file)
os.remove(csv_file)
