import os
import pandas as pd
import moviepy.editor as mp
import re

# Set input and output file names
video_file = 'input_video.mp4'
audio_file = 'audio.wav'
csv_file = 'opensmile.csv'
output_file = 'summary_video.mp4'

# Set parameters
time_allowed = 20  # allowed time in minutes
interesting_labels = ['arousal', 'valence', 'loudness', 'spectral_flux', 'mfcc_1']
emotional_labels = ['smile', 'laugh']
reference_keywords = ['john', 'matthew', 'luke', 'mark', 'genesis', 'exodus', 'leviticus', 'numbers', 'deuteronomy',
                      'psalm', 'proverb', 'ecclesiastes', 'isaiah', 'jeremiah', 'lamentations', 'ezekiel', 'daniel',
                      'hosea', 'joel', 'amos', 'obadiah', 'jonah', 'micah', 'nahum', 'habakkuk', 'zephaniah', 'haggai',
                      'zechariah', 'malachi']
anecdote_keywords = ['I', 'me', 'my', 'mine', 'we', 'our', 'us', 'ours']
theology_keywords = ['sin', 'repentance', 'redemption', 'grace', 'salvation', 'justification', 'sanctification',
                     'atonement', 'eternal life', 'faith', 'works', 'heaven', 'hell', 'predestination', 'baptism',
                     'communion', 'trinity', 'father', 'son', 'holy spirit']

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
segment_scores = []
for start, end in merged_segments:
    score = 0
    for i, row in df.loc[(df['frameTime'] >= start) & (df['frameTime'] <= end)].iterrows():
        for label in interesting_labels:
            score += abs(row[label])
    segment_scores.append((start, end, score))
sorted_segments = sorted(segment_scores, key=lambda x: x[2], reverse=True)

# Identify emotional moments
emotional_segments
