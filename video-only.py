import os
from google.cloud import videointelligence_v1p3beta1 as videointelligence

# Set up the credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'path/to/your/credentials.json'

# Define the criteria for interesting segments
interesting_labels = ['person', 'car', 'music', 'sport']

# Initialize the Video Intelligence API client
client = videointelligence.VideoIntelligenceServiceClient()

# Set the input and output file paths
input_uri = 'gs://input-bucket/input-video.mp4'
output_uri = 'gs://output-bucket/output-video.mp4'

# Set the features to detect
features = [videointelligence.Feature.LABEL_DETECTION, videointelligence.Feature.SHOT_CHANGE_DETECTION]

# Set the video context
video_context = videointelligence.VideoContext()
video_context.label_detection_config.label_detection_mode = videointelligence.LabelDetectionMode.SHOT_AND_FRAME_MODE
video_context.label_detection_config.stationary_camera = True

# Set the output video length
start_time = 0
end_time = 1200 # 20 minutes

# Set the configuration
config = videointelligence.VideoConfig(
    feature=features,
    video_context=video_context,
    output_uri=output_uri,
    start_time_offset={'seconds': start_time},
    end_time_offset={'seconds': end_time}
)

# Perform the video intelligence annotation
operation = client.annotate_video(input_uri=input_uri, video_config=config)
result = operation.result(timeout=180)

# Filter out the segments with interesting labels
segments = []
for segment in result.annotation_results[0].shot_label_annotations:
    if any(label.entity.description.lower() in interesting_labels for label in segment.labels):
        segments.append(segment)

# Create a new video based on the filtered segments
command = f'ffmpeg -i {input_uri} '
for segment in segments:
    start = segment.segment.start_time_offset.seconds
    duration = segment.segment.end_time_offset.seconds - start
    command += f'-ss {start} -t {duration} '

command += f'{output_uri}'
os.system(command)

# Print the result
print('Video annotation complete.')