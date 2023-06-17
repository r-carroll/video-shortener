
from pydub import AudioSegment

def compress_audio(input_audio, output_audio_path, format='mp3', bitrate='48k'):
    audio = input_audio.set_channels(2).set_frame_rate(44100)
    audio.export(output_audio_path + '/compressed.' + format, format=format, bitrate=bitrate)
    print("Finished compressing audio")

    return output_audio_path + '-compressed.' + format




