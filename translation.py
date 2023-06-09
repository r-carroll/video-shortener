from googletrans import Translator
import srt


def generate_srt_files(sentences, folder):
  english_srt_file = srt.SrtFile()
  spanish_srt_file = srt.SrtFile()
  portuguese_srt_file = srt.SrtFile()

  for sentence in sentences:
    english_srt_file.add_subtitle(sentence['text'], sentence['start_time'], sentence['end_time'])
    translator = Translator()
    spanish_srt_file.add_subtitle(translator.translate(sentence['text'], dest='es').text, sentence['start_time'], sentence['end_time'])
    portuguese_srt_file.add_subtitle(translator.translate(sentence['text'], dest='pt').text, sentence['start_time'], sentence['end_time'])
    
    english_srt_file.save(f"{folder}/english.srt")
    spanish_srt_file.save(f"{folder}/spanish.srt")
    portuguese_srt_file.save(f"{folder}/portuguese.srt")

  return
