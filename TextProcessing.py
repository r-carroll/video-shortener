import spacy
import json

def concatenate_sentences(transcriptions):
    nlp = spacy.load('en_core_web_sm')
    complete_sentences = []

    for transcription in transcriptions:
        text = transcription['text']
        start_time = transcription['start_time']
        end_time = transcription['end_time']

        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]

        if not sentences:
            continue

        last_sentence = sentences[0]
        for sentence in sentences[1:]:
            if last_sentence[-1] not in ('.', '!', '?') and sentence[0].islower():
                last_sentence += ' ' + sentence
            else:
                complete_sentences.append({
                    'text': last_sentence.strip(),
                    'start_time': start_time,
                    'end_time': end_time
                })
                last_sentence = sentence

        complete_sentences.append({
            'text': last_sentence.strip(),
            'start_time': start_time,
            'end_time': end_time
        })

    return complete_sentences

# Example usage
json_data = '''
[
    {
        "text": "Find if you would Esther chapter 5, the fifth chapter of the Book of Esther. You know, some might look at the story of Esther and see only a series of coincidences. Well, you know, it's just a coincidence that the king in a drunken stupor would call out to the queen vashti and insist that she present herself in front of himself and other drunk men in a way that we had degrade the office of",
        "start_time": 0.1,
        "end_time": 29.7
    },
    {
        "text": "Queen. And therefore she said that she would not come and she was eventually deposed because of that action, which then opened the door for a new Queen. We might look and say, well, you know, it's just a coincidence that Mordecai and Esther remained and Susa it's just a coincidence that Esther would have found favor in the eyes of those to whom she was responsible and that she would eventually find",
        "start_time": 30.0,
        "end_time": 59.9
    },
    {
        "text": "and favor in the eyes of the king and be elevated to the role of Queen.",
        "start_time": 60.0,
        "end_time": 66.0
    }
]
'''

transcriptions = json.loads(json_data)
complete_sentences = concatenate_sentences(transcriptions)

output_json = json.dumps(complete_sentences, indent=4)
print(output_json)
