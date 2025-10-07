import glob
import pickle
from music21 import converter, instrument, note, chord
#glob is a python module for file pattern matching. It matches specific patterns, in this case .mid
#Pickle is used from saving or accessing files. Commonly used for AI models, and files. 
#Music21 is used for analysing, generating music files. Especially .mid files. 

midi_files=glob.glob('/Users/rushil/Documents/Python_Stuff/AI_Music_Composer/data/*.mid')
notes=[]
for file in midi_files:
    midi=converter.parse(file)
    parts=instrument.partitionByInstrument(midi)
    if parts:
        notes_to_parse=parts.parts[0].recurse()
    else:
        notes_to_parse=midi.flat.notes()

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

with open('/Users/rushil/Documents/Python_Stuff/AI_Music_Composer/notes.pkl', 'wb') as f:
    pickle.dump(notes, f)
print(f'Total notes/chords: {len(notes)}')