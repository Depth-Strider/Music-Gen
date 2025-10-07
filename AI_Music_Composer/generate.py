import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.models import load_module
from music21 import stream, note, chord
import pickle
import random

with open('notes.pkl', 'rb') as f:
    notes=pickle.load(f)
pitchnames=sorted(set(notes))
n_vocab=len(pitchnames)
note_to_int={note: number for number, note in enumerate(pitchnames)}
int_to_note={number: note for number, note in enumerate(pitchnames)}
