import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
import pickle

with open('/Users/rushil/Documents/Python_Stuff/AI_Music_Composer/notes.pkl', 'rb') as f:
    notes=pickle.load(f)

pitchnames=sorted(set(notes))
print(pitchnames)
n_vocab=len(pitchnames)
notes_to_int={note:number for number, note in enumerate(pitchnames)}
sequence_len=100
network_input=[]
network_output=[]
print('total notes: ', len(notes))
if len(notes)<=sequence_len:
    raise ValueError(f'Not enough notes ({len(notes)}) for sequence length {sequence_len}')
for i in range(len(notes)-sequence_len):
    seq_in=notes[i:i+sequence_len]
    seq_out=notes[i+sequence_len]
    network_input.append([notes_to_int[n] for n in seq_in])
    network_output.append(notes_to_int[seq_out])

n_patterns=len(network_input)
network_input=np.reshape(network_input, (n_patterns, sequence_len, 1))
network_input=network_input/float(n_vocab)

print('network_output_len:', len(network_output))
network_output=to_categorical(network_output)
print("Sample network output: ", network_output[:10])
model=Sequential()
model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(n_vocab))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.fit(network_input, network_output, epochs=5, batch_size=64)
model.save('models/lstm_music.h5')
