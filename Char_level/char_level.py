
from numpy import array
from pickle import dump
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

raw_text = load_doc('gut_4files_train.txt')
tokens = raw_text.split()
raw_text = ' '.join(tokens)


length = 10
sequences = list()
for i in range(length, len(raw_text),3):
    seq = raw_text[i-length:i+1]
    sequences.append(seq)
out_filename = 'char_sequences.txt'
save_doc(sequences, out_filename)

raw_text = sequences
lines = raw_text.split('\n')
chars = sorted(list(set(raw_text)))
mapping = dict((c, i) for i, c in enumerate(chars))
sequences = list()
for line in lines:
    encoded_seq = [mapping[char] for char in line]
    sequences.append(encoded_seq)

vocab_size = len(mapping)

sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = array(sequences)
y = to_categorical(y, num_classes=vocab_size)
model = Sequential()
model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=50, verbose=2)

model.save('model_char.hdf5')
dump(mapping, open('mapping_char.pkl', 'wb'))

