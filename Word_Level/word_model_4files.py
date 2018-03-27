from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from nltk.corpus import gutenberg
import string
from keras.callbacks import ModelCheckpoint
punc = list(string.punctuation)
import string

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text
def clean_doc(doc):
    doc = doc.replace('--', ' ')
    tokens = doc.split()
    tokens = [word for word in tokens if word not in punc]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    return tokens

def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

in_filename = 'gut_4files_Train.txt'
doc = load_doc(in_filename)
tokens = clean_doc(doc)
#print('Total Tokens: %d' % len(tokens))
#print('Unique Tokens: %d' % len(set(tokens)))

small = tokens
small.append('<UNK>')
length = 50 + 1
sequences = list()
for i in range(length, len(small),3):
    seq = small[i-length:i]
    line = ' '.join(seq)
    sequences.append(line)

print('Total Sequences: %d' % len(sequences))

out_filename = 'sequences_4file_2.txt'
save_doc(sequences, out_filename)

in_filename = 'sequences_4file_2.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
vocab_size = len(tokenizer.word_index) + 1

sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]

\model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, batch_size=128, epochs=100)

model.save('model_4file_2.h5')
dump(tokenizer, open('tokenizer_4file_2.pkl', 'wb'))

