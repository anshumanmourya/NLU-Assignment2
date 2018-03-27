from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import math
import string
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text
in_filename = 'gut_4files_test.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
model = load_model('model_char.hdf5')
mapping = load(open('mapping_char.pkl', 'rb'))
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
loss = model.evaluate(X,y,batch_size=128, verbose=1)
print(math.exp(loss))

