from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import math
import string
from numpy import array

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

in_filename = 'gut_test_clean.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')

model = load_model('model_4file_2.h5')
tokenizer = load(open('tokenizer_4file_2.pkl','rb'))
sequences = tokenizer.texts_to_sequences(lines)
vocab_size = len(tokenizer.word_index) + 1
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]
loss = model.evaluate(X,y,batch_size=128,verbose=1)
print(math.exp(loss))

