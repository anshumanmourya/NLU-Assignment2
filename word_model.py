
# coding: utf-8

# In[2]:





# In[1]:


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


# In[3]:


import string

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# turn a doc into clean tokens
def clean_doc(doc):
    # replace '--' with a space ' '
    doc = doc.replace('--', ' ')
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    #table = str.maketrans('', '', string.punctuation)
    #tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word not in punc]
    # remove remaining tokens that are not alphabetic

    tokens = [word for word in tokens if word.isalpha()]
    # make lower case
    tokens = [word.lower() for word in tokens]
    return tokens

# save tokens to file, one dialog per line
def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

# load document
in_filename = 'gut_Train'
#file = gutenberg.fileids()
#doc = gutenberg.raw(file)
doc = load_doc(in_filename)
#print(doc[:200])

# clean document
tokens = clean_doc(doc)
#tokens = [item for sublist in lines for item in sublist]
#print len(tokens)
#print(tokens[:200])
print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))


# In[11]:



small = tokens#[:80000]
small.append('<UNK>')
#print small


# In[12]:



def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

# organize into sequences of tokens
length = 50 + 1
sequences = list()
for i in range(length, len(small),3):
    # select sequence of tokens
    seq = small[i-length:i]
    # convert into a line
    line = ' '.join(seq)
    # store
    sequences.append(line)

print('Total Sequences: %d' % len(sequences))

# save sequences to file
out_filename = 'sequences.txt'
save_doc(sequences, out_filename)

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# load
in_filename = 'sequences.txt'
doc = load_doc(in_filename)
#doc = sequences
lines = doc.split('\n')


# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
# vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# separate into input and output
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]

# define the checkpoint
filepath="data/weights/word-weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# define model
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X, y, batch_size=128, epochs=20, callbacks=callbacks_list)


# In[4]:


# In[6]:


# save the model to file
model.save('model.h5')
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))


# In[3]:




