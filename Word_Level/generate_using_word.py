from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
def load_doc(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
	result = list()
	in_text = seed_text
	for _ in range(n_words):
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		yhat = model.predict_classes(encoded, verbose=0)
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		in_text += ' ' + out_word
		result.append(out_word)
	return ' '.join(result)

in_filename = 'sequences_4file_2.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
seq_length = len(lines[0].split()) - 1

model = load_model('model_4file_2.h5')
tokenizer = load(open('tokenizer_4file_2.pkl', 'rb'))
seed_text = lines[randint(0,len(lines))]
#print(seed_text + '\n')
generated = generate_seq(model, tokenizer, seq_length, seed_text, 10)
print(generated)

