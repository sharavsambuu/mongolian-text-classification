from clear_text_to_array import *
from gensim.models import Word2Vec

# корпусыг ачаалах
filename = 'corpuses/ikon-news01.txt'
file = open(filename, 'rt')
text = file.read()
file.close()

sentences = clear_text_to_array(text)

model = Word2Vec(sentences, min_count=1)
model.save('model.bin')
print('word2vec model is saved as gensim file format.')
#words = list(model.wv.vocab)
#print(words)
#print(model['дээд'])

#import pdb; pdb.set_trace()
