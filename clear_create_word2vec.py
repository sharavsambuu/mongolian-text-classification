from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
import string
from mongolianstopwords import *

# корпусыг ачаалах
filename = 'corpuses/ikon-news01.txt'
file = open(filename, 'rt')
text = file.read()
file.close()

# эх текстийг өгүүлбэрүүдэд хуваах
text_sentences = sent_tokenize(text)
sentences = []
for text_sentence in text_sentences:
    # өгүүлбэрийн текстийг үгүүд болгож хувиргах
    tokens = word_tokenize(text_sentence)
    # том үсгүүдийг болиулах
    tokens = [w.lower() for w in tokens]
    # үг бүрээс тэмдэгтүүдийг хасах
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # текст бус үгүүдийг хасах
    words = [word for word in stripped if word.isalpha()]
    # stopword уудыг хасах
    stop_words = set(stopwordsmn)
    words = [w for w in words if not w in stop_words]
    sentences.append(words)

from gensim.models import Word2Vec
model = Word2Vec(sentences, min_count=1)
model.save('model.bin')
print(model)
words = list(model.wv.vocab)
print(words)
print(model['дээд'])

import pdb; pdb.set_trace()
