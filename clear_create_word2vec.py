from clear_text_to_array import *
from gensim.models import Word2Vec
import glob, json

# корпусыг ачаалах
'''
filename = 'corpuses/ikon-news01.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
'''

all_corpuses = ""

print("reading all corpuses, please wait for a little while...")
for filename in glob.iglob('corpuses/**/*.txt', recursive=True):
    with open(filename, 'r') as f:
        json_content = json.load(f)
        all_corpuses = all_corpuses + " " +json_content['title'] + " \n "+json_content["body"]+" \n "
print("reading is done.")

print("converting to the sentence array...")
sentences = clear_text_to_array(all_corpuses)
print('done.')

print("starting to create word2vec...")
model = Word2Vec(sentences, min_count=1)
model.save('model.bin')
print('word2vec model is saved as gensim file format.')
#words = list(model.wv.vocab)
#print(words)
#print(model['дээд'])
#import pdb; pdb.set_trace()