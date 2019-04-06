from clear_text_to_array import *
from gensim.models import Word2Vec
import glob, json, re

# корпусыг ачаалах
all_corpuses     = ""

max_word_count   = 0
max_word_content = ""
max_word_url     = ""
file_count       = 0
all_words        = 0

print("reading all corpuses, please wait for a little while...")

for filename in glob.iglob('corpuses/**/*.txt', recursive=True):
    with open(filename, 'r', encoding="utf8") as f:
        json_content = json.load(f)
        all_corpuses = all_corpuses + " " +json_content['title'] + ". \n "+json_content["body"]+". \n "

        file_count   = file_count + 1
        body_content = json_content['body']
        count        = len(re.findall(r'\w+', body_content))
        all_words    = all_words + count
        if count > max_word_count:
            max_word_count   = count
            max_word_content = body_content
            max_word_url     = json_content['url']

average_words_per_news = all_words/file_count

print("Reading is done. Here is some stats"              )
print("------------------------------------"             )
print("Total file count       : ", file_count            )
print("Average words per news : ", average_words_per_news)
print("Total word count       : ", all_words             )
print("Maximum word count     : ", max_word_count        )
print("Maximum word count url : ", max_word_url          )
print("------------------------------------"             )



print("converting to the sentence array...")
all_corpuses = all_corpuses + ".АННОУНҮГ."
sentences = clear_text_to_array(all_corpuses)
print('done.')

print("starting to create word2vec...")
model = Word2Vec(sentences, min_count=1)
model.save('model.bin')
print('word2vec model is saved as gensim file format.')

total_unique_word_count = len(model.wv.vocab)
print("------------------------------------"         )
print("Unique word count : ", total_unique_word_count)
print("------------------------------------"         )

#print(words)
#print(model['дээд'])
#import pdb; pdb.set_trace()