from os import listdir
from os.path import isfile, join
import shutil
import glob, os, os.path
import random
import math
import json

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from gensim.models import Word2Vec

from wordtoken_to_id import *
from clear_text_to_array import *

global_corpuses = []

def convert_to_onehot(name):
    switcher = {
        "economy"    : lambda: [1,0,0,0,0,0],
        "health"     : lambda: [0,1,0,0,0,0],
        "politics"   : lambda: [0,0,1,0,0,0],
        "society"    : lambda: [0,0,0,1,0,0],
        "technology" : lambda: [0,0,0,0,1,0],
        "world"      : lambda: [0,0,0,0,0,1]
    }
    return switcher.get(name, lambda: [0,0,0,0,0,0])()

def fix_news_body(filename):
    found = False
    jsoncontent = ""
    with open(filename, encoding="utf8") as f:
        jsoncontent = json.load(f)
        body        = jsoncontent['body'].strip()
        if not body:
            print("YES EMPTY BODY FOUND...")
            found = True
            jsoncontent['body'] = jsoncontent['title']
    if found:
        with open(filename, "w", encoding="utf8") as outfile:
            print("FIXING...", filename)
            json.dump(jsoncontent, outfile, ensure_ascii=False)

for filename in glob.iglob('corpuses/**/*.txt', recursive=True):
    fix_news_body(filename) # some news body is empty, fix it by replacing its title
    current_file_path      = os.path.abspath(filename)
    current_directory      = os.path.abspath(os.path.join(current_file_path, os.pardir))
    current_directory_name = os.path.split(current_directory)
    category               = current_directory_name[1]
    one_hot                = convert_to_onehot(category)
    only_file_name         = os.path.basename(filename)
    global_corpuses.append((only_file_name, one_hot, category))

random.shuffle(global_corpuses)
random.shuffle(global_corpuses)
random.shuffle(global_corpuses)

split_location = math.floor(80*len(global_corpuses)/100) # 80% for training, 20% for testing
training_set   = global_corpuses[:split_location]
test_set       = global_corpuses[split_location:]
dataset_info   = {
    'training' : training_set,
    'testing'  : test_set
}

temp_corpus_dir = 'temp_corpuses'
if os.path.exists(temp_corpus_dir):
    shutil.rmtree(temp_corpus_dir)
os.makedirs(temp_corpus_dir)

with open("temp_corpuses/dataset.json", "w", encoding="utf8") as outfile:
    json.dump(dataset_info, outfile, ensure_ascii=False)

#import pdb; pdb.set_trace()