from gensim.models import Word2Vec
import numpy as np

model = Word2Vec.load('model.bin')

vector_dim = 100

#word_to_id = {}
embedding_matrix = np.zeros((len(model.wv.vocab), vector_dim))
for i in range(len(model.wv.vocab)):
    embedding_vector = model.wv[model.wv.index2word[i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        #word_to_id[i] = model.wv.index2word

np.save('ids_matrix', embedding_matrix)
#import pdb; pdb.set_trace()
print('embedded ids matrix is saved as a numpy file format.')

