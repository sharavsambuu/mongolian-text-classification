from gensim.models import Word2Vec
import numpy as np

model = Word2Vec.load('model.bin')

vector_dim = 100

embedding_matrix = np.zeros((len(model.wv.vocab), vector_dim))
for i in range(len(model.wv.vocab)):
    embedding_vector = model.wv[model.wv.index2word[i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print(embedding_matrix)

np.save('ids_matrix', embedding_matrix)
import pdb; pdb.set_trace()

