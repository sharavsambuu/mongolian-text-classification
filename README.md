# resources

## develop word embeddings python gensim
https://machinelearningmastery.com/develop-word-embeddings-python-gensim/

## how to clean text for machine learning with python
https://machinelearningmastery.com/clean-text-machine-learning-python/

## using gensim word2vec embeddings in tensorflow
http://adventuresinmachinelearning.com/gensim-word2vec-tutorial/

## perform sentimental analysis with lstms using tensorflow
https://www.oreilly.com/learning/perform-sentiment-analysis-with-lstms-using-tensorflow

## What does tf.nn.embedding_lookup function do?
https://stackoverflow.com/questions/34870614/what-does-tf-nn-embedding-lookup-function-do

## How to One Hot encode categorical sequence data in python
https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
https://www.tensorflow.org/api_docs/python/tf/one_hot

## How to crawl the web politely with scrapy
https://blog.scrapinghub.com/2016/08/25/how-to-crawl-the-web-politely-with-scrapy/

# steps

- Run spider in order to collect corpuses and labels from ikon.mn
    > scrapy runspider ikon_mn_scrape.py

- Create word2vec from all files inside 'corpuses' directory
    > python3 clear_create_word2vec.py 

- Convert word2vec file to ids matrix as a numpy file format in order to use with tensorflow
    > python3 numpy_embedding_matrix_tf.py

- Use embedding matrix with tensorflow in eager mode
    > python3 convert_text_to_seqvector_through_embedmatrix.py

- Prepare training and testing dataset
    > python3 prepare_trainingset.py



# TODOs

1. Write some scrapers for ikon.mn
2. Prepare training texts with its labels, label should be a type of news. For example: politics, economy, society, health, world, technology etc
3. Train them
4. Try to classify text from other sites, for example: news.gogo.mn