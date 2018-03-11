Mongolian text classifier in tensorflow.

# STEPS

- Run spider in order to collect corpuses and labels from ikon.mn 
    > scrapy runspider ikon_mn_scrape.py

- Download corpus from here, let's respect ikon.mn, scraping can be troubling. 
    > https://drive.google.com/file/d/14NvUkqZRapivmiWc2WOOwIEu9UWyYnnJ/view?usp=sharing

- Create word2vec from all files inside 'corpuses' directory
    > python3 clear_create_word2vec.py 

- Convert word2vec file to ids matrix as a numpy file format in order to use with tensorflow
    > python3 numpy_embedding_matrix_tf.py

- Use embedding matrix with tensorflow in eager mode
    > python3 convert_text_to_seqvector_through_embedmatrix.py

- Prepare training and testing dataset
    > python3 prepare_trainingset.py

- Train lstm recurrent neural network for news classification
    > python3 training_bilstm_rnn.py
    
    > python3 training_lstm_rnn.py

- Freeze trained checkpoints to servable tf model, iteration number is depends on your trained result, see models/bilstm folder after training
    > python3 freeze_tf_model.py --name lstm --iteration 1000

    > python3 freeze_tf_model.py --name bilstm --iteration 3000

- Start classifier RPC server
    > python3 use_freezed_model_rpc.py 

- Start Django to see web interface
    > cd djangoapp

    > python manage.py runserver


# DONE
- Write some scrapers for ikon.mn
- Prepare training texts with its labels, label should be a type of news. For example: politics, economy, society, health, world, technology etc
- Train lstm classifier, also other ones like bibirectional lstm

# IN PROGRESS
- Try to classify text from other sites, for example: news.gogo.mn, write some web backend interface, maybe I can use django 2.0

# TODOs


# RESOURCEs

- checkpointing, save, restore and freeze tensorflow models
    > http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
    > https://nathanbrixius.wordpress.com/2016/05/24/checkpointing-and-reusing-tensorflow-models/
    > http://cv-tricks.com/how-to/freeze-tensorflow-models/

- develop word embeddings python gensim
    > https://machinelearningmastery.com/develop-word-embeddings-python-gensim/

- how to clean text for machine learning with python
    > https://machinelearningmastery.com/clean-text-machine-learning-python/

- using gensim word2vec embeddings in tensorflow
    > http://adventuresinmachinelearning.com/gensim-word2vec-tutorial/

- perform sentimental analysis with lstms using tensorflow
    > https://www.oreilly.com/learning/perform-sentiment-analysis-with-lstms-using-tensorflow

- What does tf.nn.embedding_lookup function do?
    > https://stackoverflow.com/questions/34870614/what-does-tf-nn-embedding-lookup-function-do

- How to One Hot encode categorical sequence data in python
    > https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
    > https://www.tensorflow.org/api_docs/python/tf/one_hot

- How to crawl the web politely with scrapy
    > https://blog.scrapinghub.com/2016/08/25/how-to-crawl-the-web-politely-with-scrapy/
