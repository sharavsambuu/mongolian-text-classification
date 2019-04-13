# mongolian-text-classification
Mongolian cyrillic text classification with modern tensorflow and some fine tuning on TugsTugi's BERT model.

# Fine tuning TugsTugi's Mongolian BERT model
On TPU mode, loading checkpoints from the file system doesn't supported by the bert and bucket should be used.

Fine tuning mongolian BERT on TPU, You need own bucket in order to finetune on TPU [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CnGd2OnNDlxe6ZUjmOa7zg__CcKk5X85)

Fine tune a mongolian BERT on GPU, a lot of computation needed, a low batch size matters due to memory limit [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1u9mVeWRh7GWLONAzZ3XpJciPfv38vHaZ)

# Classifiers using simple neural networks

No 01, Simplest classifier [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ulv6tUAjOsp-jN4sTdef3lTuJb0yX4qy)
No 02, Pretrained Word2Vec initialization from Facebook's fasttext, kind of transfer learningish. Embedding layer is not trainable in this case. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SfwdhIoRMi4kXeAN8eUjYXKuT5zig9WV)

# Upcoming series will include followings
word2vec initialization, 1D Convolution, RNN variants, Attention(with visualization), Transformer, Techniques to handle longer texts and so on...

# useful references and resources
  - Mongolian BERT models
    https://github.com/tugstugi/mongolian-bert
  - Mongolian NLP
    https://github.com/tugstugi/mongolian-nlp
  - Eduge classification baseline using SVM
  	https://colab.research.google.com/github/tugstugi/mongolian-nlp/blob/master/misc/Eduge_SVM.ipynb
  - News crawler
    https://github.com/codelucas/newspaper
  
