# mongolian-text-classification
Mongolian cyrillic text classification with modern tensorflow and some fine tuning on TugsTugi's BERT model.

# Load Mongolian BERT in Tensorflow 2
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ReDLH2DDiCt_Y800vGub8OuYJlR-TsZw)

# Generate text using Mongolian BERT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jJA-YSAsbq5gbpyGYE-8p-rCzgqSU9eX)

# Visualize Mongolian BERT using bertviz and pytorch model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UEDNlfEmXxZy1jRrE7pCTZNu8DplWVQv)

![Alt text](images/bert/mongolian-bert-attend-visualization.png?raw=true "Mongolian BERT attend")


# Fine tuning TugsTugi's Mongolian BERT model
On TPU mode, loading checkpoints from the file system doesn't supported by the bert and bucket should be used.

Fine tuning mongolian BERT on TPU, You need own bucket in order to finetune on TPU [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CnGd2OnNDlxe6ZUjmOa7zg__CcKk5X85)

Fine tune a mongolian BERT on GPU, a lot of computation needed, a low batch size matters due to memory limit [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1u9mVeWRh7GWLONAzZ3XpJciPfv38vHaZ)

# Classifiers using simple neural networks

No 01, Simplest classifier [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ulv6tUAjOsp-jN4sTdef3lTuJb0yX4qy)

No 02, Pretrained Word2Vec initialization from Facebook's fasttext, kind of transfer learningish. Embedding layer is not trainable in this case [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SfwdhIoRMi4kXeAN8eUjYXKuT5zig9WV) and with trainable embedding layer [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WQvCa6KDOxQ2YjDdb48g4zsN60_Svbhg)

No 03, 1D Convolution [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JgJN74E1w1x8RSjm9qi06uw6y0I_9k1J) and multiple 1D convnets [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lTh2dG64L4aJsCip714sCA_xQgMttxOb)

No 04, LSTM [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1j0MN3UTGz-990bl61n5B1mrtjnq8hSdh)

No 05, Visualize attention scores for classification with LSTM and Attention [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10nPgRmbZsjad46CdVJKRHklestXcEpZ5)

No 06, Visualize RNN neuron firing in text generation [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ndM1G-0qZx4wi6E9kPL1D9IjaM0pq3r9)

# Mongolian sentence interpolation experiments

Sequence loss in keras and tf2 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jlyB2fOi_JBAi4WPMVDJ_e8-_WHtQK_9)

Variational Auto Encoder for Mongolian text [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tBTudj9M5CGih3p8Uxj0R1SA6f3BJj-Z)

# Other experiments
Predict next word, greedy text generation [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1urjsJUuNTnTAAAqu_eXpIkwRWUi72xp_)

# Series included(or will) followings
word2vec initialization, 1D Convolution, RNN variants, Attention, Some weights visualization for reasoning, Transformer, Techniques to handle longer texts and so on...


# useful references and resources
  - Mongolian BERT models
    https://github.com/tugstugi/mongolian-bert
  - Mongolian NLP
    https://github.com/tugstugi/mongolian-nlp
  - Eduge classification baseline using SVM
  	https://colab.research.google.com/github/tugstugi/mongolian-nlp/blob/master/misc/Eduge_SVM.ipynb
  - News crawler
    https://github.com/codelucas/newspaper
  
# Images and screenshots

![Alt text](images/cnn-weights/1.png?raw=true "CNN weights 1")
![Alt text](images/cnn-weights/2.png?raw=true "CNN weights 2")
![Alt text](images/cnn-weights/3.png?raw=true "CNN weights 3")
![Alt text](images/cnn-weights/4.png?raw=true "CNN weights 4")
