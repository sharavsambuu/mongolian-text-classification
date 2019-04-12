# mongolian-text-classification
Mongolian cyrillic text classification with modern tensorflow and some fine tuning on TugsTugi's BERT model.

# Notebooks and colab links
On TPU mode, loading checkpoints from the file system doesn't supported by the bert and bucket should be used.

Fine tuning mongolian BERT on TPU, You need own bucket in order to finetune on TPU [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CnGd2OnNDlxe6ZUjmOa7zg__CcKk5X85)

Fine tune a mongolian BERT on GPU, a lot of computation needed, a low batch size matters due to memory limit [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1u9mVeWRh7GWLONAzZ3XpJciPfv38vHaZ)


# useful references and resources
  - Mongolian BERT models
    https://github.com/tugstugi/mongolian-bert
  - Mongolian NLP
    https://github.com/tugstugi/mongolian-nlp
  - Eduge classification baseline using SVM
  	https://colab.research.google.com/github/tugstugi/mongolian-nlp/blob/master/misc/Eduge_SVM.ipynb
  - News crawler
    https://github.com/codelucas/newspaper
  
