
# imports
import pandas as pd 
import pickle
import re

# PyTorch dependencies
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# General dependencies
import numpy as np
import pandas as pd
from collections import defaultdict
from textwrap import wrap

# Sklearn dependencies
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

import spacy 
nlp = spacy.load("en_core_web_sm")

import pickle
import os

from transformers import *
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tqdm.notebook import tqdm

import spacy 
nlp = spacy.load("en_core_web_sm")

import os
import pandas as pd
import numpy as np
import re
import pickle
import sys
from tqdm import tqdm_notebook
from gensim.models import Word2Vec

import tensorflow._api.v2.compat.v1 as tf
#tensorflow.compat.v1 as tf
from tensorflow import keras
from tensorflow.keras import layers
#print(tf.__version__)
import tensorflow_hub as hub

# Initialize session
sess = tf.compat.v1.Session()

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.text import one_hot
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Activation, Dropout, Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import GlobalMaxPooling1D
from tensorflow.python.keras.layers import Bidirectional
from tensorflow.python.keras.layers import TimeDistributed
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

from tensorflow.python.keras.layers import Input, CuDNNLSTM, CuDNNGRU, Conv1D
from tensorflow.python.keras.layers import GlobalMaxPool1D, GlobalAveragePooling1D, MaxPooling1D
from tensorflow.python.keras.layers import Input, Conv2D, MaxPool2D
from tensorflow.python.keras.layers.merge import concatenate, average, add
from tensorflow.python.keras.layers import Reshape, SpatialDropout1D
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Layer, InputSpec
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras import initializers, regularizers, constraints, optimizers, layers
from tensorflow.python.keras import backend as K
from helper import *

# python generate_predictions --test_dataset {test}

# load test dataset
test = pd.read_csv('file_name', sep="\n", header=None)
print("Test file loaded!")

print("Preprocessing...")
test = preprocess(test)
print("Preprocessing complete!")

## GET SYNTACTIC REP: TEST

print("Getting dependency and POS tags...")
test['DEP'] = [sent2feature2ngram(row) for row in test['text']]
test["parent_pos"] = [ParentPositions(row) for row in test['text']]
test['TAG'] = [sent2feature2ngram(row, feature="TAG") for row in test['text']]
print("Test tags: complete!")

## BERT REP: TEST

print("Creating BERT Embeddings...")
test_input_ids = tokenize_sentences(test['text'], bert_tokenizer_transformer, 60)
test_input_ids = pad_sequences(test_input_ids, maxlen=60, dtype="long", value=0, truncating="post", padding="post")
test_attention_masks = create_attention_masks(test_input_ids)

maxlen = 30

## COMMON TEST REP | DEP

(X_noisy_test, X_semi_test, X_structured_test) = (tokenizer_dep_noisy.texts_to_sequences(test['DEP']), \
                                                  tokenizer_dep_semi.texts_to_sequences(test['DEP']), \ 
                                                  tokenizer_dep_structured.texts_to_sequences(test['DEP']))
(X_parent_noisy_test, X_parent_semi_test, X_parent_structured_test) = (tokenizer_dep_parent_noisy.texts_to_sequences(test['parent_pos']), \
                                                                       tokenizer_dep_parent_semi.texts_to_sequences(test['parent_pos']), \
                                                                       tokenizer_dep_parent_structured.texts_to_sequences(test['parent_pos']))

X_noisy_test, X_semi_test, X_structured_test = pad_sequences(X_noisy_test, maxlen=maxlen),pad_sequences(X_semi_test, maxlen=maxlen), pad_sequences(X_structured_test, maxlen=maxlen)
X_parent_noisy_test, X_parent_semi_test, X_parent_structured_test = pad_sequences(X_parent_noisy_test, maxlen=maxlen), pad_sequences(X_parent_semi_test, maxlen=maxlen), pad_sequences(X_parent_structured_test, maxlen=maxlen)

## COMMON TEST REP | TAG

(X_tag_noisy_test, X_tag_semi_test, X_tag_structured_test) = (tokenizer_tag_noisy.texts_to_sequences(test['TAG']), \
                                                              tokenizer_tag_semi.texts_to_sequences(test['TAG']), 
                                                              tokenizer_tag_structured.texts_to_sequences(test['TAG']))

X_tag_noisy_test, X_tag_semi_test, X_tag_structured_test = pad_sequences(X_tag_noisy_test, maxlen=maxlen), pad_sequences(X_tag_semi_test, maxlen=maxlen), pad_sequences(X_tag_structured_test, maxlen=maxlen)

# load_model
print("Loading Model...\n\n")
## LOAD FINE-TUNED BERT MODEL
cwd = os.getcwd()
model_save_path = os.path.join(cwd, "seven_finetuned")

bert_tokenizer_transformer = BertTokenizer.from_pretrained(model_save_path)

noisy_vocab_size_tag = 5363
noisy_embedding_matrix_tag = load_embedding_matrix('embedding_matrix_tag_noisy.pickle')

semi_vocab_size_tag = 6137
semi_embedding_matrix_tag = load_embedding_matrix('embedding_matrix_tag_semi.pickle')

structured_vocab_size_tag = 6048
structured_embedding_matrix_tag = load_embedding_matrix('embedding_matrix_tag_structured.pickle')

## PARENT POS TOKENIZER

num_words_dep_parent_noisy = 100
tokenizer_dep_parent_noisy = load_tokenizer('tokenizer_dep_parent_noisy.pickle')

num_words_dep_parent_semi = 200
tokenizer_dep_parent_semi = load_tokenizer('tokenizer_dep_parent_semi.pickle')

num_words_dep_parent_structured = 200
tokenizer_dep_parent_structured = load_tokenizer('tokenizer_dep_parent_structured.pickle')

## LABEL TOKENIZER

num_words_dep_noisy = 6300
tokenizer_dep_noisy = load_tokenizer('tokenizer_dep_noisy.pickle')

num_words_dep_semi = 7300
tokenizer_dep_semi = load_tokenizer('tokenizer_dep_semi.pickle')

num_words_dep_structured = 7400
tokenizer_dep_structured = load_tokenizer('tokenizer_dep_structured.pickle')

## TAG TOKENIZER

tokenizer_tag_noisy = load_tokenizer('tokenizer_tag_noisy.pickle')

tokenizer_tag_semi = load_tokenizer('tokenizer_tag_semi.pickle')

tokenizer_tag_structured = load_tokenizer('tokenizer_tag_structured.pickle')

# aux model

noisy_model = ind_model_noisy(embed_dim = 20, num_heads=5, ff_dim = 128,\
                              maxlen=maxlen, vocab_label=num_words_dep_noisy, vocab_parent_pos=num_words_dep_parent_noisy)

noisy_model.load_weights('_dep_noisy.h5')

semi_model = ind_model_semi(embed_dim = 20, num_heads=5, ff_dim = 128, \
                            maxlen=maxlen, vocab_label=num_words_dep_semi, vocab_parent_pos=num_words_dep_parent_semi)

semi_model.load_weights('_dep_semi.h5')

structured_model = ind_model_structured(embed_dim = 20, num_heads=5, ff_dim = 128, \
                                        maxlen=maxlen, vocab_label=num_words_dep_structured, vocab_parent_pos=num_words_dep_parent_structured)

structured_model.load_weights('_dep_structured.h5')


parameters_dict_noisy = {
    "vocab_size_tag" : noisy_vocab_size_tag,
    "EMBEDDING_DIM_TAG" : 20,
    "embedding_matrix_tag" : noisy_embedding_matrix_tag,
    "maxlen_tag" : maxlen
}

parameters_dict_semi = {
    "vocab_size_tag" : semi_vocab_size_tag,
    "EMBEDDING_DIM_TAG" : 20,
    "embedding_matrix_tag" : semi_embedding_matrix_tag,
    "maxlen_tag" : maxlen
}

parameters_dict_structured = {
    "vocab_size_tag" : structured_vocab_size_tag,
    "EMBEDDING_DIM_TAG" : 20,
    "embedding_matrix_tag" : structured_embedding_matrix_tag,
    "maxlen_tag" : maxlen
}

final_model = final(noisy_model, semi_model, structured_model,\
                    parameters_dict_noisy, parameters_dict_semi, parameters_dict_structured,\
                    max_seq_length=60)

final_model.load_weights('_bert_comb.h5')

print("Model loaded!")

# testing
metrics = [None]*4
metrics = final_model.predict({"label_noisy": np.array(X_noisy_test), "parent_pos_noisy": np.array(X_parent_noisy_test),
                                "label_semi": np.array(X_semi_test), "parent_pos_semi": np.array(X_parent_semi_test),
                                "label_structured": np.array(X_structured_test), "parent_pos_structured": np.array(X_parent_structured_test),
                                "inp_noisy": np.array(X_tag_noisy_test),
                                "inp_semi": np.array(X_tag_semi_test),
                                "inp_structured": np.array(X_tag_structured_test),
                                'input_word_ids': np.array(test_input_ids), 'input_masks': np.array(test_attention_masks)})

y_pred_bool = [np.argmax(el) for el in metrics[-1]]
print(classification_report(y_test, y_pred_bool))