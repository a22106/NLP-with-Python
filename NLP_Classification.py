import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU,SimpleRNN
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import batch_normalization
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import seaborn as sns

import nltk
import re, string


ISLANDORA_LABEL_NUM = 67

HADOOP_LABEL_NUM = 37
FCREPO_LABEL_NUM = 22

labels_Num = {
    'ISLANDORA': ISLANDORA_LABEL_NUM, 
    'HADOOP': HADOOP_LABEL_NUM, 'FCREPO': FCREPO_LABEL_NUM
    }

class NLP_classification_aug:
    def __init__(self, dataset_name, augmenter_name, augment_size = 7, nlp_model_name = 'bert'):
        self.dataset_name = dataset_name
        self.labels_num = labels_Num[dataset_name]

        if augmenter_name == 'OCR' or augmenter_name == 'Keyboard':
            self.augmentation_type = 'char'
        else:
            self.augmentation_type = 'word'

        self.augmenter_name = augmenter_name
        self.aug_mul = augment_size
        self.nlp_model_name = nlp_model_name
        self.transform_model = {'bert': 'bert-base-uncased', 'roberta': 'roberta-base', 'xlnet': 'xlnet-base-uncased', 'distilbert': 'distilbert-base-uncased', 'xlm': 'xlm-roberta-base', 'electra': 'google/electra-base-discriminator'}

        # 데이터 위치 data location
        self.data_location_ori = 'data/{}/{}.csv'.format(self.dataset_name, self.dataset_name)
        # dataset name: hadoop, islandora, fcrepo
        # augmentation type: char, word
        # augmenter name: Synonym, Split etc.
        self.data_location_aug = 'data/{}/{}_{}_{}.csv'.format(self.dataset_name, self.dataset_name, self.augmentation_type, self.augmenter_name)     

        # 데이터 변수 입력
        self.data_ori = pd.read_csv(self.data_location_ori) # original data
        self.data = pd.read_csv(self.data_location_aug) # augmented data
        print(self.data.head())
        self.len_data = len(self.data_ori)
        self.eval_index = []
        self.test_index = []
        self.data_comp = []


    def preprocess(self):
        refined_data = []
        for item in self.data['text']:
            #1. Remove \r 
            current_desc = item.replace('\r', ' ')    
            #2. Remove URLs
            current_desc = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', current_desc)    
            #4. Remove hex code
            current_desc = re.sub(r'(\w+)0x\w+', '', current_desc) 
            #5. Change to lower case
            current_desc = current_desc.lower()   
            #6. Tokenize
            #current_desc_tokens = tokenizer(current_desc, add_special_tokens= True)
            #7. Strip trailing punctuation marks
            #current_desc_filter = [word.strip(string.punctuation) for word in current_desc_tokens]
            #8. Join the lists
            #current_data = current_desc_filter
            #current_data = list(filter(None, current_data))
            refined_data.append(current_desc)
        self.data['text'] = refined_data

    def split_data(self):
                #self.data.drop(['labels'], axis = 1, inplace= True)

        for x in range(len(self.data.component)):
            self.data_comp.append(self.data.component[x].split(',')[0])
        self.data_comp = pd.DataFrame({'component': self.data_comp})
        
        self.Y = pd.get_dummies(self.data[set(self.data_comp)])

        #원본 데이터 split
        self.train_ori, self.test_ori = train_test_split(self.data_ori, test_size = 0.2, random_state=42)
        
        self.eval_index_list = list(self.test_ori.index)
        self.eval_index_list.sort()
        # 테스트 데이터 title과 description 합쳐 text column 생성
        self.test_ori['text'] = list(self.test_ori.title + " "+ self.test_ori.description)

        self.eval_index = []
        for aug_num in range(7):
            iidf2 = [i + self.len_data* aug_num for i in self.eval_index_list]
            self.eval_index = self.eval_index + iidf2
        
        # train data 생성
        self.train_data = self.data.drop(index = self.eval_index)

        # test data 생성 (원본 데이터의 일부)
        #self.test_data = pd.get_dummies(self.y_ori[set(self.data_comp.component)]).sort_index()
        # xtrain, ytrain // xvalid, yvalid
        self.xtrain = self.train_data['text']
        self.xvalid = self.test_ori['text']
        self.ytrain = pd.get_dummies(self.train_data[set(self.data_comp.component)])
        self.yvalid = pd.get_dummies(self.test_ori[set(self.data_comp.component)])
        self.label_nums = 37

    def tokenize_LSTM(self):
        self.X_train = self.X_train.sample(frac=1).reset_index(drop=True)
        X_train_index = list(self.X_train.index)
        self.X_test = self.df['text'].drop(X_train_index)
        self.X_test = self.X_test.sample(frac=1).reset_index(drop=True)

        self.Y_train = self.Y_train.sample(frac=1).reset_index(drop=True)
        self.Y_test = self.Y.drop(X_train_index)
        self.Y_test = self.Y_test.sample(frac=1).reset_index(drop=True)


    def tokenize_CNN(self):
        # The maximum number of words to be used. (most frequent)
        self.MAX_NB_WORDS = 25000
        # Max number of words in each complaint.
        self.MAX_SEQUENCE_LENGTH = 250
        self.EMBEDDING_DIM = 100 # how big is each word vector

        token = text.Tokenizer(num_words = self.MAX_NB_WORDS)
        token.fit_on_texts(list(self.xtrain)+list(self.xvalid))
        xtrain_seq = token.texts_to_sequences(self.xtrain)
        xvalid_seq = token.texts_to_sequences(self.xvalid)

        self.xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=self.MAX_NB_WORDS)
        self.xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=self.MAX_NB_WORDS)

        self.word_index = token.word_index



    def set_model_LSTM(self):
        self.modelLSTM = Sequential()
        self.modelLSTM.add(Embedding(self.MAX_NB_WORDS, self.EMBEDDING_DIM, input_length=self.MAX_NB_WORDS))
        self.modelLSTM.add(SpatialDropout1D(0.2))
        self.modelLSTM.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        self.modelLSTM.add(Dense(37, activation='sigmoid'))
        self.modelLSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.Recall(top_k = 5)])
        print(self.modelLSTM.summary())        

    def run_model_LSTM(self, x_train, y_train):
        epochs = 20
        batch_size = 64

        self.history = self.modelLSTM.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1)

    def test_model(self, xvalid, yvalid):
        self.accr = self.modelLSTM.evaluate(xvalid, yvalid)
        print('Test set\n Loss: {:0.3f}\n Accuracy: {0.3f}'.format(self.accr[0], self.accr[1]))
#    def sef_model_CNN(self):
    
    def set_model_RNN(self):
        
        # A simpleRNN without any pretrained embeddings and one dense layer
        model = Sequential()
        model.add(Embedding(len(word_index) + 1, 300, input_length=MAX_NB_WORDS))
        model.add(SimpleRNN(100))
        model.add(Dense(label_nums, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.Recall(top_k = 5)])
            
        model.summary()
    
    
#    def set_model_transformers(self):

testClass = NLP_classification_aug("HADOOP","Synonym")

testClass.preprocess()
testClass.split_data()
testClass.tokenize_CNN()
testClass.set_model_LSTM()
testClass.run_model_LSTM(testClass.xtrain_pad, testClass.ytrain)
testClass.test_model(testClass.xvalid_pad, testClass.yvalid)