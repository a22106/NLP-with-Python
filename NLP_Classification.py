import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Activation, Flatten, GlobalMaxPooling1D, Dropout, Conv1D
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout

import re
import nltk
import chart_studio
from nltk.corpus import stopwords
from nltk import word_tokenize
STOPWORDS = set(stopwords.words('english'))
from bs4 import BeautifulSoup
import plotly.graph_objects as go
import chart_studio.plotly as py
import cufflinks
from IPython.core.interactiveshell import InteractiveShell
import plotly.figure_factory as ff


ISLANDORA_LABEL_NUM = 66

HADOOP_LABEL_NUM = 37
FCREPO_LABEL_NUM = 22

labels_Num = {
    'ISLANDORA': ISLANDORA_LABEL_NUM, 
    'HADOOP': HADOOP_LABEL_NUM, 'FCREPO': FCREPO_LABEL_NUM
    }

class NLP_classification_aug:
    def __init__(self, dataset_name, augmenter_name, augment_size = 7, nlp_model_name = 'bert'):
        # The maximum number of words to be used. (most frequent)
        self.MAX_NB_WORDS = 25000
        # Max number of words in each complaint.
        self.MAX_SEQUENCE_LENGTH = 250
        self.EMBEDDING_DIM = 100 # how big is each word vector
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
        self.data = pd.read_csv(self.data_location_aug, encoding='cp949') # augmented data

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
        for aug_num in range(2):
            iidf2 = [i + self.len_data* aug_num for i in self.eval_index_list]
            self.eval_index = self.eval_index + iidf2
        self.eval_index.__sizeof__
        # train data 생성
        self.train_data = self.data.drop(index = self.eval_index)

        # test data 생성 (원본 데이터의 일부)
        #self.test_data = pd.get_dummies(self.y_ori[set(self.data_comp.component)]).sort_index()
        # xtrain, ytrain // xvalid, yvalid
        self.xtrain = self.train_data['text']
        self.xvalid = self.test_ori['text']
        self.ytrain = pd.get_dummies(self.train_data[set(self.data_comp.component)])
        self.yvalid = pd.get_dummies(self.test_ori[set(self.data_comp.component)])
        self.label_nums = labels_Num[self.dataset_name]

    def tokenize_LSTM(self, xtrain, xvalid):
        self.tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        self.tokenizer.fit_on_texts(self.data['text'].values)
        word_index = self.tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        self.xtrain_tok_lstm = xtrain.sample(frac=1).reset_index(drop=True)
        self.xtrain_index = list(xtrain.index)
        self.xvalid_tok_lstm = xvalid.sample(frac=1).reset_index(drop=True)
        self.xvalid_index = list(xtrain.index)


    def tokenize_CNN(self):

        tokenizer = Tokenizer(num_words = self.MAX_NB_WORDS)
        tokenizer.fit_on_texts(self.data['text'].values)
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))



        self.xtrain_sample = self.xtrain.sample(frac=1).reset_index(drop=True)
        self.xvalid_sample = self.xvalid.sample(frac=1).reset_index(drop=True)

        xtrain_seq = tokenizer.texts_to_sequences(self.xtrain_sample.values)
        xvalid_seq = tokenizer.texts_to_sequences(self.xvalid_sample.values)

        self.xtrain_pad = pad_sequences(xtrain_seq, maxlen=self.MAX_SEQUENCE_LENGTH)
        self.xvalid_pad = pad_sequences(xvalid_seq, maxlen=self.MAX_SEQUENCE_LENGTH)
        print('X train Shape of data tensor:', self.xtrain_pad.shape,'X valid: ', self.xvalid_pad.shape)
        print('Y Shape of label tensor:', self.ytrain.shape,'y valid: ', self.yvalid.shape)



    def set_model_LSTM(self, topk):
        # output layer: sigmoid, hidden layer: relu
        self.modelLSTM = Sequential()
        self.modelLSTM.add(Embedding(self.MAX_NB_WORDS, self.EMBEDDING_DIM, input_length= self.MAX_SEQUENCE_LENGTH))
        self.modelLSTM.add(SpatialDropout1D(0.2))
        self.modelLSTM.add(LSTM(100, dropout=0.5, recurrent_dropout=0.2))
        self.modelLSTM.add(Dense(labels_Num[self.dataset_name], activation='sigmoid'))
        self.modelLSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.Recall(top_k = topk)])
        print(self.modelLSTM.summary())

    def run_model_LSTM(self, x_train, y_train):
        epochs = 10
        batch_size = 80

        self.history = self.modelLSTM.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1)

    def test_model(self, xvalid, yvalid):
        self.accr = self.modelLSTM.evaluate(xvalid, yvalid)
        #print('Test set\n Loss: {:0.3f}\n Accuracy: {0.3f}'.format(self.accr[0], self.accr[1]))
#    def sef_model_CNN(self):

    '''        self.MAX_NB_WORDS = 25000
            # Max number of words in each complaint.
            self.MAX_SEQUENCE_LENGTH = 250
            self.EMBEDDING_DIM = 100 # how big is each word vector'''

    def set_model_CNN(self, topk):
        self.modelCNN = Sequential()
        self.modelCNN.add(Dropout(0.1))
        self.modelCNN.add(Conv1D(self.MAX_SEQUENCE_LENGTH, 3, padding='valid', activation='relu', strides=1))
        self.modelCNN.add(GlobalMaxPooling1D())
        self.modelCNN.add(Dense(labels_Num[self.dataset_name], activation='sigmoid'))
        self.modelCNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.Recall(top_k = topk)])
        print(self.modelCNN.summary())

    def run_model_CNN(self, x_train, y_train):
        epochs = 10
        batch_size = 80

        self.history = self.modelCNN.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1)

    def test_model_CNN(self, xvalid, yvalid):
        cnn_model_test = load_model('model-conv1d.h5') 
        metrics = cnn_model_test.evaluate(xvalid, yvalid)
        print("{}: {}".format(self.modelCNN.metrics_names[0], metrics[0]))
        print("{}: {}".format(self.modelCNN.metrics_names[1], metrics[1]))
    
    def set_model_RNN(self):
        
        # A simpleRNN without any pretrained embeddings and one dense layer
        model = Sequential()
        model.add(Embedding(len(word_index) + 1, 300, input_length=MAX_NB_WORDS))
        model.add(SimpleRNN(100))
        model.add(Dense(label_nums, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.Recall(top_k = 5)])
            
        model.summary()

    
    #    def set_model_transformers(self):

#testClass = NLP_classification_aug("HADOOP","Synonym")

'''testClass.preprocess()
testClass.split_data()
testClass.tokenize_CNN()
print(testClass.ytrain)
testClass.set_model_LSTM(testClass.xtrain, 5)
#testClass.run_model_LSTM(testClass.xtrain_pad, testClass.ytrain)
#testClass.test_model(testClass.xvalid_pad, testClass.yvalid)'''

dataset_name = ['HADOOP', 'ISLANDORA', 'FCREPO']
#augmenter_name = ["OCR", "Keyboard", "Spelling", "ContextualWordEmbs", "Synonym", "Antonym", "Split"]
augmenter_name = ["Keyboard", "ContextualWordEmbs", "Synonym"]
nlp_model = ['bert', 'distilbert', 'robert']

word_hist = []
word_hist_all = []
for dataset in dataset_name:
    for augment in augmenter_name:
        for topk in range(5, 16, 5):
            ml = NLP_classification_aug(dataset, augment)
            ml.preprocess()
            ml.split_data()
            ml.tokenize_CNN()
            ml.set_model_LSTM(topk)
            ml.run_model_LSTM(ml.xtrain_pad, ml.ytrain)
            ml.test_model(ml.xvalid_pad, ml.yvalid)

            word_hist.append(ml.history)
            
        word_hist_all = []
        for i in range(1):
            word_hist_all.append(word_hist[i].history)


        for x in range(1):
            # convert the history dict to a pandas DataFrame
            hist_df = pd.DataFrame(word_hist_all[x])

            # save to csv
            hist_csv_file = 'history/20211205/HistoryRecallat{}_data{}_project{}.csv'.format(int(x%3)*5+5, augment, dataset)
            with open(hist_csv_file, mode = 'w') as f:
                hist_df.to_csv(f)

# CNN
'''word_hist = []
word_hist_all = []
for dataset in dataset_name:
    for augment in augmenter_name:
        for topk in range(5, 16, 5):
            ml = NLP_classification_aug(dataset, augment)
            ml.preprocess()
            ml.split_data()
            ml.tokenize_CNN()
            ml.set_model_LSTM(topk)
            #ml.run_model_LSTM(ml.xtrain_pad, ml.ytrain)
            #ml.test_model_LSTM(ml.xvalid_pad, ml.yvalid)

            word_hist.append(ml.history)
            
        word_hist_all = []
        for i in range(3):
            word_hist_all.append(word_hist[i].history)


        for x in range(3):
            # convert the history dict to a pandas DataFrame
            hist_df = pd.DataFrame(word_hist_all[x])

            # save to csv
            hist_csv_file = 'history/20211205/HistoryRecallat{}_data{}_project{}.csv'.format(int(x%3)*5+5, augment, dataset)
            with open(hist_csv_file, mode = 'w') as f:
                hist_df.to_csv(f)'''