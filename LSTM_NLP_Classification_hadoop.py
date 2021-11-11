
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
import re
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))
from bs4 import BeautifulSoup
import plotly.graph_objects as go
import chart_studio.plotly as py
import cufflinks
from IPython.core.interactiveshell import InteractiveShell
import plotly.figure_factory as ff
InteractiveShell.ast_node_interactivity = 'all'
from chart_studio.plotly import iplot
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')


# In[2]:


df = pd.read_csv('data/hadoop/HADOOP_aug2_word_ori.csv', encoding='cp949')


# In[3]:


df.info()


# In[4]:


print(df.component.value_counts())


# In[5]:


'''# fs, build, security, test, documentation, ipc, io, conf, util, scripts, fs/s3, metrics, native, contrib/hod, yetus, tools, record, ha, net, kms, bin
df.loc[df['component'] == 'fs/swift,test', 'component'] = 'test'
df.loc[df['component'] == 'fs/s3,io', 'component'] = 'io'
df.loc[df['component'] == 'conf,ha', 'component'] = 'conf'
df.loc[df['component'] == 'build,scripts,tools,tools/distcp', 'component'] = 'build'
df.loc[df['component'] == 'benchmarks,conf,metrics', 'component'] = 'conf'
df.loc[df['component'] == 'ha,net', 'component'] = 'ha'
df.loc[df['component'] == 'fs,ipc,security', 'component'] = 'fs'
df.loc[df['component'] == 'documentation,site', 'component'] = 'site'
df.loc[df['component'] == 'conf,ha', 'component'] = 'ha'
df.loc[df['component'] == 'fs,ha', 'component'] = 'ha'
df.loc[df['component'] == 'native,test', 'component'] = 'native'
df.loc[df['component'] == 'conf,fs,fs/s3,ha,io,metrics,performance,record,security,util', 'component'] = 'ha'
#df.loc[df['component'] == 'Credit reporting', 'component'] = 'Credit reporting, credit repair services, or other personal consumer reports'
#df.loc[df['component'] == 'Credit card', 'component'] = 'Credit card or prepaid card'
#df.loc[df['component'] == 'Payday loan', 'component'] = 'Payday loan, title loan, or personal loan'
#df.loc[df['component'] == 'Virtual currency', 'component'] = 'Money transfer, virtual currency, or money service'
#df = df[df.component != 'Other financial service']'''


# In[6]:


for x in range(len(df.component)):
    df.component[x] = df.component[x].split(',')[0]


# In[7]:


print(df.component.value_counts())


# In[8]:


#df['component'].value_counts().sort_values(ascending=False).iplot(kind = 'bar', yTitle = 'Number of Complaints', title = 'Number complaints in each component')


# In[9]:


print(df.loc[1])


# In[10]:


def print_plot(index):
    example = df[df.index == index][['text', 'component']].values[0]
    if len(example) > 0:
        print(example[0])
        print('component:', example[1])
print_plot(2)


# In[12]:


df = df.reset_index(drop=True)
df = df.astype(str)
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = text.replace('x', '')
#    text = re.sub(r'\W+', '', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text
df['text'] = df['text'].apply(clean_text)
df['text'] = df['text'].str.replace('\d+', '')


# In[27]:


# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['text'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# In[28]:


X = tokenizer.texts_to_sequences(df['text'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('X Shape of data tensor:', X.shape)

Y = pd.get_dummies(df['component']).values
print('Y Shape of label tensor:', Y.shape)


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(37, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



print(X_train.shape)
print(Y_train.shape)


epochs = 10
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
