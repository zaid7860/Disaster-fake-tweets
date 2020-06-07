#!/usr/bin/env python
# coding: utf-8

# In[265]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import os
import gc

import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

import spacy
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# In[190]:


raw = pd.read_csv('train.csv',)


# In[ ]:





# In[191]:


#raw = pd.read_csv('train.csv','test.csv')
raw=raw.fillna('')
raw.drop(['location','id',],axis=1,inplace=True)


# In[192]:


raw.head(10)


# In[193]:


# Selecting 'text' values that are non-disastrous
non_disastrous = raw[raw['target']==0]['text']
disastrous = raw[raw['target']==1]['text']

# I inputted 4 to select the 4th row of the non-disastrous values
non_disastrous.values[10]


# In[194]:


sns.countplot(x='target',data=raw)


# In[195]:


raw.describe()


# # Data Cleaning

# In[196]:


# To get the results in 4 decemal points
SAFE_DIV = 0.0001 

STOP_WORDS = stopwords.words("english")


def preprocess(x):
    x = str(x).lower()
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")                           .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")                           .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")                           .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")                           .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")                           .replace("€", " euro ").replace("'ll", " will")
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)
    x = re.sub(r"http\S+", "", x)
    
    porter = PorterStemmer()
    pattern = re.compile('\W')
    
    if type(x) == type(''):
        x = re.sub(pattern, ' ', x)
    
    
    if type(x) == type(''):
        x = porter.stem(x)
        example1 = BeautifulSoup(x)
        x = example1.get_text()
               
    
    return x
    


# In[197]:


raw['text'] = raw['text'].apply(lambda x: preprocess(x))


# In[198]:


raw['keyword']=raw['keyword'].apply(lambda x : preprocess(x))


# In[199]:


non_disastrous = raw[raw['target']==0]['text']
disastrous = raw[raw['target']==1]['text']

# I inputted 4 to select the 4th row of the non-disastrous values
non_disastrous.values[10]


# # wordcloud

# In[249]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
fig, (ax1) = plt.subplots(1, figsize=[9, 9])
wordcloud = WordCloud( background_color='white',
                        width=600,
                        height=600).generate(" ".join(disastrous[:40]))
ax1.imshow(wordcloud)
ax1.axis('off')
ax1.set_title('Frequent Words',fontsize=6);


# In[245]:


import matplotlib.pyplot as plt
fig, (ax1) = plt.subplots(1, figsize=[16, 10])
wordcloud = WordCloud( background_color='white',
                        width=600,
                        height=600).generate(" ".join(non_disastrous[:50]))
ax1.imshow(wordcloud)
ax1.axis('off')
ax1.set_title('Frequent Words',fontsize=10);


# # Training Word2Vec   using Glove

# In[288]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from sklearn import linear_model


count_vect = CountVectorizer()
count_vect.fit(raw['text'])
joblib.dump(count_vect, 'count_vect.pkl')
X = count_vect.transform(raw['text'])
print(X.shape)
Y = raw['target'].values
clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3, eta0=0.1, alpha=0.001)
clf.fit(X, Y)
joblib.dump(clf, 'model.pkl')


# In[344]:


def predict(string):
    clf = joblib.load('model.pkl')
    count_vect = joblib.load('count_vect.pkl')
    review_text = preprocess(string)
    test_vect = count_vect.transform(([review_text]))
    pred = clf.predict(test_vect)
    print(pred)
    if pred==0:
        prediction = "Non-disasterous"
    else:
        prediction = "Disasterous"
    return prediction


# In[315]:


s='help car flood'


# In[316]:


scl= preprocess(s)


# In[317]:


scl1 = count_vect.transform([scl])


# In[318]:


pred = clf.predict(scl1)


# In[345]:


preddd  =  predict('help car flood')


# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




