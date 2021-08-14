
import numpy as np
import pandas as pd
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings('ignore')

data_train = pd.read_csv('nlp-getting-started\\train.csv')
# test_df = pd.read_csv('nlp-getting-started\\test.csv')
# total = data_train.isnull().sum().sort_values(ascending=False)
# percent = (data_train.isnull().sum()/data_train.isnull().count()).sort_values(ascending=False)
# missing_data = pd.concat([total,percent],axis=1,keys=['Total', 'Percent'])
# print(total)
# print(missing_data.head(data_train.shape[1]))
data_train = data_train.drop(['location', 'keyword', 'id'], axis=1)
corpus = []
pstem = PorterStemmer()
for i in range(data_train['text'].shape[0]):
    tweet = re.sub("[^a-zA-Z]", ' ', data_train['text'][i])
    tweet = tweet.lower()
    tweet = tweet.split()
    tweet = [pstem.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    corpus.append(tweet)
# print(pd.DataFrame(corpus)[0].head(10))
uniqueWordFrequents = {}
for tweet in corpus:
    for word in tweet.split():
        if(word in uniqueWordFrequents.keys()):
            uniqueWordFrequents[word] += 1
        else:
            uniqueWordFrequents[word] = 1
uniqueWordFrequents = pd.DataFrame.from_dict(uniqueWordFrequents,orient='index',columns=['Word Frequent'])
uniqueWordFrequents.sort_values(by=['Word Frequent'], inplace=True, ascending=False)
# print(uniqueWordFrequents.head(10))
uniqueWordFrequents = uniqueWordFrequents[uniqueWordFrequents['Word Frequent'] >= 20]
counVec = CountVectorizer(max_features = uniqueWordFrequents.shape[0])
bagOfWords = counVec.fit_transform(corpus).toarray()
X = bagOfWords
y = data_train['target']
print("X shape = ",X.shape)
print("y shape = ",y.shape)
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.20, random_state=55, shuffle =True)
print('data splitting successfully')
LogisticRegression = LogisticRegression(penalty='l2', solver='saga',random_state = 55)
LogisticRegression.fit(X_train,y_train)
print("LogisticRegression Classifier model run successfully")
print(' Train Score is   : ', LogisticRegression.score(X_train, y_train))
print(' Test Score is    : ', LogisticRegression.score(X_test, y_test))
y_pred = LogisticRegression.predict(X_test)
print(' F1 Score is      : ', f1_score(y_test, y_pred))
print('--------------------------------------------------------------------------')