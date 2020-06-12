# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:36:34 2020

@author: amreen sultana
"""

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle 

#reading data
dataset=pd.read_csv("Tweets.csv")

#data preprocessing
gemeentes = ["@JetBlue", "@AmericanAi", "@SouthwestAir","@united","@USAirways","@VirginAmerica"]

dataset['text'] =  dataset['text'].str.replace('|'.join(gemeentes),'')
review =  dataset['text']

dataset.drop("tweet_id",axis=1,inplace=True)
dataset.drop("airline_sentiment_confidence",axis=1,inplace=True)
dataset.drop("negativereason",axis=1,inplace=True)
dataset.drop("negativereason_confidence",axis=1,inplace=True)
dataset.drop("airline",axis=1,inplace=True)
dataset.drop("airline_sentiment_gold",axis=1,inplace=True)
dataset.drop("name",axis=1,inplace=True)
dataset.drop("negativereason_gold",axis=1,inplace=True)
dataset.drop("retweet_count",axis=1,inplace=True)
dataset.drop("tweet_coord",axis=1,inplace=True)
dataset.drop("tweet_created",axis=1,inplace=True)
dataset.drop("tweet_location",axis=1,inplace=True)
dataset.drop("user_timezone",axis=1,inplace=True)

dataset.isnull().any()

dataset["airline_sentiment"].unique()


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dataset["airline_sentiment"]=le.fit_transform(dataset['airline_sentiment'])


from sklearn.utils import resample
df_1=dataset[dataset['airline_sentiment']==1]
df_2=dataset[dataset['airline_sentiment']==2]

df_0=(dataset[dataset['airline_sentiment']==0]).sample(n=3000,random_state=42)

df_1_upsample=resample(df_1,replace=True,n_samples=3000,random_state=123)
df_2_upsample=resample(df_2,replace=True,n_samples=3000,random_state=124)


train_df=pd.concat([df_0,df_1_upsample,df_2_upsample])

train_df['airline_sentiment'].unique()



from sklearn.preprocessing import OneHotEncoder
one=OneHotEncoder()
y=train_df.iloc[:,0:1].values



z=one.fit_transform(y[:,0:1]).toarray()
y=np.delete(y,0,axis=1)
y=np.concatenate((z,y),axis=1)
print(y)
import re

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords


from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()

data=[]
train_df.reset_index(inplace = True)

for i in range(0,9000):
    #1step1 : replace regular expressions , ! @ #. ect ae 
    review = train_df["text"][i]
    review = re.sub('[^a-zA-Z]',' ',review)
    review = review.lower()
    review = review.split()
    #stp words , is are there here where it this that 
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english')) ]
    review = ' '.join(review)
    data.append(review)


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=11000)
x1=cv.fit_transform(data).toarray()
with open('CountVectorizer','wb') as file:
    pickle.dump(cv,file)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x1,y, test_size = 0.2, random_state =0)
#model building
from keras.models import Sequential
from keras.layers import Dense
model=Sequential()


model.add(Dense(units=4000,activation="relu",init="uniform"))


model.add(Dense(units=21610,activation="sigmoid",init="uniform"))
model.add(Dense(units=21610,activation="sigmoid",init="uniform"))



model.add(Dense(units=3,activation="softmax",init="uniform"))


model.compile(optimizer="adam",loss="categorical_crossentropy")


model.fit(x_train,y_train,epochs=20,batch_size=32,verbose = 5)
model.save("sentiment.h5")
y_pred=model.predict_classes(x_test)
print(y_pred)
x_intent="it was worst"
x_intent=cv.transform([x_intent])
y_pred=model.predict(x_intent)
classes=model.predict_classes(x_intent)


