#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df= pd.read_csv('C:/Users/Fahad/Downloads/FAHAD/Data_analysis/cardio/cardio_train.csv', ';')


# In[3]:


df


# In[4]:


df['age'] = (df['age'] /365).round(0)


# In[5]:


df


# In[6]:


df.describe()


# In[7]:


y= df.cardio


# In[8]:


x=df.drop(['cardio'], axis=1)


# In[9]:


x


# In[10]:


from sklearn.feature_selection import  SelectKBest


# In[11]:


from sklearn.feature_selection import f_classif


# In[12]:


Bestfit = SelectKBest(score_func=f_classif)


# In[13]:


Bestfit.fit(x,y)


# In[14]:


best_score=pd.DataFrame(Bestfit.scores_, columns=['Score Value'])


# In[15]:


best_score


# In[16]:


Name_of_X= pd.DataFrame(x.columns)


# In[17]:


Concat= pd.concat ([Name_of_X,best_score], axis=1)


# In[18]:


Concat


# In[19]:


Concat.nlargest(8,'Score Value')


# In[20]:


x=df.drop(['cardio','id','gender','height'], axis=1)


# In[21]:


x


# In[22]:


y


# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


xtrain,xtest,ytrain,ytest =  train_test_split(x,y, test_size=0.2,random_state=1)


# In[25]:


xtrain


# In[26]:


ytrain


# In[27]:


from sklearn.ensemble import RandomForestClassifier


# In[28]:


Random_forest= RandomForestClassifier()


# In[29]:


Random_forest.fit(xtrain,ytrain)


# In[30]:


Random_forest.score(xtest,ytest)


# In[31]:


from sklearn.tree import DecisionTreeClassifier


# In[32]:


Decision_Tree= DecisionTreeClassifier()


# In[33]:


Decision_Tree.fit(xtrain,ytrain)


# In[34]:


Decision_Tree.score(xtest,ytest)


# In[ ]:




