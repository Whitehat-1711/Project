#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[4]:


dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[: , :-1].values
y = dataset.iloc[: , -1].values


# In[5]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state =0)


# In[6]:


print(x_train)


# In[7]:


print(x_test)


# In[8]:


y_train


# In[9]:


y_test


# In[10]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[12]:


x_test


# In[13]:


x_train


# In[14]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=11, metric='minkowski', p=2)
classifier.fit(x_train, y_train)


# In[15]:


print(classifier.predict(sc.transform([[55,120000]])))


# In[16]:


y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


# In[17]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test, y_pred)

