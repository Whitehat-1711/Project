#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[3]:


dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[ : , :-1].values
y = dataset.iloc[ :, -1].values


# In[4]:


x


# In[5]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# In[8]:


x_train


# In[9]:


x_test


# In[10]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[11]:


x_train


# In[12]:


x_test


# In[13]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy' , random_state = 0)
classifier.fit(x_train, y_train)


# In[15]:


print(classifier.predict(sc.transform([[30,87000]])))
print(classifier.predict(sc.transform([[40,0]])))
print(classifier.predict(sc.transform([[40,100000]])))
print(classifier.predict(sc.transform([[50,0]])))


# In[16]:


print(classifier.predict(sc.transform([[18,0]])))
print(classifier.predict(sc.transform([[22,600000]])))
print(classifier.predict(sc.transform([[35,2500000]])))
print(classifier.predict(sc.transform([[60,100000000]])))


# In[ ]:




