#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[ : , :-1].values
y = dataset.iloc[ : , -1].values


# In[3]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# In[4]:


x_train


# In[5]:


y_train


# In[6]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[7]:


x_train


# In[8]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(x_train, y_train)


# In[16]:


print(classifier.predict(sc.transform([[80,87000]])))


# In[17]:


print(classifier.predict(sc.transform([[30,87000]])))
print(classifier.predict(sc.transform([[40,0]])))
print(classifier.predict(sc.transform([[40,100000]])))
print(classifier.predict(sc.transform([[50,0]])))


# In[19]:


print(classifier.predict(sc.transform([[18,0]])))
print(classifier.predict(sc.transform([[22,600000]])))
print(classifier.predict(sc.transform([[35,2500000]])))
print(classifier.predict(sc.transform([[60,100000000]])))


# In[14]:


y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[15]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test, y_pred)

