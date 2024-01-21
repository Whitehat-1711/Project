#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[35]:


dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[ : ,:-1].values
y = dataset.iloc[: , -1].values


# In[36]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# In[37]:


x_train


# In[38]:


y_train


# In[39]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[40]:


x_train


# In[41]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)


# In[42]:


print(classifier.predict(sc.fit_transform([[30,87000]])))


# In[43]:


print(classifier.predict(sc.fit_transform([[40,0]])))


# In[44]:


print(classifier.predict(sc.fit_transform([[30,100000]])))


# In[45]:


print(classifier.predict(sc.fit_transform([[50,0]])))


# In[46]:


y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[47]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[68]:


from matplotlib.colors import ListedColormap
x_set,y_set = sc.inverse_transform(x_train),y_train
x1,x2 = np.meshgrid(np.arange(start = x_set[: , 0].min() - 10, stop = x_set[: , 0].max()+10, step = 0.25),
                    np.arange(start = x_set[: , 1].min()-1000, stop = x_set[: , 1].max()+1000, step = 0.25))
plt.contourf(x1,x2, classifier.predict(sc.transform(np.array([x1.ravel(),x2.ravel()]).T)).reshape(x1.shape),
            alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 1], x_set[y_set == j,0], c = ListedColormap(('red','green'))(i), label = j)
    
    plt.legend()
    plt.show()


# In[63]:


x1.min()


# In[64]:


x2.max()


# In[69]:


from matplotlib.colors import ListedColormap

# Assuming 'classifier' is your trained model and 'x_train', 'y_train' are your training data
x_set, y_set = sc.inverse_transform(x_train), y_train

# Creating a meshgrid for visualization
x1, x2 = np.meshgrid(
    np.arange(start=x_set[:, 0].min() - 10, stop=x_set[:, 0].max() + 10, step=0.25),
    np.arange(start=x_set[:, 1].min() - 1000, stop=x_set[:, 1].max() + 1000, step=0.25)
)

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot the contour plot in the first subplot
ax1.contourf(
    x1, x2,
    classifier.predict(sc.transform(np.array([x1.ravel(), x2.ravel()]).T)).reshape(x1.shape),
    alpha=0.75,
    cmap=ListedColormap(('red', 'green'))
)
ax1.set_xlim(x1.min(), x1.max())
ax1.set_ylim(x2.min(), x2.max())
ax1.set_title('Contour Plot')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')

# Plot the scatter plot in the second subplot
for i, j in enumerate(np.unique(y_set)):
    ax2.scatter(x_set[y_set == j, 1], x_set[y_set == j, 0], c=ListedColormap(('red', 'green'))(i), label=j)

ax2.set_title('Scatter Plot')
ax2.set_xlabel('Feature 2')
ax2.set_ylabel('Feature 1')
ax2.legend()

plt.show()


# In[70]:


from matplotlib.colors import ListedColormap

# Assuming 'classifier' is your trained model and 'x_train', 'y_train' are your training data
x_set, y_set = sc.inverse_transform(x_train), y_train

# Creating a meshgrid for visualization
x1, x2 = np.meshgrid(
    np.arange(start=x_set[:, 0].min() - 10, stop=x_set[:, 0].max() + 10, step=0.25),
    np.arange(start=x_set[:, 1].min() - 1000, stop=x_set[:, 1].max() + 1000, step=0.25)
)

# Plot the contour plot
plt.contourf(
    x1, x2,
    classifier.predict(sc.transform(np.array([x1.ravel(), x2.ravel()]).T)).reshape(x1.shape),
    alpha=0.75,
    cmap=ListedColormap(('red', 'green'))
)
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

# Plot the scatter plot on the same axes
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 1], x_set[y_set == j, 0], c=ListedColormap(('red', 'green'))(i), label=j)

plt.title('Contour and Scatter Plot Overlay')
plt.xlabel('Feature 2')
plt.ylabel('Feature 1')
plt.legend()
plt.show()


# In[71]:


from matplotlib.colors import ListedColormap

# Assuming 'classifier' is your trained model and 'x_train', 'y_train' are your training data
x_set, y_set = sc.inverse_transform(x_train), y_train

# Creating a meshgrid for visualization
x1, x2 = np.meshgrid(
    np.arange(start=x_set[:, 0].min() - 10, stop=x_set[:, 0].max() + 10, step=0.25),
    np.arange(start=x_set[:, 1].min() - 1000, stop=x_set[:, 1].max() + 1000, step=0.25)
)

# Plot the scatter plot first
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 1], x_set[y_set == j, 0], c=ListedColormap(('red', 'green'))(i), label=j)

# Plot the contour plot on the same axes
plt.contourf(
    x1, x2,
    classifier.predict(sc.transform(np.array([x1.ravel(), x2.ravel()]).T)).reshape(x1.shape),
    alpha=0.75,
    cmap=ListedColormap(('red', 'green'))
)

plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

plt.title('Contour and Scatter Plot Overlay')
plt.xlabel('Feature 2')
plt.ylabel('Feature 1')
plt.legend()
plt.show()


# In[ ]:




