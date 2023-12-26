#!/usr/bin/env python
# coding: utf-8

# # NAME : RATHNA PREETHI M A

# # HOUSE PRICE PREDICTION

# In[ ]:





# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures


# In[3]:


df = pd.read_csv("kc_house_data.csv")


# In[4]:


df.drop(columns=["id","date","zipcode"],axis=1,inplace=True)
df.drop(columns=["condition","long"],inplace=True)


# In[5]:


df.corr()


# In[6]:


chart = sns.relplot(x ="sqft_living", y ="sqft_living15", data = df)
plt.xlabel("sqft_living")
plt.ylabel("sqft_living15")
plt.gca().invert(yaxis)
plt.show()


# In[ ]:


pca = PCA(n_components = 2)
X2D1 = pca.fit_transform(df[["sqft_living","sqft_living15"]])
X2D2 = pca.fit_transform(df[["sqft_lot","sqft_lot15"]])


# In[ ]:


df["sqft_living"] = X2D1[:,0]
df["sqft_lot"] = X2D2[:,0]


# In[ ]:


df.drop(columns=["sqft_living15","sqft_lot15"],axis=1,inplace=True)


# In[ ]:


df[df["yr_renovated"] == 0]["price"].count()


# In[ ]:


df.drop(columns=["yr_renovated","view","waterfront"],axis=1,inplace=True)
df.drop(columns=["sqft_basement","sqft_lot"],axis=1,inplace=True)


# In[ ]:


X = df.drop('price',axis=1)
y = df['price']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=104, test_size=0.2, shuffle=True)


# In[ ]:


sns.set_style('whitegrid')
sns.displot(df['price'], kde=False,color ='red', bins = 30)


# In[ ]:


norm = MinMaxScaler().fit(X_train)
X_train_norm = norm.transform(X_train)
X_test_norm = norm.transform(X_test)


# In[20]:


linReg = LinearRegression()
linReg.fit(X_train,y_train)


# In[21]:


poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)


# In[22]:


polyReg = LinearRegression()
polyReg.fit(X_train_poly, y_train)


# In[23]:


linPred = linReg.predict(X_test)
polyPred = polyReg.predict(X_test_poly)


# In[24]:


print('Linear MSE : ', mean_squared_error(y_test, linPred))
print('Poly MSE : ', mean_squared_error(y_test, polyPred))


# In[25]:


print('Linear MAE : ', mean_absolute_error(y_test, linPred))
print('Poly MAE : ', mean_absolute_error(y_test, polyPred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




