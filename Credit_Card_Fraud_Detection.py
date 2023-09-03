#!/usr/bin/env python
# coding: utf-8

# # Data Science Internship
# 
# # Task 03: Credit Card Fraud Detection
# 
# # M.Faraz Shoaib
# 
# ## Importing Libraries and Data
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


credit_data = pd.read_csv('creditcard.csv')
credit_data.info()


# In[3]:


credit_data.describe()


# In[4]:


credit_data.count().isna()


# In[5]:


credit_data.corr()


# In[6]:


print(credit_data)


# ## Checking the Fraud and Correct Transaction

# In[7]:


import plotly.graph_objects as go
f = credit_data['Class'].value_counts()
credit_fault_detection_df = pd.DataFrame({'Class':f.index,'values':f.values})
print(credit_fault_detection_df)
#now we know how many are faulty transactions so plotting it for ease of eyes


# In[8]:


data = go.Bar(x = credit_fault_detection_df['Class'], y = credit_fault_detection_df['values'],text=credit_fault_detection_df['values'])
data = [data]
layout = go.Layout(title ="Credit Card Fraud Detection", xaxis_title = "Class (Not Fraud = 0 : Fraud = 1)", yaxis_title = "No. of transaction", colorway=['Green'])
fig = go.Figure(data = data,layout = layout)
fig.show()


# ## Splitting Data for Training and Testing

# In[9]:


from sklearn.model_selection import train_test_split
X = credit_data.drop(["Class"],axis = 1).values
y = credit_data["Class"].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 40)
X_train.shape


# ## Now for Pre-Processing and Normalization
# 

# In[10]:


from sklearn.preprocessing import StandardScaler 

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)
X_train_scaled.shape


# ## Importing Models and Metrics

# In[11]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


# In[12]:


lr = LogisticRegression(solver = 'newton-cg')
dtc = DecisionTreeClassifier()
gnb = GaussianNB()
knn = KNeighborsClassifier()
rfc = RandomForestClassifier()

models = [lr, dtc, gnb, knn, rfc]


# ## Model Building and Predicting 

# In[13]:


for model in models:
    model.fit(X_train_scaled,y_train)
    y_pred = model.predict(X_test_scaled)
    report = classification_report(y_test,y_pred)
    print(type(model).__name__,"The accuracy is", accuracy_score(y_test,y_pred))
    print(type(model).__name__,"The precision score is", precision_score(y_test,y_pred))
    print(type(model).__name__,"The f1_score is", f1_score(y_test,y_pred))
    print(type(model).__name__,"The recall score is", recall_score(y_test,y_pred))
    print(type(model).__name__,"Classification Report is: \n", report)


# ## Now we will observe the effect of oversampling and undersampling( Imbalancing issue solutions)

# ### Oversampling

# In[14]:


over_sampler = RandomOverSampler(sampling_strategy = 'minority')
X_resampled,y_resampled = over_sampler.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled,y_resampled,test_size = 0.2, random_state = 40)
X_train_scaled1 = sc.fit_transform(X_train)
X_test_scaled1 = sc.transform(X_test)


# In[15]:


for model in models:
    model.fit(X_train_scaled1,y_train)
    y_pred = model.predict(X_test_scaled1)
    report = classification_report(y_test,y_pred)
    print(type(model).__name__,"The accuracy is", accuracy_score(y_test,y_pred))
    print(type(model).__name__,"The precision score is", precision_score(y_test,y_pred))
    print(type(model).__name__,"The f1_score is", f1_score(y_test,y_pred))
    print(type(model).__name__,"The recall score is", recall_score(y_test,y_pred))
    print(type(model).__name__,"Classification Report is: \n", report)


# ### Undersampling

# In[16]:


under_sampler = RandomUnderSampler(sampling_strategy = 'majority')
X_resampled,y_resampled = under_sampler.fit_resample(X, y)
# standardization wrt to resampled dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled,y_resampled,test_size = 0.2, random_state = 40)
X_train_scaled2 = sc.fit_transform(X_train)
X_test_scaled2 = sc.transform(X_test)


# In[17]:


for model in models:
    model.fit(X_train_scaled2,y_train)
    y_pred = model.predict(X_test_scaled2)
    report = classification_report(y_test,y_pred)
    print(type(model).__name__,"The accuracy is", accuracy_score(y_test,y_pred))
    print(type(model).__name__,"The precision score is", precision_score(y_test,y_pred))
    print(type(model).__name__,"The f1_score is", f1_score(y_test,y_pred))
    print(type(model).__name__,"The recall score is", recall_score(y_test,y_pred))
    print(type(model).__name__,"Classification Report is: \n", report)


# ## Thank You

# In[ ]:




