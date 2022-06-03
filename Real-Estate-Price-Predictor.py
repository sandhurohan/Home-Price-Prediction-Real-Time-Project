#!/usr/bin/env python
# coding: utf-8

# # Real Estate Price Predictor

# In[1]:


import pandas as pd


# In[2]:


housing=pd.read_csv("data.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS'].value_counts()


# In[6]:


housing.describe()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins=50,figsize=(20,15))
plt.show()


# ## Train-Test Splitting

# In[8]:


# How train test spliting actually Work in Sciket Learn (For LEARNING ONLY)
#import numpy as np
# def split_train_test(data,test_ratio):
#     np.random.seed(42)
#     shuffled=np.random.permutation(len(data))
#     #print(shuffled)
#     test_set_size=int(len(data)*test_ratio)
#     test_indices=shuffled[:test_set_size]
#     train_indices=shuffled[test_set_size:]
#     return data.iloc[train_indices],data.iloc[test_indices]

# In long run normal mehod will become less effective because 
# model will see all data points at some point because we are using shuffilng.
# thats why we use random.seed() because it fix data points shuffled such that training & testing remain unique.
# 42 is usually everwhere but we can use any no.
# train_set,test_set=split_train_test(housing,0.2) #0.2 is ratio in which we are dividing the dataset


# In[9]:


# Spliting dataset using Sklearn
from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)


# In[10]:


print(f"No. of Training samples Used : {len(train_set)} \nNo. of Testing samples Used  : {len(test_set)}")


# In[11]:


# Now it is possible for some features like CHAS here, that testing set include all value some case(1 here) while training set 
# contain none of them. In this way model get confused. So to avoid this situation, We use StratifiedShuffleSplit function.

from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, test_index in split.split(housing,housing['CHAS']):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]


# In[12]:


strat_test_set['CHAS'].value_counts()


# In[13]:


strat_train_set['CHAS'].value_counts()


# In[14]:


housing=strat_train_set.copy()


# ## Looking for CoRelations

# In[15]:


# Dataset collected may have wrong values which known as outliers to identify & deal with them we use Co-Relations
corr_matrix=housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)
# 1 or close to 1 represrtaion strong relation while close to -1 represent weak relation.


# In[16]:


from pandas.plotting import scatter_matrix
# Now we choose those attributes whose attributes are either strongly corelated either in +ve or -ve way.
attributes=["MEDV","RM","ZN","LSTAT"]
scatter_matrix(housing[attributes],figsize=(12,8))
plt.show()


# In[17]:


housing.plot(kind='scatter',x='RM',y='MEDV',alpha=0.8)
plt.show()


# ## Trying out Attribute Combinations

# In[18]:


housing['TAXRM']=housing['TAX']/housing['RM']
housing.head()


# In[19]:


corr_metrix=housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[20]:


housing.plot(kind="scatter",x="TAXRM",y="MEDV",alpha=0.8)
plt.show()


# In[21]:


housing=strat_train_set.drop("MEDV",axis=1)
housing_labels=strat_train_set["MEDV"].copy()


# ## Handling Missing Values

# In[22]:


# There are 3 Ways to handle Missing Values
# 1. Get Rid Of Missing Data Points
# 2. Get Rid of whole Attribute
# 3. Set values to (median, mode , mean or 0) as per requirement


# In[23]:


#a=housing.dropna(subset=["RM"])             #Option 1
#a.shape


# In[24]:


# housing.drop("RM",axis=1)   #Option 2


# In[25]:


housing.describe() #Before We started filling values


# In[26]:


median=housing["RM"].median() #Option 3
housing["RM"].fillna(median)


# In[27]:


housing.shape


# In[28]:


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy="median")
imputer.fit(housing)


# In[29]:


imputer.statistics_


# In[30]:


X=imputer.transform(housing)


# In[31]:


housing_tr=pd.DataFrame(X,columns=housing.columns)


# In[32]:


housing_tr.describe()


# ## Sciket Learn Design

# In[33]:


# Primary there are 3 types of people:

# 1. Estimators    : It estimates some parameters based on datasets. eg. SIMPLEIMPUTER. 
                #    Fit Method    : Fits the dataset & calculate internal parameter.

# 2. Transformers  : Transform Method takes input & return output based on learning from fit(). It also has convience function 
                # called fit_transform() which fits & then transforms.

# 3. Predictor     : LinearRegression model is example of predictor. Fit() & predict() are two common function. It also give 
                    # score() function which will evaluate the predictions.

        


# # Feature Scaling

# Primarily It is considuted of 2 types :
#     
#     1. Min-Max Scaling : It is known as normalization. Sklearn provide MinMaxScaler class for this.
#                        = (value-min)/(max-min)
#                        
#     2. Standardization : Sklean provides StandardScaler class for this.
#                        = (value-mean)/std

# # Creating Pipeline

# In[34]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

my_pipeline=Pipeline([
        ('imputer',SimpleImputer(strategy="median")),
        # ........................... add as many codelines as you want
        ('std_scaler',StandardScaler()),
])


# In[35]:


housing_num_tr=my_pipeline.fit_transform(housing_tr)
housing_num_tr.shape


# # Selecting Desired Model for Predictions

# In[36]:


# In this Phase We will choose a model numerous models & check which one fits well

from sklearn.linear_model import LinearRegression
model1=LinearRegression()
model1.fit(housing_num_tr,housing_labels)

from sklearn.tree import DecisionTreeRegressor
model2=DecisionTreeRegressor()
model2.fit(housing_num_tr,housing_labels)

from sklearn.ensemble import RandomForestRegressor
model3=RandomForestRegressor()
model3.fit(housing_num_tr,housing_labels)


# In[37]:


some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]
prepared_data=my_pipeline.transform(some_data)
model1.predict(prepared_data)
model2.predict(prepared_data)
model3.predict(prepared_data)


# In[38]:


list(some_labels)


# # Evaluating The Matrix

# In[39]:


from sklearn.metrics import mean_squared_error
import numpy as np

# for linear regresser
housing_predictions=model1.predict(housing_num_tr)
lin_mse=mean_squared_error(housing_labels,housing_predictions)
lin_rmse=np.sqrt(lin_mse)
print(lin_rmse)

# For Decision Tree Regressor
housing_predictions=model2.predict(housing_num_tr)
dt_mse=mean_squared_error(housing_labels,housing_predictions)
dt_rmse=np.sqrt(dt_mse)
print(dt_rmse)

# For Random Forest Regressor
housing_predictions=model3.predict(housing_num_tr)
rf_mse=mean_squared_error(housing_labels,housing_predictions)
rf_rmse=np.sqrt(rf_mse)
print(rf_rmse)


# # Using Better Evaluation Technique (Cross-Validation)

# In cross validation training dataset is further divided into sub-parts. One part is done for training & other is used for testing. Basically we training dataset into further test & train set.

# In[40]:


from sklearn.model_selection import cross_val_score

# For Linear Regressor
scores1=cross_val_score(model1,housing_num_tr,housing_labels,scoring="neg_mean_squared_error")
rmse_scores1=np.sqrt(-scores1)

# For Decision Tree Regressor
scores2=cross_val_score(model2,housing_num_tr,housing_labels,scoring="neg_mean_squared_error")
rmse_scores2=np.sqrt(-scores2)

# For Random Forset Regressor
scores3=cross_val_score(model3,housing_num_tr,housing_labels,scoring="neg_mean_squared_error")
rmse_scores3=np.sqrt(-scores3)


# In[41]:


print(rmse_scores1)
print(rmse_scores2)
print(rmse_scores3)


# In[42]:


def print_scores(score):
    print(f"Score              : {score}")
    print(f"Mean               : {score.mean()}")
    print(f"Standard Deviation : {score.std()}")


# In[43]:


print("Linear Regression Model Results")
print(print_scores(rmse_scores1))
print()
print("Decision Regression Model Results")
print(print_scores(rmse_scores2))
print()
print("Random Forest Regression Model Results")
print(print_scores(rmse_scores3))
print()


# From results of above all models Random Forest Regressor Has Lowest Error Rates. Therefore It will be selected for further predictions.

# # Saving The Model

# In[44]:


from joblib import dump,load
dump(model3,'FinalPredictor.joblib')


# In[46]:


print(prepared_data[0])


# In[ ]:




