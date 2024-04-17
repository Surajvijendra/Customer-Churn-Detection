#!/usr/bin/env python
# coding: utf-8

# # Customer Churn detection

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score,precision_score,f1_score,fbeta_score
from sklearn.feature_selection import SelectKBest,chi2
from imblearn.combine import SMOTETomek
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from mlxtend.plotting import plot_decision_regions


# ## importing the data

# In[2]:


churn_data = pd.read_csv(r"E:\Data science\Afame tech projecct\Churn_Modelling.csv")
churn_data


# ## Data cleaning

# In[3]:


churn_data[['CustomerId']].value_counts()


# In[4]:


churn_data.info()


# In[5]:


churn_data.isnull().sum()


# RowNumber, CustomerId , Surname doesn't make the much impact on the target variable so droping it.

# In[6]:


churn_data.drop(['RowNumber','CustomerId','Surname'],axis=1,inplace=True)


# In[7]:


churn_data


# ## EDA

# In[8]:


geo_fre = churn_data[['Geography']].value_counts()
geo_fre


# **Number of people over different region in the data set**

# In[9]:


plt.figure(figsize=(4, 4))
plt.pie(geo_fre, labels= geo_fre.index, autopct='%1.1f%%')
plt.legend()
plt.title('Ditribution over region')
plt.show()


# * 50.1% of people from France
# * 25.1% of people from Germany
# * 24.8% of people from Spain

# In[10]:


gend_freq = churn_data[['Gender']].value_counts()
gend_freq


# **Gender distrubution in the data set**

# In[11]:


plt.figure(figsize=(4, 4))
plt.pie(gend_freq, labels= gend_freq.index, autopct='%1.1f%%')
plt.legend()
plt.title('Distribution by Gender')
plt.show()


# * 54.6% are Male 
# * 45.4% are Female

# In[12]:


plt.figure(figsize=(8, 6))
sns.histplot(data=churn_data, x='Age', bins=10)
plt.title('Age Frequency')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()


# * The most of the people in the data are the age between 32 - 40 i.e 3600 people.
# * Around 2300 people are the age between 25 - 32.
# * Around 1800 people are having the age between 40 - 48.

# In[13]:


churn_numbers = churn_data['Exited'].value_counts()
churn_numbers 


# **Churn Rate**

# In[14]:


plt.figure(figsize=(4, 4))
plt.pie(churn_numbers, labels= churn_numbers.index, autopct='%1.1f%%')
plt.title('Churn Rate')
plt.legend()
plt.show()


# * 20.4% of the people exited from the subscription.

# In[15]:


gender_rate = churn_data.groupby('Gender')['Exited'].sum()
gender_rate


# **Churn Rate by Gender**

# In[16]:


plt.figure(figsize=(6, 14))
plt.pie(gender_rate, labels= gender_rate.index, autopct='%1.1f%%')
plt.title('Churn Rate by Gender')
plt.legend()
plt.show()


# * 55.9% of the exited customers are Female.
# * 44.1% are Male Customers.

# In[17]:


churn_number_geo_gend = churn_data.groupby(['Geography', 'Gender'])['Exited'].sum()
churn_number_geo_gend


# **Churn Rate by Gender over different Region**

# In[18]:


plt.figure(figsize=(6, 14))
plt.pie(churn_number_geo_gend, labels= churn_number_geo_gend.index, autopct='%1.1f%%')
plt.title('Churn Rate by Gender over different Region')
plt.legend()
plt.show()


# **Churn Rate based on the Average Balance less the 50%**

# In[19]:


churn_data.groupby(churn_data['Balance'] < 97198)['Exited'].sum()


# In[20]:


plt.figure(figsize=(6, 14))
plt.pie(churn_data.groupby(churn_data['Balance'] < churn_data['Balance'].mean())['Exited'].sum(), labels= churn_data.groupby(churn_data['Balance'] < 97198)['Exited'].sum().index, autopct='%1.1f%%')
plt.title('Churn Rate based on the Average Balance less the 50%')
plt.legend()
plt.show()


# 30% of the people exited maybe because of low balence

# **Churn Rate based on the Average salary less the 50%**

# In[21]:


churn_data.groupby(churn_data['EstimatedSalary'] < churn_data['EstimatedSalary'].mean())['Exited'].sum()


# In[22]:


plt.figure(figsize=(6, 14))
plt.pie(churn_data.groupby(churn_data['EstimatedSalary'] < churn_data['EstimatedSalary'].mean())['Exited'].sum(), labels= churn_data.groupby(churn_data['Balance'] < 97198)['Exited'].sum().index, autopct='%1.1f%%')
plt.title('Churn Rate based on the Average salary less the 50%')
plt.legend()
plt.show()


# In[23]:


churn_data.describe()


# #### converting non numeric data into numeric data

# In[24]:


ohe = OneHotEncoder(sparse_output = False)


# In[25]:


gender = ohe.fit_transform(churn_data[['Gender']])
gender


# In[26]:


gender_ = pd.Series(gender.argmax(axis=1))


# In[27]:


churn_data['Gender'] = gender_


# In[28]:


geography = ohe.fit_transform(churn_data[['Geography']])
geography


# In[29]:


geography_ = pd.Series(geography.argmax(axis=1))


# In[30]:


churn_data['Geography'] = geography_


# In[31]:


columns_ = churn_data.columns
columns_


# In[32]:


churn_data.describe()


# In[ ]:





# In[33]:


churn_data


# ### checking the skewness of the data

# In[34]:


churn_data.skew()


# In[35]:


for column in churn_data.columns:
    sns.distplot(churn_data[column], kde=True)
    plt.show()


# ### Outliers Detection

# In[36]:


for col_1 in churn_data.columns:
    plt.figure(figsize=(4, 2))
    sns.boxplot(x=churn_data[col_1])
    plt.title(f'Boxplot of {col_1}')
    plt.show()


# ## Checking the corellation between the data

# In[37]:


sns.pairplot(churn_data)
plt.show()


# In[38]:


plt.figure(figsize=(18, 6))
sns.heatmap(churn_data.corr(),annot=True)
plt.show()


# ### Data Splitting

# In[39]:


x = churn_data.drop('Exited',axis=1)
x


# In[40]:


y = churn_data[['Exited']]
y


# In[41]:


y.value_counts()


# Data containes 7963 members doesn't churned
# 
# 2037 members churned so it's clearly imbalenced data

# **Over Sampling the Data**

# In[42]:


st =SMOTETomek(random_state=42)


# In[43]:


x_over,y_over = st.fit_resample(x,y)


# In[44]:


y_over.value_counts()


# Now the data is balenced so it deos not affects the results

# **Scaling the data**

# In[45]:


scalar = StandardScaler()


# In[46]:


x_scaled = scalar.fit_transform(x_over)
x_scaled = pd.DataFrame(x_scaled,columns = x.columns)
x_scaled


# In[47]:


x_scaled.skew()


# # Modeling

# In[48]:


x_train,x_test,y_train,y_test = train_test_split(x_scaled,y_over,test_size=.2,random_state=20)


# **Logistic Regression**

# In[49]:


Logistic = LogisticRegression()


# In[50]:


Logistic.fit(x_train,y_train)


# In[51]:


predicted_log = Logistic.predict(x_test)


# In[52]:


accuracy_score(predicted_log,y_test)


# In[53]:


precision_score(predicted_log,y_test)


# In[54]:


precision_score(Logistic.predict(x_train),y_train)


# In[55]:


recall_score(predicted_log,y_test)


# In[56]:


recall_score(Logistic.predict(x_train),y_train)


# In[57]:


fbeta_score(predicted_log,y_test,beta=1)


# In[58]:


fbeta_score(Logistic.predict(x_train),y_train,beta=1)


# **Getting bit low accurecy 77% but the model was generalized**

# **Random Forest**

# In[59]:


ranndom = RandomForestClassifier(n_estimators=200,max_depth=8,max_samples=2000,random_state=42)


# In[60]:


ranndom.fit(x_train,y_train)


# In[61]:


pred = ranndom.predict(x_test)


# In[62]:


accuracy_score(pred,y_test)


# In[63]:


precision_score(pred,y_test)


# In[64]:


precision_score(ranndom.predict(x_train),y_train)


# In[65]:


recall_score(ranndom.predict(x_train),y_train)


# In[66]:


recall_score(pred,y_test)


# In[67]:


fbeta_score(ranndom.predict(x_train),y_train,beta=1)


# In[68]:


fbeta_score(pred,y_test,beta=1)


# **model was slightly over fitted the data but stills it's good model with 84% of fbeta score**
# * Recall rate is 84%
# * Precision score is 85%

# **Cross Validation**

# In[69]:


cros_val_score = cross_val_score(ranndom, x_train, y_train, cv=10, scoring='accuracy')
cros_val_score.mean()


# **Gradient boosting**

# In[70]:


grad_boost = GradientBoostingClassifier(learning_rate=.1,n_estimators=90)


# In[71]:


grad_boost.fit(x_train,y_train)


# In[72]:


precision_score(grad_boost.predict(x_test),y_test)


# In[73]:


precision_score(grad_boost.predict(x_train),y_train)


# In[74]:


recall_score(grad_boost.predict(x_test),y_test)


# In[75]:


recall_score(grad_boost.predict(x_train),y_train)


# In[76]:


fbeta_score(grad_boost.predict(x_test),y_test,beta=1)


# In[77]:


fbeta_score(grad_boost.predict(x_train),y_train,beta=1)


# **model was well generalized**
# * Recall score 85% for churn rate detection it will be good it detected 85% of Flase -ve rate
# * Precision score also 84.6% still good

# In[78]:


cros_val_score = cross_val_score(grad_boost, x_train, y_train, cv=10, scoring='accuracy')
cros_val_score.mean()


# In[ ]:





# **XGBOOST**

# In[79]:


xg = xgb.XGBClassifier()


# In[80]:


xg.fit(x_train,y_train)


# In[81]:


xg_pred = xg.predict(x_test)


# In[82]:


accuracy_score(xg_pred,y_test)


# In[83]:


precision_score(xg_pred,y_test)


# In[84]:


recall_score(xg_pred,y_test)


# In[85]:


fbeta_score(xg_pred,y_test,beta=1)


# In[86]:


fbeta_score(xg.predict(x_train),y_train,beta=1)


# In[87]:


accuracy_score(xg.predict(x_train),y_train)


# In[88]:


precision_score(xg.predict(x_train),y_train)


# In[89]:


recall_score(xg.predict(x_train),y_train)


# the model was ove fitted

# **SVM**

# In[90]:


sv = SVC()


# In[91]:


sv.fit(x_train,y_train)


# In[92]:


sv_pred = sv.predict(x_test)


# In[93]:


accuracy_score(sv_pred,y_test)


# In[94]:


fbeta_score(sv_pred,y_test,beta=1)


# In[95]:


precision_score(sv_pred,y_test)


# In[96]:


recall_score(sv_pred,y_test)


# **Decision Tree**

# In[97]:


tre = DecisionTreeClassifier(max_depth=15,max_leaf_nodes=150,random_state=42)


# In[98]:


tre.fit(x_train,y_train)


# In[99]:


tre_predict = tre.predict(x_test)


# In[100]:


accuracy_score(tre_predict,y_test)


# In[101]:


precision_score(tre_predict,y_test)


# In[102]:


precision_score(tre.predict(x_train),y_train)


# In[103]:


recall_score(tre_predict,y_test)


# In[104]:


recall_score(tre.predict(x_train),y_train)


# In[105]:


fbeta_score(tre_predict,y_test,beta=1)


# In[106]:


fbeta_score(tre.predict(x_train),y_train,beta=1)


# **converting the data into normal distribution**

# In[107]:


from sklearn.preprocessing import PowerTransformer


# In[108]:


x_scaled


# In[109]:


y_over


# In[110]:


x_scaled.skew()


# In[111]:


cred = np.absolute(x_scaled[['CreditScore']])
cred.skew()


# In[112]:


box =  PowerTransformer(method='yeo-johnson')


# In[113]:


creditScore = box.fit_transform(cred)
creditScore = pd.DataFrame(creditScore,columns=cred.columns)
creditScore.skew()


# In[114]:


sns.distplot(creditScore,kde=True)
plt.show()


# In[115]:


geogr = np.absolute(x_scaled[['Geography']])
geogr.skew()


# In[116]:


geography = box.fit_transform(geogr)
geography = pd.DataFrame(geography,columns=x_scaled.columns[[1]])
geography.skew()


# In[117]:


sns.distplot(geography,kde=True)
plt.show()


# In[118]:


age = np.absolute(x_scaled[['Age']])
age.skew()


# In[119]:


Age = box.fit_transform(age)
Age = pd.DataFrame(Age,columns=x_scaled.columns[[3]])
Age.isnull().sum()


# In[120]:


sns.distplot(Age,kde=True)
plt.show()


# In[121]:


balnc = np.square(x_scaled[['Balance']])
balnc.skew()


# In[122]:


Balance = box.fit_transform(balnc)
Balance = pd.DataFrame(Balance,columns=x_scaled.columns[[5]])
Balance.skew()


# In[123]:


sns.distplot(Balance,kde=True)
plt.show()


# In[124]:


num_of = np.sin(x_scaled[['NumOfProducts']])
num_of.skew()


# In[125]:


NumOfProducts = box.fit_transform(num_of)
NumOfProducts = pd.DataFrame(NumOfProducts,columns=x_scaled.columns[[6]])
NumOfProducts.skew()


# In[126]:


sns.distplot(NumOfProducts,kde=True)
plt.show()


# In[127]:


Age.isnull().sum()


# In[128]:


card = np.sin(x_scaled[['IsActiveMember']])
card.skew()


# In[129]:


x_scaled.columns


# In[130]:


new_data = pd.concat([creditScore,geography,x_scaled[['Gender']],Age,x_scaled[['Tenure']],Balance,x_scaled[['NumOfProducts']],x_scaled[['HasCrCard']],x_scaled[['IsActiveMember']],x_scaled[['EstimatedSalary']],y_over],axis=1)
new_data


# In[131]:


new_data.isnull().sum()


# **scling the new modified data**

# In[132]:


scaled_new_data = scalar.fit_transform(new_data)
scaled_new_data = pd.DataFrame(scaled_new_data,columns = new_data.columns)


# In[133]:


scaled_new_data.skew()


# **Removing the outliers from the data**

# In[134]:


from scipy.stats import zscore


# In[135]:


z_scores = zscore(new_data)


# In[136]:


abs_z_scores = abs(z_scores)
threshold = 3


# In[137]:


outlier_rows = (abs_z_scores > threshold).any(axis=1)
outliers = scaled_new_data[outlier_rows]
outliers


# In[138]:


clean_data = scaled_new_data.drop(churn_data[outlier_rows].index)


# In[139]:


clean_data


# In[140]:


e = clean_data.drop('Exited',axis=1)
e


# In[141]:


e_scaled = scalar.fit_transform(e)
e_scaled = pd.DataFrame(e_scaled,columns=e.columns)
e_scaled


# In[142]:


f = clean_data[['Exited']]
f


# In[143]:


e_over,f_over = st.fit_resample(e,f)
e_over


# In[144]:


f_over


# In[145]:


e_scaled =scalar.fit_transform(e_over)
e_scaled


# In[146]:


e_scaled = pd.DataFrame(e_scaled,columns=e.columns)
e_scaled


# In[147]:


f_over


# In[148]:


c_train,c_test,d_train,d_test = train_test_split(e_scaled,f_over,test_size=.2,random_state=42)


# In[149]:


ranndom_ = RandomForestClassifier(n_estimators=150,max_depth=10,max_samples=.35,random_state=42)


# In[150]:


ranndom_.fit(c_train,d_train)


# In[151]:


fbeta_score(ranndom_.predict(c_test),d_test,beta=1)


# In[152]:


precision_score(ranndom_.predict(c_test),d_test)


# In[153]:


recall_score(ranndom_.predict(c_test),d_test)


# In[154]:


fbeta_score(ranndom_.predict(c_train),d_train,beta = 1)


# After removel of outliers getting bit low score

# In[155]:


print(f'Random forest score {fbeta_score(pred,y_test,beta=1)}')
print(f'Gboost score {fbeta_score(grad_boost.predict(x_test),y_test,beta=.1)}')


# over all Gradient boosting getting the best results and it's well generalised model

# In[ ]:





# In[ ]:





# In[157]:


import tensorflow as tf


# In[158]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten


# In[169]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(x_train.shape[1],)), # First layer with input shape specified
    tf.keras.layers.Dense(32, activation='relu'), # Subsequent layers without input shape specified
    tf.keras.layers.Dense(10, activation='softmax')
])


# In[170]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[171]:


model.fit(x_train, y_train, epochs=5)


# In[172]:


test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")


# In[ ]:




