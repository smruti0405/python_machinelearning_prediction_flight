#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


get_ipython().system('pip install matplotlib')


# In[4]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[5]:


train_data = pd.read_excel(r"C:\Users\ACER\Desktop\flight_price_dataset/Data_Train.xlsx")


# In[7]:


train_data.info()


# In[8]:


train_data.isnull()


# In[9]:


train_data[train_data['Route'].isnull()]


# In[10]:


train_data.dropna(inplace=True)


# In[11]:


train_data.isnull().sum()


# In[12]:


train_data.info(memory_usage='deep')


# In[13]:


import warnings
warnings.filterwarnings('ignore')


# In[14]:


data = train_data.copy()


# In[15]:


def change_into_Datetime(col):
    data[col] = pd.to_datetime(data[col])


# In[16]:


for feature in ['Dep_Time', 'Arrival_Time', 'Date_of_Journey']:
    change_into_Datetime(feature)


# In[17]:


data.dtypes


# In[18]:


data['Journey_day'] = data['Date_of_Journey'].dt.day
data['Journey_month'] = data['Date_of_Journey'].dt.month
data['Journey_year'] = data['Date_of_Journey'].dt.year


# In[19]:


data.head(2)


# In[20]:


def extract_hour_min(df, col):
    df[col+ "_hour"] = df[col].dt.hour
    df[col+ "_minute"] = df[col].dt.minute
    return df.head(3)


# In[21]:


extract_hour_min (data , "Dep_Time")


# In[22]:


extract_hour_min (data , 'Arrival_Time')


# In[23]:


cols_to_drop =["Arrival_Time" , "Dep_Time"]
data.drop(cols_to_drop , axis = 1 ,inplace = True)


# In[24]:


data.head(2)


# In[26]:


data.shape


# In[27]:


def flight_dep_time(x):
    if(x>4) and (x<=8):
        return "Early morning"
    elif (x>8) and (x<=12):
        return "Morning"
    elif (x>12) and (x<=16):
        return "Noon"
    elif (x>16) and (x<=20):
        return "Evening"
    elif (x>20) and (x<=24):
        return "Night"
    
    else:
        return "Late night"


# In[28]:


data['Dep_Time_hour'].apply(flight_dep_time).value_counts()


# In[29]:


data["Dep_Time_hour"].apply(flight_dep_time).value_counts().plot()


# In[30]:


data["Dep_Time_hour"].apply(flight_dep_time).value_counts().plot(kind="bar",color = "yellow")


# In[31]:


import sys
print(sys.executable)


# In[34]:


get_ipython().system('pip install plotly')


# In[36]:


get_ipython().system('pip install cufflinks')


# In[37]:


import plotly
import cufflinks as cf
from cufflinks.offline import go_offline
from plotly.offline import plot, iplot, init_notebook_mode, download_plotlyjs
init_notebook_mode(connected=True)
cf.go_offline()


# In[38]:


data["Dep_Time_hour"].apply(flight_dep_time).value_counts()


# In[39]:


data["Dep_Time_hour"].apply(flight_dep_time).value_counts().iplot(kind="bar", color= 'Orange')


# In[40]:


def preprocess_duration(x):
    if 'h' not in x:
        x ='0h' + ' ' + x
    elif 'm' not in x:
       x = x + ' '+ '0m'
    return x


# In[41]:


data['Duration'].apply(preprocess_duration)


# In[42]:


data['Duration'] [0]


# In[43]:


'2h 50m' .split (' ')[0]


# In[44]:


int('2h 50m' .split (' ')[0][0:-1])


# In[45]:


int('2h 50m' .split (' ')[1][0:-1])


# In[46]:


data['Duration_hours'] = data['Duration'].apply(lambda x : int(x .split (' ')[0][0:-1]))


# In[47]:


data['Duration_mins'] = data['Duration'].apply(lambda x : int(x .split (' ')[-1][0:-1]))


# In[48]:


data.head(3)


# In[49]:


eval('2*60')


# In[50]:


data["Duration_total_mins"] =data['Duration'].str.replace('h',"*60").str.replace(' ', '+').str.replace('m',"*1").apply(eval)
data["Duration_total_mins"]


# In[51]:


sns.scatterplot(x = "Duration_total_mins", y = "Price" , data= data)


# In[52]:


sns.scatterplot(x = "Duration_total_mins", y = "Price" , hue = "Total_Stops", data= data)


# In[53]:


sns.lmplot(x = "Duration_total_mins", y = "Price" , hue = "Total_Stops", data= data)


# In[54]:


data['Airline'] == 'Jet Airways'


# In[55]:


data[data['Airline'] == 'Jet Airways']


# In[56]:


data[data['Airline'] == 'Jet Airways'].groupby('Route').count()


# In[57]:


data[data['Airline'] == 'IndiGo'].groupby('Route').size()


# In[58]:


data[data['Airline'] == 'Jet Airways'].groupby('Route').size().sort_values(ascending=False)


# In[59]:


sns.boxplot(y='Price' , x= 'Airline' , data = data.sort_values('Price', ascending= False))
plt.xticks(rotation = "vertical")
plt.show()


# In[60]:


data.head(3)


# In[61]:


cat_col = [col for col in data.columns if data[col].dtype == "object"]


# In[62]:


num_col = [col for col in data.columns if data[col].dtype != "object"]


# In[63]:


cat_col


# In[64]:


data["Source"].unique()


# In[65]:


data["Source"].apply(lambda x : 1 if x == 'Banglore' else 0)


# In[66]:


for sub_category in data["Source"].unique():
    data['Source_' +sub_category ] = data["Source"].apply(lambda x : 1 if x == sub_category else 0)


# In[67]:


data.head(3)


# In[68]:


cat_col


# In[69]:


data['Airline'].nunique()


# In[70]:


data.groupby(['Airline'])['Price'].mean().sort_values()


# In[71]:


airline= data.groupby(['Airline'])['Price'].mean().sort_values().index
airline


# In[72]:


{key: index for index , key in enumerate (airline, 0)}


# In[73]:


dict_airline = {key: index for index , key in enumerate (airline, 0)}


# In[74]:


data['Airline'] = data['Airline'].map(dict_airline)


# In[75]:


data['Airline']


# In[76]:


data.head(3)


# In[77]:


data['Destination'].unique()


# In[82]:


data['Destination'].replace('New Delhi', 'Delhi', inplace = True)


# In[83]:


data['Destination'].unique()


# In[84]:


dest= data.groupby(['Destination'])['Price'].mean().sort_values().index


# In[81]:


dest


# In[85]:


dict_dest = {key: index for index , key in enumerate (dest, 0)}


# In[86]:


dict_dest


# In[87]:


data['Destination'] = data['Destination'].map(dict_dest)


# In[88]:


data['Destination'] 


# In[89]:


data.head(4)


# In[90]:


data['Total_Stops']


# In[91]:


data['Total_Stops'].unique()


# In[92]:


stop = {'non-stop' : 0, '2 stops' :2, '1 stop' :1, '3 stops': 3, '4 stops' : 4}


# In[93]:


data['Total_Stops'] = data['Total_Stops'].map(stop)


# In[94]:


data['Total_Stops']


# In[95]:


data.head(2)


# In[96]:


data.columns


# In[97]:


data['Additional_Info'].value_counts()/len(data)*100


# In[98]:


data.head(3)


# In[99]:


data.drop(columns= ['Additional_Info', 'Date_of_Journey', 'Duration_total_mins' , 'Source'], axis = 1 , inplace= True)


# In[100]:


data['Journey_year'].unique()


# In[101]:


data.columns


# In[102]:


data.head(2)


# In[103]:


data.drop(columns= ['Route', 'Duration'], axis = 1, inplace = True)


# In[104]:


data.head(4)


# In[105]:


data.drop(columns= ['Journey_year'], axis = 1, inplace = True)


# In[106]:


data.head(4)


# In[107]:


def plot(df, col):
    fig,(ax1, ax2, ax3) = plt.subplots(3,1)
    
    sns.distplot(df[col], ax =ax1)
    sns.boxplot( x = df[col] , ax=ax2)
    sns.distplot(df[col], ax =ax3, kde=False)
    
plot (data, 'Price')


# In[108]:


q1 = data['Price'].quantile(0.25)
q3 = data['Price'].quantile(0.75)

iqr = q3-q1

maximum = q3 + 1.5*iqr
minimum = q1 - 1.5*iqr


# In[109]:


print(maximum)


# In[110]:


print(minimum)


# In[111]:


print ([price for price in data ['Price'] if price> maximum or price < minimum])


# In[112]:


len([price for price in data ['Price'] if price> maximum or price < minimum])


# In[113]:


np.where(data ['Price']>= 35000, data['Price'].median(), data ['Price'])


# In[114]:


data['Price'] = np.where(data ['Price']>= 35000, data['Price'].median(), data ['Price'])


# In[115]:


plot(data, 'Price')


# In[116]:


X = data.drop(['Price'], axis =1)


# In[117]:


y = data['Price']


# In[118]:


from sklearn.feature_selection import mutual_info_regression


# In[119]:


imp = mutual_info_regression(X , y)


# In[120]:


imp


# In[121]:


import pandas as pd
df = pd.DataFrame(data)


# In[122]:


pd.DataFrame(imp)


# In[123]:


pd.DataFrame(imp, index= X.columns)


# In[124]:


imp_df = pd.DataFrame(imp, index= X.columns)
imp_df


# In[125]:


imp_df.columns= ['importance']
imp_df


# In[126]:


imp_df.sort_values(by='importance', ascending = False)


# In[127]:


from sklearn.model_selection import train_test_split


# In[128]:


X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.25, random_state=42)


# In[129]:


from sklearn.ensemble import RandomForestRegressor


# In[130]:


ml_model = RandomForestRegressor ()


# In[131]:


ml_model.fit(X_train, y_train)


# In[132]:


y_pred = ml_model.predict(X_test)


# In[133]:


y_pred


# In[134]:


from sklearn import metrics


# In[135]:


metrics.r2_score(y_test, y_pred)


# In[136]:


import pickle


# In[137]:


file = open(r'C:\Users\ACER\Desktop\flight_price_dataset/rf_randompki', 'wb')


# In[138]:


pickle.dump(ml_model, file)


# In[139]:


model = open(r'C:\Users\ACER\Desktop\flight_price_dataset/rf_randompki', 'rb')


# In[140]:


forest = pickle.load(model)


# In[141]:


y_pred2 =forest.predict(X_test)


# In[142]:


metrics.r2_score(y_test, y_pred2)


# In[147]:


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np. mean(np.abs( y_true - y_pred ) /y_true) * 100


# In[148]:


mape(y_test, y_pred)


# In[149]:


from sklearn import metrics


# In[150]:


def predict(ml_model):
    model = ml_model.fit(X_train, y_train)
    print('Trainning score: {}'.format(model.score(X_train, y_train)))
    y_predection = model.predict(X_test)
    print('predictions are : {}'.format(y_predection))
    print('\n')
    r2_score= metrics.r2_score(y_test, y_predection)
    print('r2_score : {}'.format(r2_score))
    print('MAE : {}'.format(metrics.mean_absolute_error(y_test, y_predection)))
    print('MSE : {}'.format(metrics.mean_squared_error(y_test, y_predection)))
    print('RMSE : {}'.format(np.sqrt(metrics.mean_squared_error(y_test, y_predection))))
    print('MAPE : {}'.format(mape(y_test, y_predection)))
    sns.distplot(y_test - y_predection)
    


# In[151]:


predict(RandomForestRegressor())


# In[152]:


from sklearn.tree import DecisionTreeRegressor


# In[153]:


predict(DecisionTreeRegressor())


# In[154]:


RandomForestRegressor()


# In[155]:


from sklearn.model_selection import RandomizedSearchCV


# In[156]:


reg_rf = RandomForestRegressor()


# In[157]:


np.linspace(start= 100 , stop =1200, num= 6)


# In[158]:


n_estimators = [int(x) for x in np.linspace(start= 100 , stop =1200, num= 6)]
max_features = ["auto", "sqrt"]
max_depth = [int(x) for x in np.linspace(start= 5 , stop =30, num= 4)]
min_samples_split = [5,10,15,100]
                                         


# In[159]:


random_grid = {
    'n_estimators' : n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split
}


# In[160]:


random_grid


# In[162]:


rf_random = RandomizedSearchCV(estimator=reg_rf, param_distributions= random_grid, cv=3, n_jobs= -1 , verbose = 2)


# In[163]:


rf_random.fit(X_train, y_train)


# In[164]:


rf_random.best_params_


# In[165]:


rf_random.best_estimator_


# In[166]:


rf_random.best_score_


# In[ ]:




