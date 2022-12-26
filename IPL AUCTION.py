#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style(style='darkgrid')


# In[2]:


ipl_auction=pd.read_csv('IPl Auction 2013-2022.csv')
ipl_auction.head(2)


# # Data Analysis

# In[102]:


df=ipl_auction.groupby(['year'])
ans=pd.DataFrame(columns=['year','total_spent','spent_on_batsman','spent_on_all_rounder','spent_on_bowler','spent_on_wicket_keeper'])
for g,d in df:
    ans=ans.append(pd.DataFrame([[g,
        d['sold_price'].sum(),
                   d.loc[d['type']=='Batsman',['sold_price']].sum().values[0],
                   
                   d.loc[d['type']=='All-Rounder',['sold_price']].sum().values[0],
                   d.loc[d['type']=='Bowler',['sold_price']].sum().values[0],
                   d.loc[d['type']=='Wicket Keeper',['sold_price']].sum().values[0]
                  ]],columns=['year','total_spent','spent_on_batsman','spent_on_all_rounder','spent_on_bowler','spent_on_wicket_keeper']
                ))


# In[106]:


ans['spent_on_batsman %']=ans['spent_on_batsman']/ans['total_spent']
ans['spent_on_bowler %']=ans['spent_on_bowler']/ans['total_spent']
ans['spent_on_all_rounder %']=ans['spent_on_all_rounder']/ans['total_spent']
ans['spent_on_wicket_keeper %']=ans['spent_on_wicket_keeper']/ans['total_spent']
ans.loc[:,['year','spent_on_batsman %','spent_on_bowler %','spent_on_all_rounder %','spent_on_wicket_keeper %']].set_index('year').plot(figsize=(16,8))


# In[107]:


# Now money spent on all types over the years
plt.figure(figsize=(16,8))
df=ipl_auction.groupby(['year','type']).agg({'sold_price':'sum'}).reset_index()
sns.lineplot(x='year',y='sold_price',data=df,hue='type')
plt.xticks(df['year'].unique())
plt.show()


# In[ ]:





# In[8]:


# The number of players from each country sold in an ipl auction
plt.pie(ipl_auction['nationality'].value_counts().values,autopct='%.2f%%')
plt.legend(ipl_auction['nationality'].value_counts().index)


# In[9]:


# 67 % players in the ipl are from India


# In[54]:


# each country type of players
df=ipl_auction.loc[:,['player_name','nationality','type']].groupby(['nationality','type']).agg({'player_name':'count'}).rename(columns={'player_name':'number_of_players'})
for i in ipl_auction['nationality'].dropna().unique():
    df.loc[i,'rank']=df.loc[i,'number_of_players'].rank(ascending=False).values
#     
df


# In[61]:


df1=df.loc[:,'number_of_players'].reset_index()
sns.barplot(x='nationality',y='number_of_players',data=df1,hue='type')


# In[33]:


ipl_auction['teams']=ipl_auction['teams'].replace('Delhi Daredevils','Delhi Capitals')
ipl_auction['teams']=ipl_auction['teams'].replace('Kings XI Punjab','Punjab Kings')


# In[34]:


# spending from each team over the years
ipl_auction.groupby(['teams']).agg({'sold_price':'sum'}).plot(kind='barh')


# In[51]:


# Punjab kings are big investors whereas CSK are less investors


# In[50]:


df=ipl_auction.groupby(['teams','year']).agg({'sold_price':'sum'}).reset_index()
plt.figure(figsize=(16,8))
sns.lineplot(x='teams',y='sold_price',data=df,hue='year')
plt.tight_layout()
plt.xticks(rotation=72)
plt.show()


# In[52]:


# Year wise money they spent in each years ipl season


# In[68]:


# Different sponsors of ipl over the years
ipl_auction.loc[:,['year','sponsored_by']].drop_duplicates().reset_index(drop='first')


# In[74]:


# Each year's highest buy
year=ipl_auction.groupby('year')
ans=pd.DataFrame(columns=['year','player_name','sold_price'])
for g,d in year:
    ans=ans.append(d.loc[d['sold_price']==d['sold_price'].max(),['year','player_name','type','sold_price']])


# In[75]:


ans


# In[79]:


# Total money spent on all type
ipl_auction.groupby(['type']).agg({'sold_price':'sum'}).plot(kind='barh')


# In[112]:


ipl_auction.groupby(['nationality']).agg({'sold_price':'mean'}).plot(kind='bar')


# In[ ]:





# In[142]:


def team_wise(team):
    ans=pd.DataFrame(columns=['year','total_batsman','total_all_rounder','total_bowler','total_wicket_keeper','total_bought',
                             'total_spent','total% spent_on batsman','total% spent_on all-rounder','total% spent_on bowler',
                              'total% spent_on wicket keeper'])
    t=ipl_auction.loc[ipl_auction['teams']==team,:]
    for g,d in t.groupby(['year']):
        ans=ans.append(pd.DataFrame([[
            g,
            d.loc[t['type']=='Batsman'].shape[0],
            d.loc[t['type']=='All-Rounder'].shape[0],
            d.loc[t['type']=='Bowler'].shape[0],
            d.loc[t['type']=='Wicket Keeper'].shape[0],
            d.shape[0],
            d['sold_price'].sum(),
            d.loc[t['type']=='Batsman','sold_price'].sum()/d['sold_price'].sum(),
            d.loc[t['type']=='All-Rounder','sold_price'].sum()/d['sold_price'].sum(),
            d.loc[t['type']=='Bowler','sold_price'].sum()/d['sold_price'].sum(),
            d.loc[t['type']=='Wicket Keeper','sold_price'].sum()/d['sold_price'].sum()
        ]],columns=['year','total_batsman','total_all_rounder','total_bowler','total_wicket_keeper','total_bought',
                             'total_spent','total% spent_on batsman','total% spent_on all-rounder','total% spent_on bowler',
                              'total% spent_on wicket keeper']),ignore_index=True)
    return ans


# In[151]:


ans=team_wise('Chennai Super Kings')


# In[152]:


ans


# # Model building

# In[158]:


ipl_auction.drop(columns=['player_name'],inplace=True)
X=ipl_auction.drop(columns='sold_price')
y=ipl_auction['sold_price']


# In[159]:


X.isna().sum()


# In[160]:


X.fillna(value='unknown',inplace=True)


# In[161]:


X.dtypes


# In[162]:


from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(drop='first',dtype=int)


# In[169]:


cat_col=X.dtypes=='O'
cat_col=X.dtypes[cat_col].index


# In[183]:


X[list(range(19))]=ohe.fit_transform(X[cat_col]).toarray()


# In[185]:


X.drop(columns=cat_col,inplace=True)


# In[190]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train_sc=sc.fit_transform(X_train,y_train)
X_test_sc=sc.transform(X_test)


# In[191]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
from sklearn.metrics import mean_squared_error


# In[192]:


lr.fit(X_train_sc,y_train)


# In[193]:


np.sqrt(mean_squared_error(y_test,lr.predict(X_test_sc)))


# In[203]:


ans=pd.DataFrame(lr.predict(X_test_sc),columns=['prediction'])
ans['actual']=y_test.values
ans


# In[ ]:




