#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[4]:


# Load and sort data
df = pd.read_csv('cleaned_sales.csv', parse_dates=['date'])
df = df.sort_values('date')


# In[6]:


# Lag Features 
df['lag_units_sold_1'] = df['units_sold'].shift(1)
df['lag_units_sold_7'] = df['units_sold'].shift(7)
df['lag_revenue_1'] = df['revenue'].shift(1)


# In[8]:


# Rolling Features
df['rolling_units_7'] = df['units_sold'].rolling(window=7).mean()
df['rolling_units_30'] = df['units_sold'].rolling(window=30).mean()
df['rolling_revenue_7'] = df['revenue'].rolling(window=7).mean()


# In[10]:


# Date Features
df['day'] = df['date'].dt.day
df['week'] = df['date'].dt.isocalendar().week
df['quarter'] = df['date'].dt.quarter
df['is_weekend'] = df['date'].dt.weekday >= 5
df['is_month_end'] = df['date'].dt.is_month_end


# In[12]:


#Promotion Features
df['has_discount'] = df['discount'] > 0
df['discount_percent'] = (df['unit_price'] - df['final_price']) / df['unit_price']
df['is_promotion'] = df['promotion_type'].notnull()


# In[14]:


# Competitor Influence
df['price_gap'] = df['final_price'] - df['competitor_price']
df['price_advantage'] = df['price_gap'] < 0


# In[16]:


# Customer Sentiment
df['adjusted_sentiment'] = df['sentiment_score'] * df['customer_rating']
df['high_rating'] = df['customer_rating'] >= 4


# In[18]:


# Return Behavior 
df['net_units_sold'] = df['units_sold'] * (1 - df['return_rate'])


# In[20]:


# Profit Metrics
df['profit'] = df['revenue'] * df['profit_margin']


# In[22]:


# Demand Momentum
df['sales_diff'] = df['units_sold'].diff()


# In[24]:


# Regional Match
df['same_region'] = df['region'] == df['customer_region']


# In[26]:


# Handle NaNs 
df = df.dropna().reset_index(drop=True)
df.head()


# In[ ]:




