#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np


# In[7]:


df = pd.read_csv(r"C:\Users\samee\OneDrive\Desktop\azzu\sales.csv")


# In[ ]:


df.info()


# In[11]:


df.head()


# In[13]:


print(df.isnull().sum())


# In[15]:


print(df.dtypes)


# In[17]:


print(f"Number of duplicate rows: {df.duplicated().sum()}")


# In[19]:


missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])


# In[21]:


df['promotion_type'] = df['promotion_type'].fillna('None')


# In[23]:


df['date'] = pd.to_datetime(df['date'])


# In[25]:


numeric_cols = ['units_sold', 'unit_price', 'revenue', 'discount', 'final_price', 
                'marketing_spend', 'customer_rating', 'return_rate', 'stock_available',
                'competitor_price', 'delivery_time_days', 'platform_fee', 'profit_margin',
                'sentiment_score', 'customer_age']


# In[27]:


for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# In[29]:


def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers


# In[31]:


outlier_columns = ['units_sold', 'unit_price', 'revenue', 'final_price']
for col in outlier_columns:
    print(f"Outliers in {col}:")
    print(detect_outliers(df, col).shape[0])


# In[33]:


calculated_final = df['unit_price'] * (1 - df['discount']/100)
inconsistent = np.abs(calculated_final - df['final_price']) > 0.01


# In[35]:


print(f"Inconsistent final prices: {inconsistent.sum()}")


# In[37]:


calculated_revenue = df['units_sold'] * df['final_price']
inconsistent_revenue = np.abs(calculated_revenue - df['revenue']) > 0.01
print(f"Inconsistent revenue calculations: {inconsistent_revenue.sum()}")


# In[39]:


text_columns = ['platform', 'product_name', 'category', 'sub_category', 'region', 
                'promotion_type', 'customer_name', 'customer_region']


# In[41]:


for col in text_columns:
    df[col] = df[col].str.strip()
    df[col] = df[col].str.title()


# In[43]:


df['mobile_number'] = df['mobile_number'].astype(str)
invalid_mobile = df['mobile_number'].str.len() != 10
print(f"Invalid mobile numbers: {invalid_mobile.sum()}")


# In[45]:


print(df.info())


# In[47]:


print(df.isnull().sum())


# In[49]:


df.to_csv('cleaned_sales.csv', index=False)


# In[ ]:




