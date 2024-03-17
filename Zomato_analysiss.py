#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


import pandas as pd


# In[4]:


import sqlite3


# In[5]:


#how to read data if it is available in some database file


# In[6]:


con = sqlite3.connect(r"C:\Users\lata6\Downloads\zomato_rawdata.sqlite")


# In[7]:


pd.read_sql_query("SELECT * FROM Users", con).head(2)


# In[8]:


df = pd.read_sql_query("SELECT * FROM Users", con)


# In[9]:


df.shape


# In[10]:


df.columns


# In[ ]:





# In[11]:


#how to deal with missing value 


# In[12]:


#data cleaning 
df.head(6) #->6 values


# In[13]:


df.isnull().sum() #every value is null or not 
#rate, phone, location, rest_type, dish_liked, cuisines, approx_cost contains missing values


# In[14]:


df.isnull().sum()/len(df)*100
#percentage of missing value


# In[15]:


df['rate'].unique()


# In[16]:


df['rate'].replace(('NEW', '-'), np.nan, inplace= True)
#denote my missing value


# In[17]:


df['rate'].unique()


# In[18]:


df['rate']= df['rate'].apply(lambda x : float(x.split('/')[0]) if type(x)==str else x)


# In[ ]:





# In[19]:


#is there a relation between online order option and rating of the restaurant


# In[20]:


#achieving pivot table using pandas -> 


# In[21]:


x= pd.crosstab(df['rate'], df['online_order'])


# In[22]:


x


# In[23]:


x.plot(kind= 'bar', stacked= True)
#stacked bar chart


# In[24]:


# 100% stacked bar chart
x.sum(axis=1).astype(float)


# In[25]:


normalize_df= x.div(x.sum(axis=1).astype(float), axis=0)


# In[26]:


(normalize_df*100).plot(kind= 'bar', stacked= True)


# In[ ]:





# In[27]:


#data cleaning to perform text
#TEXT ANALYSIS


# In[28]:


df['rest_type'].isnull().sum()


# In[29]:


data = df.dropna(subset=['rest_type'])


# In[30]:


data['rest_type'].isnull().sum()


# In[31]:


quick_bites_df= data[data['rest_type'].str.contains('Quick Bites')]


# In[32]:


quick_bites_df.shape


# In[33]:


quick_bites_df.columns


# In[34]:


quick_bites_df['reviews_list']


# In[35]:


quick_bites_df['reviews_list'] = quick_bites_df['reviews_list'].apply(lambda x:x.lower())


# In[36]:


#removing special characters 


# In[37]:


#regular expression in python 
#tokenize 


# In[38]:


from nltk.corpus import RegexpTokenizer


# In[39]:


Tokenizer = RegexpTokenizer("[a-zA-Z]+")


# In[40]:


Tokenizer.tokenize(quick_bites_df['reviews_list'][3])


# In[41]:


sample= data[0:10000]


# In[42]:


reviews_tokens = sample['reviews_list'].apply(Tokenizer.tokenize)


# In[43]:


#unigram analysis & removal of stopwords (is, am, are, they, etc)


# In[44]:


reviews_tokens


# In[45]:


import nltk
nltk.download('stopwords')


# In[46]:


from nltk.corpus import stopwords


# In[47]:


stop = stopwords.words('english')


# In[48]:


print(stop)


# In[49]:


#adding more words to the dictionary 
stop.extend(['rated', "n","nan","x","RATED","Rated"])


# In[50]:


print(stop)


# In[51]:


rev3 = reviews_tokens[3]
print(rev3)


# In[52]:


#anonymous function lambda
reviews_tokens_clean = reviews_tokens.apply(lambda each_review : [token for token in each_review if token not in stop])


# In[53]:


reviews_tokens_clean


# In[54]:


type(reviews_tokens_clean)


# In[55]:


#unigram analysis (pick one word and analyse that)


# In[56]:


total_reviews_2D = list(reviews_tokens_clean)


# In[57]:


total_reviews_1D = []

for review in total_reviews_2D:
    for word in review:
        total_reviews_1D.append(word)


# In[58]:


total_reviews_1D


# In[59]:


from nltk import FreqDist


# In[60]:


fd = FreqDist()


# In[61]:


for word in total_reviews_1D:
    fd[word] = fd[word] + 1


# In[62]:


fd.most_common(20)


# In[63]:


fd.plot(20)


# In[64]:


#bigram (collection of 2 words) or 
#trigram(collection of 3 words) analyis


# In[65]:


from nltk import FreqDist , bigrams , trigrams


# In[66]:


bi_grams = bigrams(total_reviews_1D)


# In[67]:


bi_grams


# In[ ]:





# In[68]:


fd_bigrams = FreqDist()

for bigram in bi_grams:
    fd_bigrams[bigram] = fd_bigrams[bigram] + 1


# In[69]:


fd_bigrams.most_common(20)


# In[70]:


fd_bigrams.plot(20)


# In[71]:


fd_bigrams.most_common(100)


# In[72]:


#trigram analysis for more meaningful analysis


# In[73]:


tri_grams = trigrams(total_reviews_1D)


# In[74]:


fd_trigrams = FreqDist()

for trigram in tri_grams:
    fd_trigrams[trigram] = fd_trigrams[trigram] + 1


# In[75]:


fd_trigrams.most_common(50)


# In[76]:


#extracting geographical coordinates from data


# In[77]:


df.head(5)


# In[78]:


get_ipython().system('pip install geocoder')
get_ipython().system('pip install geopy')


# In[79]:


df['location']


# In[80]:


len(df['location'].unique())


# In[81]:


df['location'].unique()


# In[82]:


df['location'] = df['location'] + ", Bangalore , Karnataka , India"


# In[83]:


df['location']


# In[84]:


df['location'].unique()


# In[85]:


df_copy = df.copy()


# In[86]:


df_copy['location'].isnull().sum()


# In[87]:


df_copy = df_copy.dropna(subset=['location'])


# In[88]:


df_copy['location'].isnull().sum()


# In[89]:


locations = pd.DataFrame(df_copy['location'].unique())


# In[90]:


locations.columns = ['name']


# In[91]:


locations


# In[92]:


from geopy.geocoders import Nominatim


# In[93]:


geolocator = Nominatim(user_agent = "app", timeout = None)


# In[94]:


lat = []
lon = []

for location in locations['name']:
    location = geolocator.geocode(location)
    if location is None: #what if the nominatim is unable to return a lon and lat
        lat.append(np.nan)
        lon.append(np.nan)
    else:
        lat.append(location.latitude)
        lon.append(location.longitude)


# In[95]:


locations['latitude'] = lat
locations['longitude'] = lon


# In[96]:


locations


# In[ ]:





# In[97]:


#geographical heat map
#we need the count feature in the new data frame


# In[98]:


locations.isnull().sum()


# In[99]:


locations[locations['latitude'].isna()]


# In[100]:


import warnings 
from warnings import filterwarnings
filterwarnings('ignore')


# In[101]:


locations['latitude'][79] = 13.0085
locations['longitude'][79] = 77.6737


# In[102]:


locations[locations['latitude'].isna()]


# In[103]:


locations['latitude'][85] = 18.404379
locations['longitude'][85] = 78.269214


# In[104]:


locations[locations['latitude'].isna()]


# In[105]:


#building a geographical heat map for north indian restaurant 


# In[106]:


df['cuisines'].isnull().sum()


# In[107]:


df = df.dropna(subset=['cuisines'])


# In[ ]:





# In[108]:


north_india = df[df['cuisines'].str.contains('North Indian')]


# In[109]:


north_india.shape


# In[110]:


north_india.head(2)


# In[111]:


north_india['location']


# In[112]:


north_india['location'].value_counts()


# In[113]:


north_india_reset_count = north_india['location'].value_counts().reset_index().rename(columns={'location' : 'name'})


# In[114]:


north_india_rest_count


# In[115]:


locations


# In[116]:


heatmap_df = north_india_rest_count.merge(locations, on = 'name', how = 'left')


# In[117]:


heatmap_df


# In[118]:


#creating a base map


# In[119]:


get_ipython().system('pip install folium')


# In[120]:


import folium


# In[121]:


basemap = folium.Map()


# In[122]:


basemap


# In[123]:


get_ipython().system('pip install Heatmap')


# In[124]:


heatmap_df.columns


# In[125]:


from folium.plugins import HeatMap


# In[126]:


HeatMap(heatmap_df[['latitude', 'longitude', "count"]]).add_to(basemap) 


# In[127]:


basemap


# In[128]:


#how to automate your data analysis


# In[139]:


def get_heatmap(cuisine):
    cuisine_df = df[df['cuisines'].str.contains(cuisine)]
    
    cuisine_rest_count = cuisine_df['location'].value_counts().reset_index().rename(columns={'location' : 'name'})
    heatmap_df = cuisine_rest_count.merge(locations, on = 'name', how = 'left')
    print(heatmap_df.head(4))
    
    basemap = folium.Map()
    HeatMap(heatmap_df[['latitude', 'longitude', "count"]]).add_to(basemap) 
    return basemap


# In[140]:


get_heatmap('South Indian')


# In[142]:


df['cuisines'].unique()

