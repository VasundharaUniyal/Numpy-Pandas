#!/usr/bin/env python
# coding: utf-8

# ## Python Numpy Arrays
# ## Part-1

# ## Numpy Tutorials
# NumPy is a general-purpose array-processing package. It provides a high-performance multidimensional array object, and tools for working with these arrays. It is the fundamental package for scientific computing with Python

# ## What is an array
# An array is a data structure that stores values of same data type. In Python, this is the main difference between arrays and lists. While python lists can contain values corresponding to different data types, arrays in python can only contain values corresponding to same data type

# In[1]:


#import the library
import numpy as np #np is an allice


# In[2]:


# Covert into array
lst=[1,2,3,4]
arr=np.array(lst) # single dimensional array (), 2-(()), 3-((()))


# In[3]:


type(arr) #ndarray- n dimensional array


# In[4]:


arr.shape


# In[5]:


lst1=[1,2,3,4,5]
lst2=[2,3,4,5,6]
lst3=[3,4,5,6,7]

arr1=np.array([lst1,lst2,lst3])


# In[6]:


arr1


# In[7]:


arr1.shape


# In[8]:


# indexing
arr[3]


# In[9]:


arr


# In[10]:


arr[3]=5


# In[103]:


arr


# In[104]:


arr[1:]


# In[105]:


arr[1:3]


# In[106]:


arr[1:4]


# In[107]:


arr[-1]


# In[108]:


arr[-5:-1]


# In[17]:


arr[:-1]


# In[18]:


arr[::-1]


# In[19]:


arr[::-2]


# In[20]:


arr[::-3]


# In[21]:


#Multi dimension
arr1[:,1]


# In[22]:


arr1


# In[23]:


arr1[:,3:]


# In[24]:


arr1[:,3:].shape


# In[25]:


arr1[1:,1:3]


# In[26]:


arr1[1:,3:]


# In[161]:


arr1[:,4:] # Assignment


# In[164]:


arr1[: , [0,4]] 


# In[142]:


arr1[:,-1]


# In[ ]:


#EDA


# In[110]:


arr


# In[111]:


arr<2


# In[112]:


#with indexing
arr[arr<2]


# In[113]:


arr1.shape


# In[114]:


# reshape array
arr1.reshape(5,3)


# In[116]:


# mechanism to create an array
np.arange(1,10,1).reshape(2,5)


# In[117]:


np.arange(1,10,2).reshape(1,10)


# In[118]:


arr*arr


# In[119]:


arr1*2


# In[120]:


np.ones((5,3))


# In[121]:


np.random.randint(10,50,4).reshape(2,2)


# In[122]:


# Based on normal distribution -means=0 sd=1
np.random.randn(5,6)


# In[11]:


np.random.sample(4,7)


# ## Python Pandas
# 
# In Part we are going to learn about
# 
# Pandas Dataframe
# Pandas Series
# Pandas Basic Operations

# In[ ]:


import pandas as pd
import numpy as np


# In[47]:


np.arange(0,20).reshape(5,4)


# In[48]:


# create dataframes 
df=pd.DataFrame(data=np.arange(0,20).reshape(5,4), index=['Row1','Row2','Row3',
                                                       'Row4','Row5'],columns=['Column1',
                                                                               'Column2','Column3','Column4'])


# In[49]:


df


# In[50]:


df.head()


# In[51]:


df.tail()


# In[52]:


type(df)


# In[53]:


df.info()


# In[54]:


df.describe()


# In[55]:


# indexing can be done in 2 ways
# directly by using column name, rowindex[loc]-location, rowindex number, columnindex number[.loc] - index location


# In[56]:


df.head()


# In[57]:


df[['Column1','Column2','Column3']] # to retreive 3 column


# In[58]:


# by using column names
df['Column1']


# In[59]:


type(df[['Column1','Column2','Column3']]) #dataframe - multiple row and multile columns


# In[60]:


type(df['Column1']) #series-either one row or one column


# In[61]:


df.loc['Row3']


# In[62]:


type(df.loc['Row3'])


# In[63]:


# by using row index name loc
df.loc[['Row3','Row4']]


# In[64]:


df.head()


# In[65]:


df.iloc[2:4,0:2]


# In[66]:


df.iloc[2:,1:]


# In[68]:


df.iloc[0:5,0:]


# In[69]:


# convert dataframes into arrays
df.iloc[:,1:].values


# In[70]:


# Basic operations
df.isnull().sum()


# In[71]:


df=pd.DataFrame(data=[[1,np.nan,2],[1,3,4]],index=['Row1','Row2'],columns=['Column1',
                                                                               'Column2','Column3'])


# In[72]:


df


# In[74]:


df=pd.DataFrame(data=[[1,np.nan,2],[1,3,4]],index=["Row1",
                                                     "Row2"],columns=["Column1",
                                                                            "Column2",
                                                                            "Column3",
                                                                            ])


# In[75]:


df


# In[76]:


df.isnull().sum()


# In[77]:


df[~df.isnull()]


# In[78]:


df.isnull().sum()==0


# In[79]:


df


# In[80]:


df['Column3'].value_counts()


# In[81]:


# to find unique values
df['Column2'].unique()


# In[82]:


df=pd.DataFrame(data=np.arange(0,20).reshape(5,4), index=['Row1','Row2','Row3',
                                                       'Row4','Row5'],columns=['Column1',
                                                                               'Column2','Column3','Column4'])


# In[83]:


df


# In[84]:


df['Column2'].unique()


# In[85]:


df>2


# In[86]:


df[df['Column2']>2]


# In[ ]:


# This is called conditional indexing


# ## Python Pandas - Part 2
# In Part we are going to learn about
# 
# 1.StringIO
# 2.Pandas read_csv

# In[87]:


### Reading Different Data sources with the help of pandas
from io import StringIO
# string io is called in memory file object


# In[29]:


import pandas as pd


# In[30]:


df=pd.read_csv('mercedesbenz.csv')
df


# In[31]:


df.head()


# In[32]:


type(df)


# In[33]:


data=('Col1,Col2,Col3\n'
     'X,Y,1\n'
     'A,B,2\n'
     'C,D,3')


# In[34]:


type(data)


# In[35]:


# in-memory file object
StringIO(data)


# In[36]:


pd.read_csv(StringIO(data))


# In[37]:


pd.read_csv(StringIO(data),usecols=['Col1','Col2'])


# In[38]:


df=pd.read_csv('mercedesbenz.csv')
df


# In[39]:


df=pd.read_csv('mercedesbenz.csv',usecols=['X0','X1','X2','X3','X4'])
df


# In[40]:


df=pd.read_csv('mercedesbenz.csv',usecols=['X0','X1','X2','X3','X4','X5'])
df


# In[41]:


#COVERT INTO CSV FILE

df.to_csv('test.csv')


# In[42]:


df.to_csv('test.csv', index=False)


# In[43]:


#Datatypes in csv
data = ('a,b,c,d\n'
            '1,2,3,4\n'
            '5,6,7,8\n'
            '9,10,11')


# In[88]:


pd.read_csv(StringIO(data))


# In[90]:


df=pd.read_csv(StringIO(data))


# In[91]:


df


# In[46]:


df.info()


# In[165]:


df.isnull().sum()


# In[167]:


df['a'][0]


# In[168]:


#datatypes in csv
data = ('a,b,c,d\n'
            '1,2,3,4\n'
            '5,6,7,8\n'
            '9,10,11')


# In[169]:


df=pd.read_csv(StringIO(data),dtype={'a':int,'b':float,'c':int})


# In[170]:


df


# In[171]:


df.dtypes


# In[172]:


data = ('index,a,b,c\n'
           '4,apple,bat,5.7\n'
            '8,orange,cow,10')


# In[173]:


pd.read_csv(StringIO(data),index_col=0)


# In[176]:


pd.read_csv(StringIO(data))


# In[179]:


pd.read_csv('https://download.bls.gov/pub/time.series/cu/cu.item')   


# In[184]:


pd.read_csv('https://download.bls.gov/pub/time.series/cu/cu.item')


# In[183]:


url = 'https://download.bls.gov/pub/time.series/cu/cu.item'
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # If successful, read the content as CSV data
    df = pd.read_csv(pd.compat.StringIO(response.text))
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")


# In[182]:





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




