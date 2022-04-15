#!/usr/bin/env python
# coding: utf-8

# ---
# <a name = Section1></a>
# # **1. Introduction**
# ---
# 
# **PROJECT DESCRIPTION:**
# ============================
# 
# See https://projects.insaid.co/capstone2/index.php
# 

# ---
# <a name = Section2></a>
# # **2. Installing and importing libraries**
# ---
# 
# 

# <a name = Section21></a>
# ### **2.1 Installing Libraries**
# 
# 1.   Restart Runtime (in Colab it is menu -> Runtime -> Restart Runtime)
# 2.   Run the below steps
# 

# In[1]:


#!pip install -q datascience                   # Package that is required by pandas profiling
#!pip install -q pandas-profiling              # Library to generate basic statistics about data


# In[2]:


#!pip install -q --upgrade pandas-profiling


# In[3]:


#!pip install mysql-connector-python ## DB Connection ##


# 3.   Restart Runtime again 
# 4.   Run step 2 again
# 5.   Run the code below (to install the other libraries)

# In[ ]:





# In[4]:


#------------------------------------------------------------------------------------------------------------------------------
import pandas as pd                                                 # Importing for panel data analysis
from pandas_profiling import ProfileReport                          # Import Pandas Profiling (To generate Univariate Analysis)
pd.set_option('display.max_columns', None)                          # Unfolding hidden features if the cardinality is high
pd.set_option('display.max_rows', None)                             # Unfolding hidden data points if the cardinality is high
pd.set_option('mode.chained_assignment', None)                      # Removing restriction over chained assignments operations
#-------------------------------------------------------------------------------------------------------------------------------
import numpy as np                                                  # Importing package numpys (For Numerical Python)
from scipy.stats import randint as sp_randint                       # For initializing random integer values
#-------------------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt                                     # Importing pyplot interface using matplotlib
import seaborn as sns                                               # Importin seaborm library for interactive visualization
get_ipython().run_line_magic('matplotlib', 'inline')
#-------------------------------------------------------------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler                    # To scaled data with mean 0 and variance 1
from sklearn.model_selection import train_test_split                # To split the data in training and testing part
from sklearn.model_selection import RandomizedSearchCV              # To find best hyperparamter setting for the algorithm
from sklearn.ensemble import RandomForestClassifier                 # To implement random forest classifier
from sklearn.tree import DecisionTreeClassifier                     # To implement decision tree classifier
from sklearn.metrics import classification_report                   # To generate classification report
from sklearn.metrics import plot_confusion_matrix                   # To plot confusion matrix
import pydotplus                                                    # To generate pydot file
from IPython.display import Image                                   # To generate image using pydot file
#-------------------------------------------------------------------------------------------------------------------------------
import warnings                                                     # Importing warning to disable runtime warnings
warnings.filterwarnings("ignore")                                   # Warnings will appear only once


# ---
# <a name = Section3></a>
# # **3. Loading Train and Test data**
# ---
# 

# 
# <a name = Section31></a>
# ### **3.1 Loading Training data**

# In[16]:


#------ DB Connection ------
import mysql.connector
mydb = mysql.connector.connect(
 user='student', password='student',
 host='cpanel.insaid.co',
 database='Capstone2')

mycursor = mydb.cursor()


# In[17]:


# --- Function get table data as a data frame ---
def getTableDataAsDataFrame(dbCursor, tableName):

  ## Get the Columns  of the table
  dbCursor.execute("show columns from " + tableName)
  cols_result = dbCursor.fetchall()
  cols_df = pd.DataFrame(cols_result)
  cols = cols_df.iloc[:, 0]

  ## Then, get the data stored into a DataFrame
  dbCursor.execute("select * from " + tableName)
  table_data_result = dbCursor.fetchall()
  table_data_df = pd.DataFrame(table_data_result, columns = cols)
  
  ## return the Table data as a DataFrame
  return table_data_df


# In[ ]:





# In[18]:


# === TABLES of the DATABASE ===

# Department Table
department_data_df = getTableDataAsDataFrame(mycursor, 'department_data')
display(department_data_df.head(20));


# In[10]:


department_data_df.info()


# In[12]:


department_data_df.describe()


# In[ ]:


# There are 3 columns and all are categorical
# Total 11 departments are there in an organization
# There is no null values in department table


# In[19]:


# Employee Details table
employee_details_data_df = getTableDataAsDataFrame(mycursor, 'employee_details_data')
display(employee_details_data_df.head());


# In[21]:


employee_details_data_df.info()


# In[26]:


employee_details_data_df.describe(include='all')


# In[ ]:


#employee_details_data dataset have 4 columns. 2 are Integer and 2 are categorical
#total 14245 employee details are there and no null values


# In[9]:


# Employee data
### MAIN TABLE, Has most of the Features required for the model ###
employee_data_df = getTableDataAsDataFrame(mycursor, 'employee_data')
display(employee_data_df.head(10));


# In[ ]:





# In[12]:


employee_data_df.describe()


# In[ ]:


employee_data_df.describe()


# In[22]:


employee_data_df['avg_monthly_hrs'].unique()


# In[13]:


employee_data_df.info()


# In[21]:



## Get the Columns  of the table
mycursor.execute("show columns from employee_data")

cols_result = mycursor.fetchall()
cols_df = pd.DataFrame(cols_result)
cols_df.head(20)
#cols_df.iloc[:, 0]


# <a name = Section32></a>
# ### **3.2 Loading Test data**

# ---
# <a name = Section4></a>
# # **4. Pre-Profiling Report**
# ---

# In[64]:


#!pip install pandas-profiling==2.7.1
#profile = ProfileReport(df = employee_train)
#profile.to_file(output_file = 'Pre Profiling Report.html')
#print('Accomplished!')


# In[65]:


#from google.colab import files                   # Use only if you are using Google Colab, otherwise remove it
#files.download('Pre Profiling Report.html')      # Use only if you are using Google Colab, otherwise remove it


# ---
# <a name = Section5></a>
# # **5. Exploratory Data Analysis**
# ---

# <a name = Section51></a>
# ### **5.1 Pairwise Plots**

# <a name = Section52></a>
# ### **5.2 Heatmaps**

# In[66]:


#### Detailed HEAT MAP of Correlations ####


# ---
# <a name = Section6></a>
# # **6. Feature Selection**
# ---

# ---
# <a name = Section7></a>
# # **7. Filling Missing / Null values if any**
# ---

# 
# <a name = Section71></a>
# ### **7.1 Data Description (Mean, median, std. dev. etc)**

# 
# <a name = Section72></a>
# ### **7.2 Fill missing / null values with Mean/Median/Mode**

# ---
# <a name = Section8></a>
# # **8. Feature Engineering**
# ---

# ---
# <a name = Section9></a>
# # **9. Data Preparation (before model creation)**
# ---

# <a name = Section91></a>
# ### **9.1 Scaling**

# <a name = Section92></a>
# ### **9.2 X and y creation**

# <a name = Section93></a>
# ### **9.3 Train-Test Split**

# <a name = Section94></a>
# ### **9.4 Scaling (Fit-Transform for Train, Transform for Test)**

# ---
# <a name = Section10></a>
# # **10. Model creation and prediction**
# ---

# <a name = Section91></a>
# ### **10.1 Random Forest classification model**

# <a name = Section102></a>
# ### **10.2 Prediction**

# ---
# <a name = Section11></a>
# # **11. Model Evaluation**
# ---
