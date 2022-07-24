# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 08:12:34 2022
Python Programming (ACFI827)
Assignment 2. Fraud dection from customer transactions data
@author: rasbe
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import copy
import re

# =============================================================================
# Task 1 Importing Data Set
# =============================================================================

""" 
Read data from csv and json
"""
def readfile(csv_file,json_file):
    
    #Import dataset Identity.json and transaction.
    directory_path= os.path.abspath(os.getcwd())
    ('Get directory Path:', directory_path)
    file_path_csv = csv_file
    file_path_json = json_file
    df_transaction = pd.read_csv(directory_path + file_path_csv)
    df_identity = pd.read_json(directory_path + file_path_json)
    return df_transaction,df_identity


""" 
# Output
# Get directory Path
"""

""" 
Get transaction data and number of rows, columns
"""
def get_data():   
    df_transaction, df_identity = readfile(csv_file='/assignment2/transaction.csv',
         json_file='/assignment2/identity.json')
    #Display first 4 rows of both DataFrame
    print("\nRead transaction data\n", df_transaction.head(4))
    print("\nRead identity data\n", df_identity.head(4))
    
    #Shape of both datasets
    print(f"There are {df_transaction.shape[0]} rows and {df_transaction.shape[1]} columns in Transaction.csv")
    print(f"There are {df_identity.shape[0]} rows and {df_transaction.shape[1]} columns in Identity.json")
    
    return df_transaction,df_identity

""" 
# Output
Read transaction data
    TransactionID  isFraud  TransactionDT  ...  dist2 P_emaildomain  R_emaildomain
0        2987000        0          86400  ...    NaN           NaN            NaN
1        2987001        0          86401  ...    NaN     gmail.com            NaN
2        2987002        0          86469  ...    NaN   outlook.com            NaN
3        2987003        0          86499  ...    NaN     yahoo.com            NaN

[4 rows x 17 columns]

Read identity data
    TransactionID  Device_rating  ... DeviceType                     DeviceInfo
0        2987004              0  ...     mobile  SAMSUNG SM-G892A Build/NRD90M
1        2987008             -5  ...     mobile                     iOS Device
2        2987010             -5  ...    desktop                        Windows
3        2987011             -5  ...    desktop                           None

[4 rows x 6 columns]
"""
""" 
# Output
There are 590540 rows and 17 columns in Transaction.csv
There are 144233 rows and 17 columns in Identity.json
"""

# =============================================================================
# Task 2 Merge Data
# =============================================================================

# Merge identity and transaction dataset by transaction ID

""" 
Merge csv data and identity data 
"""
def merge_data():
    df_trans, df_idy = get_data()
    df_merge = pd.merge(df_trans,df_idy,on='TransactionID')
    return df_merge



""" 
Get number of rows and columns,columns names of merge data
"""
def merge_datainfo():
# What is the shape of merged/combined dataset?
    df_merge = merge_data()
    print(f"There are {df_merge.shape[0]} rows and {df_merge.shape[1]} columns in the merge dataset.")
# Display column names in the merged DataFrame
    print("\nDisplay Column Names: \n",list(df_merge.columns))

merge_datainfo()

""" 
# Output
There are 144233 rows and 22 columns in the merge dataset.
Display Column Names:
 ['TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt', 
  'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 
  'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain', 
  'Device_rating', 'Device_system', 'Browser', 'DeviceType', 'DeviceInfo']
"""

# =============================================================================
# Task 3 , Descriptive Data Analysis 
# ============================================================================= 

#Checking data types of each column

""" 
Analyze data
3.1.Find mission data
3.2.Find percentage of missing data 
3.3.Visualize missing data
3.4.Detect outliers
3.5.Statistical analysis of numerical & categorical data
3.6.Visualize statistical analysis
3.7.Extract fraud & valid transactions from merge data
3.8.Convert categorical data into numerical data
3.9.Find correlation between data
3.10.Visualize correlation

"""
#Draw histogram
def draw_histogram(data,df,title):
    df[data].plot.hist(bins=10)
    plt.title(title)
    plt.show()

#Draw bar graph using seaborn library
def draw_snsbargraph(w,h,p_color,xval,df):
    plt.figure(figsize=(w, h))
    sns.set_palette(p_color)
    sns.countplot(x=xval,data=df)
    plt.show()

def analyze_data():
    df_merge = merge_data()
    print("\n===Merge Data Type===\n")
    print(df_merge.info())
    
#3.1.Find mission data
    print("\n===Missing Data===\n")
    print(df_merge.isna().sum())

#3.2.Percentage of data missing in each feature
    print("\n===Missing Percentage of Each Feature===\n")
    missing_percent = df_merge.isna().sum() * 100 / len(df_merge)
    print(round(missing_percent,2))
    
#3.3.Visualization to show the amount of missing data
        
    missing_percent.plot.bar(rot=90,color='red')
    plt.title('Missing Data Visualization', fontdict={'fontname': 'Tahoma', 'fontsize' : 20})
    plt.xlabel('Transaction & Identity Data',fontdict={'fontname':'Tahoma', 'fontsize': 13})
    plt.ylabel('Percentage % ',fontdict={'fontname':'Tahoma', 'fontsize':13})
    plt.legend(['Missing percentage'])
    plt.show()
        
#Statistical analysis of numerical and categorical features. 
    #Numerical feature
    #dist1 is 100% missing data so it will be excluded 
    print("\n====Unique Values===\n")
    print(df_merge.nunique())

#3.4.Detect outliers
    
    #Detecting Outliers 
    
    df_merge.boxplot(column=['TransactionAmt'], by='ProductCD')
    df_merge.boxplot(column=['TransactionAmt'], by='card4')
    df_merge.boxplot(column=['TransactionAmt'], by='card6')
        
    #Checking frequency distribution
    draw_histogram('card1',df_merge,'Card1 frequency distribution')
    draw_histogram('card2',df_merge,'Card2 frequency distribution')
    draw_histogram('card3',df_merge,'Card3 frequency distribution')  
    draw_histogram('card5',df_merge,'Card5 frequency distribution')  
    draw_histogram('addr1',df_merge,'Address1 frequency distribution') 
    draw_histogram('Device_rating', df_merge, 'Device rating frequency distribution')  
    
    df_numerical = df_merge.loc[:,['TransactionAmt','dist2',
                                   'Device_rating']]
    print("\n===Statistical Analysis of Numerical Feature===\n")
    print(df_numerical.describe())

#Report statistical analysis results through visualizations
#3.5.Statistical analysis of numerical & categorical data    

    df_numerical_g = df_numerical.describe()
    df_numerical_g[1:].plot()
        
    #Categorical feature
    df_categorical = df_merge.loc[:,['ProductCD',
                                  'card4','card6',
                                   'P_emaildomain','R_emaildomain',
                                  'Device_system','Browser','DeviceType']]
    
    print("\n===Statistical Analysis of Categorical Feature===\n")
    print(df_categorical.describe())

#3.6.Visualize statistical analysis    
    
    for column in df_categorical.columns:
        plt.figure(figsize=(30, 15))
        cat_chart = sns.countplot(x = column, data = df_categorical,palette ="Set2")
        cat_chart.axes.set_title(column,fontsize=50)
        cat_chart.set_xticklabels(cat_chart.get_xticklabels(),rotation=90,size=17)
        plt.show()
    
     
#Correlation between features in the data and fraud (isFraud)
#3.7.Extract fraudulent & valid transactions from merge data
    #Visualize fraud and valid transactions 
    draw_snsbargraph(10,6,"Set2",'isFraud',df_merge)
    
    #Make separate dataframe
    df_isfraud = copy.deepcopy(df_merge[df_merge['isFraud'] == 1])
    df_valid = copy.deepcopy(df_merge[df_merge['isFraud'] == 0])
    print("\n===Fraud Data===\n")
    print(df_isfraud.head(3))  
    print("There are {}".format(len(df_isfraud)),"fraud transaction")
    print("There are {}".format(len(df_valid)),"valid transaction")
    
    #Checking amount details of fraud transactions
    print("\n===Fraud Transaction Amount===\n",df_isfraud.TransactionAmt.describe())
    print("\n===Valid Transaction Amount===\n",df_valid.TransactionAmt.describe())
       
    #Which card has more fraud transactions 
    draw_snsbargraph(10,6,"Set2",'card4',df_isfraud)
   
    #Which card(debit/card) has more fraud transactions 
    draw_snsbargraph(10,6,"Set2",'card6', df_isfraud)   
    
    #Which address has more fraud transactions
    
    plt.figure(figsize=(20, 8))
    add_plot = sns.countplot(x='addr1',data=df_isfraud)
    add_plot.set_xticklabels(add_plot.get_xticklabels(),rotation=90,size=13)
    
    
    #Which address has more fraud transactions
    plt.figure(figsize=(20, 8))
    add_plot = sns.countplot(x='addr2',data=df_isfraud)
    add_plot.set_xticklabels(add_plot.get_xticklabels(),rotation=90,size=13)
    
        
    #Which device system has more fraud transactions
    pattern ="[0-9_.]"
    clean_ds=[]

    for item in df_isfraud['Device_system'].values:
       if(isinstance(item, str)):
           #Remove empty space at the end of the string
           item_r = re.sub(pattern,"",item)
           item_r = item_r.strip(' ')
           clean_ds.append(item_r)
       else:
           clean_ds.append("None")
        
    df_isfraud['Clean_Device_system']= clean_ds  
    
    plt.figure(figsize=(20, 8))
    sns.set_palette("Paired")
    add_plot = sns.countplot(x='Clean_Device_system',data=df_isfraud)
    add_plot.set_xticklabels(add_plot.get_xticklabels(),rotation=90,size=13)
    
    
#3.8.Convert categorical data into numerical data
    labelencoder = LabelEncoder()
    df_isfraud['ProductCD'] = labelencoder.fit_transform(df_isfraud['ProductCD'])
    df_isfraud['card4']     = labelencoder.fit_transform(df_isfraud['card4'])
    df_isfraud['card6']     = labelencoder.fit_transform(df_isfraud['card6'])
    df_isfraud['P_emaildomain'] = labelencoder.fit_transform(df_isfraud['P_emaildomain'])
    df_isfraud['R_emaildomain'] = labelencoder.fit_transform(df_isfraud['R_emaildomain'])
    df_isfraud['Device_system'] = labelencoder.fit_transform(df_isfraud['Device_system'])
    df_isfraud['DeviceType'] = labelencoder.fit_transform(df_isfraud['DeviceType'])
    df_isfraud['DeviceInfo'] = labelencoder.fit_transform(df_isfraud['DeviceInfo'])
    df_isfraud['Browser']    = labelencoder.fit_transform(df_isfraud['Browser'])
    
    #Remove some columns 
    df_isfraud = df_isfraud.drop(['TransactionID','isFraud','dist1'],axis=1)
    
#3.9.Find correlation between data
    coor_matrix = df_isfraud.corr(method='pearson').round(4)
    print("\n===Matrix===\n", coor_matrix)
   

#3.10.Visualize correlation

    plt.figure(figsize=(20,20))
    plt.title('Data Correlation',fontdict={'fontname':'Tahoma','fontsize': 40})
    sns.heatmap(coor_matrix,annot=True,vmax=1,vmin=-1,center=0,cmap='vlag') 
    plt.show()   

    
analyze_data()
""" 
# Output
===Merge Data Type===
<class 'pandas.core.frame.DataFrame'>
Int64Index: 144233 entries, 0 to 144232
Data columns (total 22 columns):
 #   Column          Non-Null Count   Dtype  
---  ------          --------------   -----  
 0   TransactionID   144233 non-null  int64  
 1   isFraud         144233 non-null  int64  
 2   TransactionDT   144233 non-null  int64  
 3   TransactionAmt  144233 non-null  float64
 4   ProductCD       144233 non-null  object 
 5   card1           144233 non-null  int64  
 6   card2           143331 non-null  float64
 7   card3           144061 non-null  float64
 8   card4           144049 non-null  object 
 9   card5           143277 non-null  float64
 10  card6           144055 non-null  object 
 11  addr1           83786 non-null   float64
 12  addr2           83786 non-null   float64
 13  dist1           0 non-null       float64
 14  dist2           37593 non-null   float64
 15  P_emaildomain   130842 non-null  object 
 16  R_emaildomain   131083 non-null  object 
 17  Device_rating   144233 non-null  int64  
 18  Device_system   77565 non-null   object 
 19  Browser         140282 non-null  object 
 20  DeviceType      140810 non-null  object 
 21  DeviceInfo      118666 non-null  object 
dtypes: float64(8), int64(5), object(9)
memory usage: 25.3+ MB
None

===Missing Data===

TransactionID          0
isFraud                0
TransactionDT          0
TransactionAmt         0
ProductCD              0
card1                  0
card2                902
card3                172
card4                184
card5                956
card6                178
addr1              60447
addr2              60447
dist1             144233
dist2             106640
P_emaildomain      13391
R_emaildomain      13150
Device_rating          0
Device_system      66668
Browser             3951
DeviceType          3423
DeviceInfo         25567
dtype: int64

===Missing Percentage of Each Feature===

TransactionID       0.00
isFraud             0.00
TransactionDT       0.00
TransactionAmt      0.00
ProductCD           0.00
card1               0.00
card2               0.63
card3               0.12
card4               0.13
card5               0.66
card6               0.12
addr1              41.91
addr2              41.91
dist1             100.00
dist2              73.94
P_emaildomain       9.28
R_emaildomain       9.12
Device_rating       0.00
Device_system      46.22
Browser             2.74
DeviceType          2.37
DeviceInfo         17.73
dtype: float64

===Statistical Analysis of Numerical Feature===
       TransactionAmt  dist1         dist2  Device_rating
count   144233.000000    0.0  37593.000000  144233.000000
mean        83.554533    NaN    231.945575     -10.170502
std         99.850258    NaN    529.251862      14.347949
min          0.251000    NaN      0.000000    -100.000000
25%         25.453000    NaN      7.000000     -10.000000
50%         50.000000    NaN     37.000000      -5.000000
75%        100.000000    NaN    206.000000      -5.000000
max       1800.000000    NaN  11623.000000       0.000000


===Statistical Analysis of Categorical Feature===

       ProductCD   card4   card6  ... Device_system      Browser DeviceType
count     144233  144049  144055  ...         77565       140282     140810
unique         4       4       3  ...            75          130          2
top            C    visa  credit  ...    Windows 10  chrome 63.0    desktop
freq       62192   89299   75090  ...         21155        22000      85165

[4 rows x 8 columns]

===Fraud Data===

    TransactionID  isFraud  ...  DeviceType                 DeviceInfo
52        2987240        1  ...      mobile  Redmi Note 4 Build/MMB29M
53        2987243        1  ...      mobile  Redmi Note 4 Build/MMB29M
54        2987245        1  ...      mobile  Redmi Note 4 Build/MMB29M

[3 rows x 22 columns]

There are 11318 fraud transaction
There are 132915 valid transaction

===Fraud Transaction Amount===
 count    11318.000000
mean        88.810342
std        107.991313
min          0.292000
25%         25.000000
50%         50.000000
75%        100.032250
max       1800.000000
Name: TransactionAmt, dtype: float64

===Valid Transaction Amount===
 count    132915.000000
mean         83.106989
std          99.113705
min           0.251000
25%          25.576000
50%          50.000000
75%         100.000000
max        1800.000000
Name: TransactionAmt, dtype: float64

===Matrix===
                 TransactionDT  TransactionAmt  ...  DeviceType  DeviceInfo
TransactionDT          1.0000         -0.0662  ...      0.0119     -0.0802
TransactionAmt        -0.0662          1.0000  ...      0.0674      0.0821
ProductCD             -0.0293          0.4382  ...      0.0168      0.0994
card1                  0.0446         -0.0129  ...      0.0308      0.0150
card2                 -0.0259          0.1117  ...     -0.0032     -0.0060
card3                  0.0656         -0.4516  ...     -0.0269     -0.1102
card4                  0.0245         -0.0799  ...     -0.0044      0.0023
card5                 -0.0295          0.0558  ...     -0.0258      0.0356
card6                  0.0496         -0.0888  ...      0.0137     -0.0243
addr1                  0.0598          0.0045  ...      0.0402      0.0316
addr2                  0.2295          0.1847  ...      0.0921     -0.0548
dist2                 -0.0364          0.0136  ...      0.0183     -0.0245
P_emaildomain          0.0296         -0.0048  ...     -0.0326      0.0070
R_emaildomain         -0.0227          0.0718  ...     -0.0606     -0.0124
Device_rating         -0.0259          0.0785  ...     -0.1255      0.0318
Device_system          0.0879         -0.3727  ...      0.0677      0.0679
Browser                0.0767          0.2394  ...      0.2572      0.2041
DeviceType             0.0119          0.0674  ...      1.0000     -0.2128
DeviceInfo            -0.0802          0.0821  ...     -0.2128      1.0000

[19 rows x 19 columns]

"""

















