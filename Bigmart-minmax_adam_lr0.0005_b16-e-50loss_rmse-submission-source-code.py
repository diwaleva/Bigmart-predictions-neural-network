## A. Problem Statement

### The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. The aim is to build a predictive model and predict the sales of each product at a particular outlet.

### Using this model, BigMart will try to understand the properties of products and outlets which play a key role in increasing sales.
# B. Data Dictionary

### Variable	           -       Description
* **Item_Identifier**  - 	     Unique product ID
* **Item_Weight**      -       Weight of product
* **Item_Fat_Content** -	     Whether the product is low fat or not
* **Item_Visibility**  -	     The % of total display area of all products in a store allocated to the particular product
* **Item_Type**        -  	 The category to which the product belongs
* **Item_MRP**         -	     Maximum Retail Price (list price) of the product
* **Outlet_Identifier**-	     Unique store ID
* **Outlet_Establishment_Year** - The year in which store was established
* **Outlet_Size**      -	     The size of the store in terms of ground area covered
* **Outlet_Location_Type**  -  The type of city in which the store is located
* **Outlet_Type**      -	     Whether the outlet is just a grocery store or some sort of supermarket
* **Item_Outlet_Sales** -  	 Sales of the product in the particular store. This is the outcome variable to be predicted
# **C. Pre-processing Steps**

<ol>1. Filling the missing values</ol>
<ol>2. Converting categories to numbers</ol>
<ol>3. Bring all the variables in range 0 to 1</ol>
# importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
from math import sqrt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
# check version on pandas
print('Version of pandas:', pd.__version__)
# reading the loan prediction data
data = pd.read_csv('/content/sample_data/train_XnW6LSF.csv')
#data = pd.read_csv('train_XnW6LSF.csv')
# looking at the first five rows of the data
data.head()
# shape of the data
data.shape
backup_actual_data = pd.DataFrame()
backup_actual_data['Item_Identifier'] = data['Item_Identifier'].copy()
backup_actual_data['Outlet_Identifier'] = data['Outlet_Identifier'].copy()
backup_actual_data['Item_Outlet_Sales'] = data['Item_Outlet_Sales'].copy()
data['Item_Outlet_Sales'].max()
# checking missing values in the data
data.isnull().sum()
# data types of the variables
data.dtypes
## C-1. Variable itendification and coversion to respective data types.
# Identifying variables with object datatype
data.dtypes[data.dtypes == 'object']
#Converting object data type to category
data['Item_Identifier'] = data['Item_Identifier'].astype('category')
data['Item_Fat_Content'] = data['Item_Fat_Content'].astype('category')
data['Item_Type'] = data['Item_Type'].astype('category')
data['Outlet_Identifier'] = data['Outlet_Identifier'].astype('category')
data['Outlet_Size'] = data['Outlet_Size'].astype('category')
data['Outlet_Location_Type'] = data['Outlet_Location_Type'].astype('category')
data['Outlet_Type'] = data['Outlet_Type'].astype('category')
data['Outlet_Establishment_Year'] = data['Outlet_Establishment_Year'].astype('category')
# check the updated datatypes.
data.dtypes
# isolating categorical datatypes
categorical = data.select_dtypes(include=['category'])[:]
categorical.dtypes
### Since the Item_Outlet_Sales has float and continuous values, **this is a problem of Regression type**.
## C-2. Filling the missing values
### Categorical data: Mode
# filling missing values of categorical variables with mode

data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0], inplace=True)
### Continuous data: Mean
# filling missing values of continuous variables with mean
data['Item_Weight'].fillna(data['Item_Weight'].mean(), inplace=True)
# checking missing values after imputation
data.isnull().sum()
## C-3. Converting categories to numbers since our model only processes numerical data.
### a. Identifying Unique Values in Categories
# identififying unique category values for the category variables

item_identifier_unique_categories = data['Item_Identifier'].unique()
item_identifier_unique_categories.shape
data['Item_Fat_Content'].unique().value_counts()
item_type_unique_categories = data['Item_Type'].unique()
item_type_unique_categories.shape
outlet_identifier_unique_categories = data['Outlet_Identifier'].unique()
outlet_identifier_unique_categories.shape
outlet_size_unique_categories = data['Outlet_Size'].unique()
outlet_size_unique_categories.shape

outlet_location_type_unique_categories = data['Outlet_Location_Type'].unique()
outlet_location_type_unique_categories.shape
outlet_type_unique_categories = data['Outlet_Type'].unique()
outlet_type_unique_categories.shape
outlet_establishment_year_unique_categories = data['Outlet_Establishment_Year'].unique()
outlet_establishment_year_unique_categories.shape
item_identifier_unique_categories = data['Item_Identifier'].unique()
item_identifier_unique_categories.shape
data.dtypes
### b. Normalizing and Encoding categories
# Normalizing categories
data['Item_Fat_Content'] = data['Item_Fat_Content'].map({'LF':'Low Fat', 'low fat': 'Low Fat', 'Low Fat':'Low Fat','reg':'Regular', 'Regular':'Regular'})
item_fat_content_unique_categories = data['Item_Fat_Content'].unique()
item_fat_content_unique_categories.shape
#### C3-c. Label Encoder
# converting the categories into numbers using map function
lblencoder = LabelEncoder()
ohencoder = OneHotEncoder()


data['Item_Fat_Content'] = lblencoder.fit_transform(data['Item_Fat_Content'])

data['Outlet_Size'] = lblencoder.fit_transform(data['Outlet_Size'])

data['Outlet_Type'] = lblencoder.fit_transform(data['Outlet_Type'])

data['Outlet_Location_Type'] = lblencoder.fit_transform(data['Outlet_Location_Type'])

data.head()
data.dtypes
#### C3-d. Univariate Analysis
# custom function for easy and efficient analysis of numerical univariate

def UVA_numeric(data, var_group, n1):
  '''
  Univariate_Analysis_numeric
  takes a group of variables (INTEGER and FLOAT) and plot/print all the descriptives and properties along with KDE.

  Runs a loop: calculate all the descriptives of i(th) variable and plot/print it
  '''

  size = len(var_group)
  plt.figure(figsize = (size*n1,3), dpi = 300)

  #looping for each variable
  for j,i in enumerate(var_group):

    # calculating descriptives of variable
    mini = data[i].min()
    maxi = data[i].max()
    ran = data[i].max()-data[i].min()
    mean = data[i].mean()
    median = data[i].median()
    st_dev = data[i].std()
    skew = data[i].skew()
    kurt = data[i].kurtosis()

    # calculating points of standard deviation
    points = mean-st_dev, mean+st_dev

    #Plotting the variable with every information
    plt.subplot(1,size,j+1)
    sns.kdeplot(x=data[i], fill=True)
    sns.lineplot(x=points, y=[0,0], color = 'black', label = "within 1 std_dev")
    sns.scatterplot(x=[mini,maxi], y=[0,0], color = 'orange', label = "min/max")
    sns.scatterplot(x=[mean], y=[0], color = 'red', label = "mean")
    sns.scatterplot(x=[median], y=[0], color = 'blue', label = "median")
    plt.xlabel('{}'.format(i), fontsize = 20)
    plt.ylabel('density')
    plt.title('within 1 std_dev = {}; kurtosis = {};\nskew = {}; range = {}\nmean = {}; median = {}'.format((round(points[0],2),round(points[1],2)),
                                                                                                   round(kurt,2),
                                                                                                   round(skew,2),
                                                                                                   (round(mini,2),round(maxi,2),round(ran,2)),
                                                                                                   round(mean,2),
                                                                                                   round(median,2)))
* -ve Values of Skew indicates bias towards right side and +ve values indicates bias towards left side.

* "As a general guideline, a **skewness value between −1 and +1 is considered excellent**, but a **value between −2 and +2 is generally considered acceptable**. **Values beyond −2 and +2 are considered indicative of substantial nonnormality.**" (Hair et al., 2022, p. 66).

* "the general guideline is that if the **kurtosis is greater than +2, the distribution is too peaked**. Likewise, **a kurtosis of less than −2 indicates a distribution that is too flat**. When **both skewness and kurtosis are close to zero, the pattern of responses is considered a normal distribution**(George & Mallery, 2019)." (Hair et al., 2022, p. 66).

* "When both skewness and kurtosis are zero (a situation that researchers are very unlikely to ever encounter), the pattern of responses is considered a normal distribution."

References

Hair, J. F., Hult, G. T. M., Ringle, C. M., & Sarstedt, M. (2022). A Primer on Partial Least Squares Structural Equation Modeling (PLS-SEM) (3 ed.). Thousand Oaks, CA: Sage.
df_numerical_columnss_values = data.select_dtypes(include=['int64','float64','Int32'])
numerical_columnss_values = df_numerical_columnss_values.columns.tolist()
UVA_numeric(df_numerical_columnss_values, numerical_columnss_values[0:3], 6)
UVA_numeric(df_numerical_columnss_values, numerical_columnss_values[3:6], 6)
UVA_numeric(df_numerical_columnss_values, numerical_columnss_values[6:9], 6)
### From above we observe that the bias and outliers for the variables in the dataset are in excellent to acceptable range and therefore we can say the variables are normally distributed and **we would not need any outlier removal.**
#### C4-d. Removing outliers from the dataset


# custom function for easy outlier analysis

def UVA_outlier(data, var_group, include_outlier = True):
  '''
  Univariate_Analysis_outlier:
  takes a group of variables (INTEGER and FLOAT) and plot/print boplot and descriptives\n
  Runs a loop: calculate all the descriptives of i(th) variable and plot/print it \n\n

  data : dataframe from which to plot from\n
  var_group : {list} type Group of Continuous variables\n
  include_outlier : {bool} whether to include outliers or not, default = True\n
  '''

  size = len(var_group)
  plt.figure(figsize = (7*size,4), dpi = 100)

  #looping for each variable
  for j,i in enumerate(var_group):

    # calculating descriptives of variable
    quant25 = data[i].quantile(0.25)
    quant75 = data[i].quantile(0.75)
    IQR = quant75 - quant25
    med = data[i].median()
    whis_low = quant25-(1.5*IQR)
    whis_high = quant75+(1.5*IQR)

    # Calculating Number of Outliers
    outlier_high = len(data[i][data[i]>whis_high])
    outlier_low = len(data[i][data[i]<whis_low])

    if include_outlier == True:
      print(include_outlier)
      #Plotting the variable with every information
      plt.subplot(1,size,j+1)
      sns.boxplot(data=data[i], orient="v")
      plt.ylabel('{}'.format(i))
      plt.title('With Outliers\nIQR = {}; Median = {} \n 1st,3rd  quartile = {};\n Outlier (low/high) = {} \n'.format(
                                                                                                   round(IQR,2),
                                                                                                   round(med,2),
                                                                                                   (round(quant25,2),round(quant75,2)),
                                                                                                   (outlier_low,outlier_high)
                                                                                                   ))

    else:
      # replacing outliers with max/min whisker
      data2 = data[var_group][:]
      data2[i][data2[i]>whis_high] = whis_high+1
      data2[i][data2[i]<whis_low] = whis_low-1

      # plotting without outliers
      plt.subplot(1,size,j+1)
      sns.boxplot(data=data2[i], orient="v")
      plt.ylabel('{}'.format(i))
      plt.title('Without Outliers\nIQR = {}; Median = {} \n 1st,3rd  quartile = {};\n Outlier (low/high) = {} \n'.format(
                                                                                                   round(IQR,2),
                                                                                                   round(med,2),
                                                                                                   (round(quant25,2),round(quant75,2)),
                                                                                                   (outlier_low,outlier_high)
                                                                                                   ))
#### C4-e. Viewing Value Counts to Understand the Distribution of The Variables in the Dataset.
df_data_column_values =  data.select_dtypes(include=['category', 'int64','float64','Int32'])
data_column_values = df_data_column_values.columns.tolist()
data_column_values
def generate_valuecounts_vargroup(df, numerical_columnss_values):
  #
  rd_data_test = df[numerical_columnss_values]

  # filtering using standard deviation (not considering obseravtions > mean + 3* standard deviation)
  for indices, row in rd_data_test.iterrows():
      for column in numerical_columnss_values:
         print(column+'\n')
         print(rd_data_test[column].value_counts())


#generate_valuecounts_vargroup(data, data_column_values[1:5])
#### C4-f. Mapping Correlations between the Variables
# %matplotlib inline
plt.figure(figsize=(46,43))
cor = df_numerical_columnss_values.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
## **D-1. Encoding all the features in the dataset for training a model with ALL features**

df_encoded_all = data.copy()  # Creates a copy of all variables in the input data. It is partially encoded. Some categorica
#
df_encoded_all.dtypes
df_backup_encoded_all = df_encoded_all.copy()


df_encoded_all = df_encoded_all.drop('Item_Identifier', axis=1)

df_encoded_all.select_dtypes(include=['int64','float64','Int32'])[:]
df_encoded_bigmart_cat_column_list_all = df_encoded_all.select_dtypes(include=['category']).columns.tolist()
def encode_categories(df_to_encode, df_selected_columns, dataname):
  #Extract categorical columns from the dataframe
  #Here we extract the columns with object datatype as they are the categorical columns
  #categorical_columns = df_to_encode.select_dtypes(include=['category']).columns.tolist()
  #categorical_columns.remove('Item_Identifier')
  #categorical_columns.remove('Outlet_Identifier')
  categorical_columns = df_selected_columns

  #Initialize OneHotEncoder
  encoder = OneHotEncoder(sparse_output=False)

  # Apply one-hot encoding to the categorical columns
  one_hot_encoded = encoder.fit_transform(df_to_encode[categorical_columns])

  #Create a DataFrame with the one-hot encoded columns
  #We use get_feature_names_out() to get the column names for the encoded data
  one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

  # Concatenate the one-hot encoded dataframe with the original dataframe
  df_to_encode = pd.concat([one_hot_df, df_to_encode], axis=1)

  # Drop the original categorical columns
  df_to_encode = df_to_encode.drop(categorical_columns, axis=1)

  # Display the resulting dataframe
  print(f"Encoded ' + dataname + 'data : \n{df_to_encode}")

  #df_to_encode.head()
  return df_to_encode.copy()
df_encoded_all.shape
df_encoded_all.columns
df_encoded_bigmart_cat_column_list_all
df_encoded_all = encode_categories(df_encoded_all, df_encoded_bigmart_cat_column_list_all, "Bigmart All Features Dataset")
## **E. Now Scaling the Training set Using  Minmax, Standard, and Robust Scaler Techniques**
### E1. Scaling the Data for Normal Distribution
#### By using Normalization all variables are scaled and brought into certain range eith 0 to 1 or -1 t0 1 or column min and max. This helps with efficient model training.
### Below we are using MinMax Scaler, Standard Scalar, and Robust Scalar and  View Density and Distribution.
# For ALL Features
from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()
minmax_all = scaler.fit_transform(df_encoded_all)
minmax_all = pd.DataFrame(minmax_all, columns=df_encoded_all.columns.tolist())

scaler = preprocessing.StandardScaler()
standard_all = scaler.fit_transform(df_encoded_all)
standard_all = pd.DataFrame(standard_all, columns=df_encoded_all.columns.tolist())

scaler = preprocessing.RobustScaler()
robust_all = scaler.fit_transform(df_encoded_all)
robust_all = pd.DataFrame(robust_all, columns=df_encoded_all.columns.tolist())

display(df_encoded_all.head())
print('*' * 45)
print('\033[1m'+'With minmax scaling:'+'\033[0m')
display(minmax_all.head())
print('*' * 45)
print('\033[1m'+'With standard scaling:'+'\033[0m')
display(standard_all.head())
print('*' * 45 )
print('\033[1m'+'With robust scaling:'+'\033[0m')
display(robust_all.head())
import matplotlib.colors as mcolors
import random

def random_color_generator():
    color = random.choice(list(mcolors.CSS4_COLORS.keys()))
    return color
def plot_kde(df_encoded_scaled, x, scalingstr):
  plt.subplot(1,4,x)
  numerical_columns_values = df_encoded_scaled.columns.tolist()
  for indices, row in df_encoded_scaled.iterrows():
      for column in numerical_columns_values:
          rnd_color = random_color_generator()
          sns.kdeplot(df_encoded_scaled[column], color = rnd_color)
  plt.title(scalingstr)
#plot_kde(df_numerical_columnss_values, 1, 'Without Scaling')
plt.figure(figsize=(15,4))

## Baseline Dataset
plt.subplot(1,4,1)
sns.kdeplot(df_encoded_all['Item_Fat_Content'],  color ='green')
sns.kdeplot(df_encoded_all['Item_Visibility'], color ='red')
sns.kdeplot(df_encoded_all['Item_MRP'],  color ='orange')
sns.kdeplot(df_encoded_all['Outlet_Size'],  color ='brown')
sns.kdeplot(df_encoded_all['Outlet_Location_Type'],  color ='yellow')
sns.kdeplot(df_encoded_all['Outlet_Type'],  color ='#F72585')
sns.kdeplot(df_encoded_all['Item_Outlet_Sales'],  color ='#7209B7')
plt.title('Without Scaling')

plt.subplot(1,4,4)
sns.kdeplot(minmax_all['Item_Fat_Content'],  color ='green')
sns.kdeplot(minmax_all['Item_Visibility'], color ='red')
sns.kdeplot(minmax_all['Item_MRP'],  color ='orange')
sns.kdeplot(minmax_all['Outlet_Size'],  color ='brown')
sns.kdeplot(minmax_all['Outlet_Location_Type'],  color ='yellow')
sns.kdeplot(minmax_all['Outlet_Type'],  color ='#F72585')
sns.kdeplot(minmax_all['Item_Outlet_Sales'],  color ='#7209B7')
plt.title('After Min-Max Scaling')

plt.subplot(1,4,3)
sns.kdeplot(standard_all['Item_Fat_Content'],  color ='green')
sns.kdeplot(standard_all['Item_Visibility'], color ='red')
sns.kdeplot(standard_all['Item_MRP'],  color ='orange')
sns.kdeplot(standard_all['Outlet_Size'],  color ='brown')
sns.kdeplot(standard_all['Outlet_Location_Type'],  color ='yellow')
sns.kdeplot(standard_all['Outlet_Type'],  color ='#F72585')
sns.kdeplot(standard_all['Item_Outlet_Sales'],  color ='#7209B7')
plt.title('After Standard Scaling')

plt.subplot(1,4,2)
sns.kdeplot(robust_all['Item_Fat_Content'],  color ='green')
sns.kdeplot(robust_all['Item_Visibility'], color ='red')
sns.kdeplot(robust_all['Item_MRP'],  color ='orange')
sns.kdeplot(robust_all['Outlet_Size'],  color ='brown')
sns.kdeplot(robust_all['Outlet_Location_Type'],  color ='yellow')
sns.kdeplot(robust_all['Outlet_Type'],  color ='#F72585')
sns.kdeplot(robust_all['Item_Outlet_Sales'],  color ='#7209B7')
plt.title('After Robust Scaling')


plt.tight_layout()
plt.show()
#### **From above we see that some of the variables have outliers present at the hogher end of their distributions. Overall the Min Max Scaler produces a balanced spead and so we will use Min Max Scaled Data for Model Training**
minmax_all
minmax_all.columns
### E2. Model Training, Predictions, And Checking Assumptions.
#importing Linear Regression and metric mean square error
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_absolute_error as mae
#from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import r2_score as r2
#### E3. Segregating variables: Independent and Dependent Variables
#seperating independent and dependent variables
x_6 = minmax_all.drop(['Item_Outlet_Sales'], axis=1)

y = minmax_all['Item_Outlet_Sales']
y_train_a = y
x_6.shape, y.shape
### E4. Splitting the data into train set and the validation set
# Importing the train test split function
from sklearn.model_selection import train_test_split
train_x_6,val_x_6,train_y,val_y = train_test_split(x_6,y, random_state = 56)
val_x_6.shape
val_y.shape
train_x_6.shape
train_y.shape

### E5. Implementing Linear Regression and Model Using XGBoost
#### **Setting the Model Learning Rate -aa**
learning_rate = 0.0005
#### **Fitting the Linear Regression Model over the Train Set** - with All Significant Features
# Creating instance of Linear Regresssion -ALL
lr = LR()

# Fitting the model
lr.fit(train_x_6, train_y)
train_y.mean()
### E6. Predicting over the **Normalized Train Set** and calculating error using **All features -LR**
# Predicting over the Normalized Train Set and calculating error
train_predict_6 = lr.predict(train_x_6)
k = mae(train_y, train_predict_6)
print('Training Mean Absolute Error', k )
#r = rmse(train_y, train_predict_6)
#print('Training Root Mean Sqaured Error', r )
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(train_y, train_predict_6)))
s = r2_score(train_y, train_predict_6)
print('Training R2 Score', s )
### E7. Comparing the results of LR with a Model using XGBoost
train_x_6.shape
#### **Fitting the XGBoost Model Over the Training Set**
train_x_6
import xgboost as xgb
model = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = learning_rate, max_depth = 10, n_estimators = 100)
model.fit(train_x_6, train_y)
train_predict_6 = model.predict(train_x_6)
result = model.score(train_x_6, train_predict_6)
print("Accuracy : {}".format(result))
### E8. Predicting over the **Normalized Train Set** and calculating error using **All features - XGBoost**
k = train_x_6.shape[1]
n = len(train_predict_6)
RMSE = float(format(np.sqrt(mean_squared_error(train_y, train_predict_6)),'.3f'))
MSE = mean_squared_error(train_y, train_predict_6)
MAE = mean_absolute_error(train_y, train_predict_6)
r2 = r2(train_y, train_predict_6)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2)
### E9. Predicting over the **Normalized Validation Set** and calculating error using **All Features and LR**.
# Predicting over the Normalized Test Set and calculating error
validation_predict_6 = lr.predict(val_x_6)
k = mae(val_y, validation_predict_6)
print('Validation Mean Absolute Error', k )
# r = rmse(val_y, validation_predict_6)
# print('Validation Root Mean Sqaured Error', r )
s= r2_score(val_y, validation_predict_6)
print('Validation R2 Score', s )
### E10. Predicting over the **Normalized Validation Set** and calculating error using **All Features and XGBoost**.
val_predict_6 = model.predict(val_x_6)
result_test = model.score(val_x_6, val_predict_6)
print("Accuracy : {}".format(result_test))
k = val_x_6.shape[1]
n = len(val_predict_6)
RMSE = float(format(np.sqrt(mean_squared_error(val_y, val_predict_6)),'.3f'))
MSE = mean_squared_error(val_y, val_predict_6)
MAE = mean_absolute_error(val_y, val_predict_6)
r2 = r2_score(val_y, val_predict_6)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2)
### E11. Parameters of Linear Regression
lr.coef_
### E11. Plotting the coefficients
plt.figure(figsize=(8, 6), dpi=120, facecolor='w', edgecolor='b')
x = range(len(train_x_6.columns))
y = lr.coef_
plt.bar( x, y )
plt.xlabel( "Variables")
plt.ylabel('Coefficients')
plt.title('Coefficient plot')

# df_minmax_lr_assumptions_checked_normalized = df_minmax_check_lr_assumptions.copy()
## F. Model Interpretability

### F. Arranging coefficients with features with Normalized Data
Coefficients = pd.DataFrame({
    'Variable'    : train_x_6.columns,
    'coefficient' : lr.coef_
})
Coefficients.head(1601).sort_values(ascending=False, by=['coefficient'])
train_x_6.shape
## G. Extracting variables with sigificance
#### **Extracting Features with significance greater than 0.04 (Filtering Significant Features)**
### Using Variance Inflation Factor
# Importing Variance_inflation_Factor funtion from the Statsmodels
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Calculating VIF for every column (only works for the not Catagorical)
VIF = pd.Series([variance_inflation_factor(train_x_6.values, i) for i in range(train_x_6.shape[1])], index =train_x_6.columns)
VIF
#### Feature with VIF > 5 indicates Multi-colinearity meaning the variable is highly correlated with some other variables in the set. So we should plan to remove multi-colinearity for improving model performance and predictions.
#### Extracting variables with sigificance greater than 5( Filtering Significant Features)
sig_var = Coefficients[Coefficients.coefficient >= 5]

for_export_to_train_model_2 = pd.DataFrame()
VIF = train_x_6[sig_var['Variable'].values].copy()
VIF
VIF_columns = VIF.columns.tolist()
for_export_to_train_model_2 = train_x_6.drop(VIF_columns, axis=1)
for_export_to_train_model_2
##  Predictions and Evaluation using LR and XGBoost for Dataset with 2 Significant Features.
train_x_2 = for_export_to_train_model_2.copy()
train_y
val_x_2 = val_x_6.drop(VIF_columns, axis=1)
val_x_2
val_y
#### G-2. Training Model  using Features of Significance from the Normalized Data Train Set
# Creating instance of Linear Regresssion with Normalised Data
lr_2 = LR()

# Fitting the model
lr_2.fit(train_x_2, train_y)
train_y.mean()
#### G-3. Predicting over the train set using the Normalized Training Data Feature  of Significance Set using LR
# Predicting over the Train Set and calculating error using Normalized Final Seceted Features
train_predict_2 = lr_2.predict(train_x_2)
k = mae(train_y, train_predict_2)
print('Training Mean Absolute Error', k )
# r = rmse(train_y, train_predict_2)
#print('Training Root Mean Sqaured Error', r )
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(train_y, train_predict_2)))
s = r2_score(train_y, train_predict_2)
print('Training R2 Score', s )
train_predict_2
train_predict_2.mean()
#### G-4. Predicting Over Normalized Training Features of Significance Dataset Using the XGBoost Model
model_2 = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = learning_rate, max_depth = 10, n_estimators = 100)
model_2.fit(train_x_2, train_y)
train_predict_2 = model_2.predict(train_x_2)
result_xgb = model_2.score(train_x_2, train_predict_2)
print("Accuracy : {}".format(result_xgb))
k = train_x_2.shape[1]
n = len(train_predict_2)
RMSE = float(format(np.sqrt(mean_squared_error(train_y, train_predict_2)),'.3f'))
MSE = mean_squared_error(train_y, train_predict_2)
MAE = mean_absolute_error(train_y, train_predict_2)
r2 = r2_score(train_y, train_predict_2)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2)
### G-5. Predicting over the validation set using the Normalized Significant Feature Validation Data Set Using LR
# Predicting over the Test Set and calculating error using Normalized Final Seceted Features
validation_predict_2 = lr_2.predict(val_x_2)
k = mae(val_y, validation_predict_2)
print('Validation Mean Absolute Error    ', k )
# r = rmse(val_y, validation_predict_2)
# print('Validation Root Mean Sqaured Error', r )
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(val_y, validation_predict_2)))
s = r2_score(val_y, validation_predict_2)
print('Validation R2 Score', s )
validation_predict_2
### G-6. Predicting Over the Normalized Significant Feature Validation Set Using the XGBoost Model
validation_predict_2 = model_2.predict(val_x_2)
result_validation = model_2.score(val_x_2, validation_predict_2)
print("Accuracy : {}".format(result_validation))
k = val_x_2.shape[1]
n = len(validation_predict_2)
RMSE = float(format(np.sqrt(mean_squared_error(val_y, validation_predict_2)),'.3f'))
MSE = mean_squared_error(val_y, validation_predict_2)
MAE = mean_absolute_error(val_y, validation_predict_2)
r2 = r2_score(val_y, validation_predict_2)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2)
validation_predict_2
### G-7. Plotting the coefficients of LR
plt.figure(figsize=(8, 6), dpi=120, facecolor='w', edgecolor='b')
x = range(len(train_x_2.columns))
y = lr_2.coef_
plt.bar( x, y )
plt.xlabel( "Variables")
plt.ylabel('Coefficients')
plt.title('Normalized Coefficient plot')
---------------------------------
## F. Generate a CSV File with Scaled and Normalized Data with Selected Features for Training the Model Using Neural Network
## Exporting Dataset with significant Features For NN Training
### Exporting Dataset ALL Features For NN Training
### Exporting the Normalized ALL features for Model Training.
for_export_to_train_model_6 = pd.DataFrame()
for_export_to_train_model_6 = x_6.copy()
for_export_to_train_model_6.shape
for_export_to_train_model_6
#for_export_to_train_model_6.insert(0, 'Item_Identifier', data['Item_Identifier'])
#for_export_to_train_model_6.insert(1, 'Outet_Identifier', data['Outlet_Identifier'])
#for_export_to_train_model_6.insert(1601, 'Item_Outlet_Sales', df_minmax_check_lr_assumptions['Item_Outlet_Sales'])
for_export_to_train_model_6
### Exporting Dataset with Features having Significance level> threshold For NN Training
for_export_to_train_model_2
# for_export_to_train_model_2.insert(0, 'Item_Identifier', df_encoded['Item_Identifier'])
# for_export_to_train_model_2.insert(1, 'Outet_Identifier', df_encoded['Outlet_Identifier'])
# for_export_to_train_model_2.insert(47, 'Item_Outlet_Sales', df_minmax_check_lr_assumptions['Item_Outlet_Sales'])
# for_export_to_train_model_2

# Create an export the extracted normalized features for Model Training Using Neural Network. - 6 Features
for_export_to_train_model_6.to_csv('bigmart_outlet_sales_prediction_minmax_data_selected_features_for_model_training_6_features.csv', index=False)
# Create an export the extracted normalized features for Model Training Using Neural Networ.
for_export_to_train_model_2.to_csv('bigmart_outlet_sales_prediction_minmax_data_selected_features_for_model_training_2_features.csv', index=False)
# **Load the Test data file and Preprocess it for Prediction Generation and Evaluation**
# loading the test dataset
data_test = pd.read_csv('/content/sample_data/test_FewQE9B.csv')

#data_test = pd.read_csv('test_FewQE9B.csv')
data_test.shape
## 1. Selecting Features to Match the Training Set and Creating a New Subset
### 1-a. Creating a subset with 6 significant features
#seperating independent and dependent variables
#x_t_6 = data_test.drop(['Item_Identifier', 'Item_Weight', 'Item_Type', 'Outlet_Identifier', 'Outlet_Establishment_Year'], axis=1)
x_t_6 = data_test.copy()
x_t_6
## 2. Variable Type Identification and Conversion
x_t_6.dtypes
x_t_6.shape
## 2-b. Converting categories to numbers since our model only processes numerical data.
### a. Identifying Unique Values in Categories
# identififying unique category values for the category variables

item_identifier_unique_categories = x_t_6['Item_Identifier'].unique()
item_identifier_unique_categories.shape
x_t_6['Item_Fat_Content'].unique()
item_type_unique_categories = x_t_6['Item_Type'].unique()
item_type_unique_categories.shape
outlet_identifier_unique_categories = x_t_6['Outlet_Identifier'].unique()
outlet_identifier_unique_categories.shape
outlet_size_unique_categories = x_t_6['Outlet_Size'].unique()
outlet_size_unique_categories.shape

outlet_location_type_unique_categories = x_t_6['Outlet_Location_Type'].unique()
outlet_location_type_unique_categories.shape
outlet_type_unique_categories = x_t_6['Outlet_Type'].unique()
outlet_type_unique_categories.shape
outlet_establishment_year_unique_categories = x_t_6['Outlet_Establishment_Year'].unique()
outlet_establishment_year_unique_categories.shape
item_identifier_unique_categories = x_t_6['Item_Identifier'].unique()
item_identifier_unique_categories.shape
x_t_6.dtypes
# Identifying variables with object datatype
x_t_6.dtypes[x_t_6.dtypes == 'object']
#Converting object data type to category
x_t_6['Item_Identifier'] = x_t_6['Item_Identifier'].astype('category')
x_t_6['Item_Fat_Content'] = x_t_6['Item_Fat_Content'].astype('category')
x_t_6['Item_Type'] = x_t_6['Item_Type'].astype('category')
x_t_6['Outlet_Identifier'] = x_t_6['Outlet_Identifier'].astype('category')
x_t_6['Outlet_Size'] = x_t_6['Outlet_Size'].astype('category')
x_t_6['Outlet_Location_Type'] = x_t_6['Outlet_Location_Type'].astype('category')
x_t_6['Outlet_Type'] = x_t_6['Outlet_Type'].astype('category')

# check the updated datatypes.
x_t_6.dtypes
# isolating categorical datatypes
categorical = x_t_6.select_dtypes(include=['category'])[:]
categorical.dtypes
### 2-c. Check for missing values
x_t_6.isnull().sum()
x_t_6.dtypes
### 2-d. Fill the missing values in the Test Data set.
# filling missing values of continuous variables with mean
x_t_6['Item_Weight'].fillna(x_t_6['Item_Weight'].mean(), inplace=True)
# filling missing values of categorical variables with mode
x_t_6['Outlet_Size'].fillna(x_t_6['Outlet_Size'].mode()[0], inplace=True)


x_t_6.isnull().sum()
## 3. Identifying Outliers and Skew in the variables.
bigmart_numerical_test = x_t_6.select_dtypes(include=['int64','float64','Int32'])[:]
bigmart_details_numerical_test_values = bigmart_numerical_test.columns.tolist()

bigmart_numerical_test
UVA_numeric(x_t_6, bigmart_details_numerical_test_values[0:4], 6)
### **From above we observe that the bias and the outliers in the test dataset variables are in the excellent to acceptable range for normal distribution. Therefore no outliers need to be removed.**
## 6. Normalizing and Encoding categories
# # Creating list of unique categories for encoding
# item_fat_unique_cat_list = ['Low Fat', 'Regular']
# outlet_size_unique_cat_list = ['Small', 'Medium', 'High']
# outlet_type_unique_cat_list =['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3']
# outlet_location_type_unique_cat_list = ['Tier 1', 'Tier 2', 'Tier 3']
# #item_type_unique_cat_list = ['Baking Goods', 'Breads', 'Breakfast', 'Canned', 'Dairy', 'Frozen Foods', 'Fruits and Vegetables', 'Hard Drinks', 'Health and Hygiene', 'Household', 'Meat', 'Others', 'Seafood', 'Snack Foods', 'Soft Drinks', 'Starchy Foods']
# #outlet_identified_unique_cat_list = ['OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019', 'OUT027', 'OUT035', 'OUT045', 'OUT046', 'OUT049']

item_fat_content_unique_categories = x_t_6['Item_Type'].unique()
item_fat_content_unique_categories.categories
item_outlet_identifier_unique_categories = x_t_6['Outlet_Identifier'].unique()
item_outlet_identifier_unique_categories.categories
item_outlet_type_unique_categories = x_t_6['Outlet_Type'].unique()
item_outlet_type_unique_categories
item_outlet_established_year_unique_categories = x_t_6['Outlet_Establishment_Year'].unique()
item_outlet_established_year_unique_categories
# Normalizing categories
x_t_6['Item_Fat_Content'] = x_t_6['Item_Fat_Content'].map({'LF':'Low Fat', 'low fat': 'Low Fat', 'Low Fat':'Low Fat','reg':'Regular', 'Regular':'Regular'})
x_t_6['Item_Fat_Content'] = x_t_6['Item_Fat_Content'].astype('category')
x_t_6['Item_Fat_Content']
x_t_6['Outlet_Establishment_Year'].value_counts()
x_t_6.dtypes
#### 6-a. Normalizing Variables and Label Encoding
# converting the categories into numbers using map function
lblencoder = LabelEncoder()
ohencoder = OneHotEncoder()


x_t_6['Item_Fat_Content'] = lblencoder.fit_transform(x_t_6['Item_Fat_Content'])

x_t_6['Outlet_Size'] = lblencoder.fit_transform(x_t_6['Outlet_Size'])

x_t_6['Outlet_Type'] = lblencoder.fit_transform(x_t_6['Outlet_Type'])

x_t_6['Outlet_Location_Type'] = lblencoder.fit_transform(x_t_6['Outlet_Location_Type'])

x_t_6.head()
x_t_6.dtypes
x_t_6.shape
# Drop the Item_Identifier to match the Training ste used to fit the model.
x_t_6 = x_t_6.drop('Item_Identifier', axis=1)
#x_t_6 = x_t_6.drop('Outlet_Identifier', axis=1)
x_t_6.columns
#### 6-b. One Hot Encoding - convert categorical values to numerical
x_t_6['Outlet_Establishment_Year'] = x_t_6['Outlet_Establishment_Year'].astype('category')
test_categorical_columns_all = x_t_6.select_dtypes(include=['category']).columns.tolist()
test_categorical_columns_all
df_test_encoded_all = encode_categories(x_t_6, test_categorical_columns_all, "Bigmart All Features Test Dataset")
### Checking if there are any remaining null values.
df_test_encoded_all.isnull().sum()
### 6-c. Isolating numerical datatypes
# Isolating numerical types for analysis
bigmart_numerical = df_test_encoded_all.select_dtypes(include=['int64','float64','Int32'])[:]
bigmart_details_numerical_values = bigmart_numerical.columns.tolist()

# Item_Identfier List - segregating numerica variables in ggroups for convinience, faster compute, and analysis.
bigmart_details_numerical_values_1_16 = bigmart_numerical.columns[:16].tolist()
bigmart_details_numerical_values_1_16
# Outlet_Identifier List -segregating numerica variables in ggroups for convinience, faster compute, and analysis.
bigmart_details_numerical_values_16_26 = bigmart_numerical.columns[16:26].tolist()
bigmart_details_numerical_values_16_26
# Item_Type List-segregating numerica variables in ggroups for convinience, faster compute, and analysis.
bigmart_details_numerical_values_1575_1585 = bigmart_numerical.columns[26:38].tolist()
bigmart_details_numerical_values_1575_1585
# Outlet_Establishment Year List -segregating numerica variables in ggroups for convinience, faster compute, and analysis.
bigmart_details_numerical_values_1585_1594 = bigmart_numerical.columns[1585:1594].tolist()
#bigmart_details_numerical_values_1585_1594
# Outlet_Establishment Year List -segregating numerica variables in ggroups for convinience, faster compute, and analysis.
bigmart_details_numerical_values_1594_1602 = bigmart_numerical.columns[1594:1602].tolist()
#bigmart_details_numerical_values_1594_1602
# x_t_6['Item_Fat_Content'] = x_t_6['Item_Fat_Content'].astype('category')
# x_t_6['Outlet_Size'] = x_t_6['Outlet_Size'].astype('category')
# x_t_6['Outlet_Location_Type'] = x_t_6['Outlet_Location_Type'].astype('category')
# x_t_6['Outlet_Type'] = x_t_6['Outlet_Type'].astype('category')

## 7. Selecting Features to Match the Train Data Set
len(df_test_encoded_all)
df_test_encoded_all.describe()
## 7. Scaling Test Data
scaler = preprocessing.MinMaxScaler()
minmax_t = scaler.fit_transform(df_test_encoded_all)
minmax_t = pd.DataFrame(minmax_t, columns=df_test_encoded_all.columns.to_list())

scaler = preprocessing.StandardScaler()
standard_t = scaler.fit_transform(df_test_encoded_all)
standard_t = pd.DataFrame(standard_t, columns=df_test_encoded_all.columns.to_list())

scaler = preprocessing.RobustScaler()
robust_t = scaler.fit_transform(df_test_encoded_all)
robust_t = pd.DataFrame(robust_t, columns=df_test_encoded_all.columns.to_list())
minmax_all.shape
minmax_t.head()
minmax_t.shape
print('*' * 45)
print('\033[1m'+'With minmax scaling:'+'\033[0m')
display(minmax_t.head())
print('*' * 45)
print('\033[1m'+'With standard scaling:'+'\033[0m')
display(standard_t.head())
print('*' * 45 )
print('\033[1m'+'With robust scaling:'+'\033[0m')
display(robust_t.head())
minmax_all.shape
### 7-b. Segregating the Independent and Dependent Variables
# Scaled
minmax_t_X = minmax_t
## 8.Predicting over the Normalized Test Set and Calculating Errors Using LR.
### 8-a.Predictions for Test Dataset with All Features using Linear Regression
test_predict_6 = lr.predict(minmax_t_X)  # We are using the previously fitted model to predict target values for the test dataset using training set with all features.
test_predict_6.shape
test_predict_6.mean()
train_y.shape
train_y_sh = train_y.head(5681)
train_y_sh.shape   # From the previous spit of the Train set.
k = mae(train_y_sh, test_predict_6)
print('Test Mean Absolute Error    ', k )
# r = rmse(train_y, test_predict_6_sh)
# print('Test Root Mean Sqaured Error', r )
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(train_y_sh, test_predict_6)))
s = r2_score(train_y_sh, test_predict_6)
print('Test R2 Score', s ) # R2 Score Indicates the strenght of relationship between the dependent and the independent variables. Higher scores indicate good fit with the data.
# It is the % of variance in the dependent variable is explained by the independent variables
#### Note: _ve values of R2 score indicate the model average values perform better thna the predicted values.
### 8-b. Predicting Over the Normalized and Scaled Features Test Set Using the XGBoost Model - All Features
test_predict_t_6 = model.predict(minmax_t_X)
result_validation = model.score(minmax_t_X, test_predict_6)
print("Accuracy : {}".format(result_validation))
test_predict_t_6.shape
train_y.shape
y_train_a.shape
y_train_a_mean = y_train_a.mean()
y_train_a_mean
y_train_a_sh = y_train_a.head(5681)
k = minmax_t.shape[1]
n = len(test_predict_t_6)
RMSE = float(format(np.sqrt(mean_squared_error(y_train_a_sh, test_predict_t_6)),'.3f'))
MSE = mean_squared_error(y_train_a_sh, test_predict_t_6)
MAE = mean_absolute_error(y_train_a_sh, test_predict_t_6)
r2 = r2_score(y_train_a_sh, test_predict_t_6)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2)
test_predict_t_6_mean = test_predict_t_6.mean()
test_predict_t_6_mean
#### Negative R2 indicates that the mean value of the target variable in the training set performs better than the predicted model value. The target variable mean i.e y_train_a_mean is 0.16455128126640875.
test_predict_t_6
### 8-b. Predicting Over the Normalized Features Test Set Using the LR Model - Features of Significance
test_x_2 = pd.DataFrame

# test_x_2.drop(['Item_Fat_Content', 'Item_Visibility', 'Outlet_Size', 'Outlet_Location_Type'], axis=1, inplace=True)
test_x_2 = minmax_t_X.drop(VIF_columns, axis=1)
test_x_2.columns
test_predict_lr_2 = lr_2.predict(test_x_2)  # We are using the previously fitted model to predict target values for the test dataset using training set with two features.
test_predict_lr_2.mean()
test_predict_lr_2.shape
y.shape
k = mae(y_train_a_sh, test_predict_lr_2)
print('Test Mean Absolute Error    ', k )
#r = rmse(y_train_a, test_predict_lr_2)
# print('Test Root Mean Sqaured Error', r )
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train_a_sh, test_predict_lr_2)))
s = r2_score(y_train_a_sh, test_predict_lr_2)
print('Test R2 Score', s ) # R2 Score Indicates the strenght of relationship between the dependent and the independent variables. Higher scores indicate good fit with the data.
# It is the % of variance in the dependent variable is explained by the independent variables
### 8-c. Predicting Over the Normalized Features Test Set Using the XGBoost Model - 2 Features
test_predict_t_2 = model_2.predict(test_x_2)
result_validation = model_2.score(test_x_2, test_predict_t_2)
print("Accuracy : {}".format(result_validation))
k = test_x_2.shape[1]
n = len(test_predict_t_2)
RMSE = float(format(np.sqrt(mean_squared_error(y_train_a_sh, test_predict_t_2)),'.3f'))
MSE = mean_squared_error(y_train_a_sh, test_predict_t_2)
MAE = mean_absolute_error(y_train_a_sh, test_predict_t_2)
r2 = r2_score(y_train_a_sh, test_predict_t_2)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2)
test_predict_t_2
test_predict_t_2.mean()
df_test_predict= pd.DataFrame()
df_test_predict['Item_Outlet_Sales']= test_predict_t_6
df_test_predict
## Export The Scaled and Normalized Test Data Using MinMax Scaler to Generate Item Outlet Sales Predictions using Neural Network.
minmax_t
minmax_t.shape
test_x_2.shape
data_test.shape
#minmax_t.insert(42, 'Item_Outlet_Sales', test_predict_t_6)
### Preparing The Evaluation Test Data Set for the Neural Network Model
 minmax_t.insert(0, 'Item_Identifier', data_test['Item_Identifier'])
 minmax_t.insert(1, 'Outlet_Identifier', data_test['Outlet_Identifier'])
 minmax_t.insert(42, 'Item_Outlet_Sales_Actual', data['Item_Outlet_Sales'])

 test_x_2.insert(0, 'Item_Identifier', data_test['Item_Identifier'])
 test_x_2.insert(1, 'Outlet_Identifier', data_test['Outlet_Identifier'])
 test_x_2.insert(30, 'Item_Outlet_Sales_Actual', data['Item_Outlet_Sales'])

minmax_t.to_csv('bigmart_outlet_sales_test_data_selected_features_for_modelevaluation_all_features.csv', index=False)
test_x_2.to_csv('bigmart_outlet_sales_test_data_selected_features_for_modelevaluation_significant_features.csv', index=False)
# **Model Training using Neural Network using Keras**
## Steps to build a Neural Network using Keras

<ol>1. Loading the dataset</ol>
<ol>2. Creating training and validation set</ol>
<ol>3. Defining the architecture of the model</ol>
<ol>4. Compiling the model (defining loss function, optimizer)</ol>
<ol>5. Training the model</ol>
<ol>6. Evaluating model performance on training and validation set</ol>
## 1. Loading the dataset
# importing the required libraries
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import metrics
# check version on sklearn
print('Version of sklearn:', sklearn.__version__)
# loading the pre-processed dataset
data_t = pd.read_csv('/content/bigmart_outlet_sales_test_data_selected_features_for_modelevaluation_all_features.csv')
#data_t = pd.read_csv('bigmart_outlet_sales_test_data_selected_features_for_modelevaluation_all_features.csv')
# looking at the first five rows of the dataset
data_t.head()
# checking missing values
data_t.isnull().sum()
# checking the data type
data_t.dtypes
# looking at the shape of the data
data_t.shape
# backing up the Item_Identifier and Outlet_Identifier
df_backup_train_t = pd.DataFrame()
df_backup_train_t = data_t.copy()
#df_backup_train['Outlet_Identifier'] = data['Outlet_Identifier'].copy()
data_t.drop('Item_Identifier', axis=1, inplace=True)
data_t.drop('Outlet_Identifier', axis=1, inplace=True)
data_t.shape
data_t.dtypes
#data['Item_Outlet_Sales']
# separating the independent and dependent variables

# storing the dependent variable as y
#y = data['Item_Outlet_Sales']
#df_backup_train['Item_Outlet_Sales'] = data['Item_Outlet_Sales'].copy()

# storing all the independent variables as X
X = data_t.copy()
#X = data_t.drop('Item_Outlet_Sales', axis=1)

y = y_train_a_sh
# shape of independent and dependent variables
X.shape, y.shape
## 2. Creating training and validation set
# Creating training and validation set

# stratify will make sure that the distribution of classes in train and validation set it similar
# random state to regenerate the same train and validation set
# test size 0.2 will keep 20% data in validation and remaining 80% in train set

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=8,test_size=0.2)
# shape of training and validation set
(X_train.shape, y_train.shape), (X_test.shape, y_test.shape)
(X_train.size, y_train.size), (X_test.size, y_test.size)
## 3. Defining the architecture of the model
# checking the version of keras
import keras
print(keras.__version__)
# checking the version of tensorflow
import tensorflow as tf
print(tf.__version__)
### a. Create a model
# importing the sequential model
from keras.models import Sequential
### b. Defining different layers
# importing different layers from keras
from keras.layers import InputLayer, Dense

# number of input neurons
X_train.shape
# number of features in the data
X_train.shape[1]
# defining input neurons  - Number of input Neurons is equal to the number of features in the input data.
input_neurons = X_train.shape[1]
# number of output neurons

# since bigmart outlet sales prediction is a regression problem, we will have single neuron in the output layer
# define number of output neurons
output_neurons = 1
# number of hidden layers and hidden neurons

# It is a hyperparameter and we can pick the hidden layers and hidden neurons on our own
# define hidden layers and neuron in each layer
number_of_hidden_layers = 2
neuron_hidden_layer_1 = 10
neuron_hidden_layer_2 = 1
# activation function of different layers

# for now we have picked relu as an activation function for hidden layers, we can change it as well
# since it is a regression problem, we have used linear activation function in the final layer
# defining the architecture of the model
model = Sequential()
model.add(InputLayer(shape=(input_neurons,)))
model.add(Dense(units=neuron_hidden_layer_1, activation='relu'))
model.add(Dense(units=neuron_hidden_layer_2, activation='relu'))
model.add(Dense(units=output_neurons, activation='linear'))
# summary of the model
model.summary()
# number of parameters between input and first hidden layer

input_neurons*neuron_hidden_layer_1
# number of parameters between input and first hidden layer

# adding the bias for each neuron of first hidden layer

input_neurons*neuron_hidden_layer_1 + 10
# number of parameters between first and second hidden layer

neuron_hidden_layer_1*neuron_hidden_layer_2 + 5
# number of parameters between second hidden and output layer

neuron_hidden_layer_2*output_neurons + 1
X_train.shape
X_train.size
xtrain= X_train.head(1704)
xtrain.size
xval = X_test.head(1704)
xval.size
#ytrain = y_train.head(6816)
ytrain = y_train.head(1704)
ytrain.size
yval = y_test.head(1704)
yval.size
## 4. Compiling the model (defining loss function, optimizer)
# compiling the model

# loss as regression, since we have regression problem
# defining the optimizer as adam
# Evaluation metric as rmse

#model.compile(loss='mse',optimizer='Adam',metrics=['rmse'])
#MODEL_1
# model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
#               loss='mse', metrics=[keras.metrics.RootMeanSquaredError()])

#MODEL_2
model.compile(optimizer=tf.keras.optimizers.Adam(
    #learning_rate=0.044,
   ## learning_rate=0.054,
    learning_rate=learning_rate,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8), loss='mean_squared_error', metrics=[keras.metrics.RootMeanSquaredError(), keras.metrics.R2Score()])
ytrain
## 5. Training the model
# training the model

# passing the independent and dependent features for training set for training the model

# validation data will be evaluated at the end of each epoch

# setting the epochs as 50

# storing the trained model in model_history variable which will be used to visualize the training process

model_history = model.fit(xtrain, ytrain, validation_data=(xval, yval), batch_size=16, epochs=50)

## 6. Generate Predictions Using the Validation Set
# getting predictions for the Validation set
predictions_validationset = {}
predictions_validationset_all = model.predict(xval)
# Predicted Item_Outlet_Sales for the Validation Set
predictions_validationset_all

yval_1704  = yval.head(1704)
yval_1704.shape, yval_1704.size
predictions_validationset_all.shape, predictions_validationset_all.size
yval_1704, predictions_validationset_all
# list all data in history
print(model_history.history.keys())
# Evaluate and Save the Model -Validation
# Evaluate the model
val_loss, val_metric, val_metric_1 = model.evaluate(X_test, y_test, verbose=2)

# Save Model Data
#Saving the results
save_model = pd.DataFrame()
bigmart_model_predictions_test_results = {}

#MODEL_1
# bigmart_model_predictions_test_results['model_definition_1']="optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),loss='mse', metrics=[keras.metrics.RootMeanSquaredError()]"
# bigmart_model_predictions_test_results['model_summary_1_validationset'] = model.summary()
# bigmart_model_predictions_test_results['validation_set_predictions_1']= predictions_validationset
# bigmart_model_predictions_test_results['validation_set_model_evalutation_1']= loss, metric

# save_model['model_1'] = bigmart_model_predictions_test_results
# model.save("save_model['model_1'].keras")
# save_model.to_csv('bigmart_outlet_sales_prediction_minmax_rmprop_rmse_valset_model_1.csv', index=False)



#MODEL_2
bigmart_model_predictions_test_results['model_definition_adam_all']="optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate,beta_1=0.9, beta_2=0.999,epsilon=1e-8,decay=0.0), loss='mse', metrics=[keras.metrics.RootMeanSquaredError()]]"
bigmart_model_predictions_test_results['model_summary_2_validationset_adam_all'] = model.summary()
bigmart_model_predictions_test_results['validation_set_predictions_adam_all']= predictions_validationset
bigmart_model_predictions_test_results['validation_set_model_evalutation_adam_all']= val_loss, val_metric

save_model['model_adam_all'] = bigmart_model_predictions_test_results

model.save("save_model['model_adam_all'].keras")
save_model.to_csv('bigmart_outlet_sales_prediction_minmax_adam_rmse_valset_model_adam_all.csv', index=False)


# Train Data Set Item_Outlet_Sales
#y_test
## Visualizing Model Performance for Validation Set
# Summerize Loss and Validation Loss
def plot_metric(history, valtest, metricname, metric):
    plt.figure(figsize=(20,5))
    plt.plot(history.history[metricname], 'g', label='Training '+metric)
    plt.plot(history.history['val_'+metricname], 'b', label=valtest+' '+metric)
    plt.xlim([0, 50])
    plt.ylim([0, 0.5])
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.legend()
    plt.title('Model Metric')
    plt.grid(True)
plot_metric(model_history, "Validation", "root_mean_squared_error", "RMSE")
plot_metric(model_history, "Validation", "loss", "Loss")
# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# Generate feature data that spans the range of interest for the independent variable.
x_val = X_test['Outlet_Type']

# Use the model to predict the dependent variable using one independent variable.
y_val = model.predict(X_test)
x_val.shape
y_val.shape
y_val.size
x_val.size
xtrain.shape
xtrain.size
xtrain_sh = xtrain.head(39)
xtrain_sh.size
ytrain.size
ytrain.size
ytrain_sh = ytrain.head(1677)
ytrain_sh.size
def plot_data(x_data, y_data, x, y, title=None):

    plt.figure(figsize=(15,5))
    plt.scatter(x_data, y_data, label='Ground Truth', color='green', alpha=0.5)
    plt.plot(x, y, color='k', label='Model Predictions')
    plt.xlim([0,2])
    plt.ylim([0,1])
    plt.xlabel('Outlet Type')
    plt.ylabel('Outlet Sales')
    plt.title(title)
    plt.grid(True)
    plt.legend()
plot_data(xtrain_sh, ytrain_sh, x_val, y_val, title='')
## ------
# 7. Now making the Predictions using Test Dataset and Evaluating Model Performance
## a. Load the Scaled and Normalized Test Dataset Using MinMax Scaler
## loading the pre-processed dataset
data_test_minmax = pd.read_csv('/content/bigmart_outlet_sales_test_data_selected_features_for_modelevaluation_all_features.csv')

data_test_minmax_selected = pd.read_csv('/content/bigmart_outlet_sales_test_data_selected_features_for_modelevaluation_significant_features.csv')
data_test_minmax.head(5)
data_test_minmax_selected.head(5)
bkp_data_test_minmax_Item_Identifier = data_test_minmax['Item_Identifier'].copy()
bkp_data_test_minmax_Outlet_Identifier = data_test_minmax['Outlet_Identifier'].copy()

bkp_data_test_minmax_selected_Item_Identifier = data_test_minmax_selected['Item_Identifier'].copy()
bkp_data_test_minmax_selected_Outlet_Identifier = data_test_minmax_selected['Outlet_Identifier'].copy()
bkp_data_test_minmax_selected_Outlet_Item_Outlet_Sales_Actual = data_test_minmax_selected['Item_Outlet_Sales_Actual'].copy()
data_test_minmax.drop('Item_Identifier', axis=1, inplace=True)
data_test_minmax.drop('Outlet_Identifier', axis=1, inplace=True)

data_test_minmax_selected.drop('Item_Identifier', axis=1, inplace=True)
data_test_minmax_selected.drop('Outlet_Identifier', axis=1, inplace=True)

data_test_minmax_selected.drop('Item_Outlet_Sales_Actual', axis=1, inplace=True)
## b. Generate Predictions Using the Test data Set
data_test_minmax_X = data_test_minmax
data_test_minmax_y = y_train_a
data_test_minmax_X.shape
y_train_a.shape
data_test_minmax_selected_X = data_test_minmax_selected
data_test_minmax_selected_y = y_train_a
data_test_minmax_selected_X.shape
data_test_minmax_selected_X.dtypes
# getting predictions for the  set using all test features
predictions_testset_all = model.predict(data_test_minmax_X)
# getting predictions for the  set using significant features
predictions_testset_selected = model_2.predict(data_test_minmax_selected_X)
predictions_testset_all
predictions_testset_all.shape
predictions_testset_selected
predictions_testset_all.mean()
# list all data in history
print(model_history.history.keys())
## c. Evaluate and Save the Model Using Test Data
# Evaluate the model
#loss_test, metric_test, metric_test_1 = model.evaluate(data_test_minmax_X, predictions_testset_all, verbose=2)

loss_test, metric_test, metric_test_1 = model.evaluate(data_test_minmax_X, y_train_a_sh, verbose=2)
# Save Model Data-test (Model_1)
# bigmart_model_predictions_test_results['model_definition_1']="optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),loss='mse', metrics=[keras.metrics.RootMeanSquaredError()]"
# bigmart_model_predictions_test_results['model_summary_1_testset'] = model.summary()
# bigmart_model_predictions_test_results['test_set_predictions_1']= predictions_testset_all
# bigmart_model_predictions_test_results['test_set_model_evalutation_1']= loss, metric

# save_model['model_1'] = bigmart_model_predictions_test_results

# model.save("save_model['model_1'].keras")
# save_model.to_csv('bigmart_outlet_sales_prediction_minmax_rmprop_rmse_model_1_test.csv', index=False)

# MODEL_2
bigmart_model_predictions_test_results['model_definition_test_adam_all']="optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate,beta_1=0.9, beta_2=0.999,epsilon=1e-8,decay=0.0), loss='mse', metrics=[keras.metrics.RootMeanSquaredError()]]"
bigmart_model_predictions_test_results['model_summary_2_testset_adam_all'] = model.summary()
bigmart_model_predictions_test_results['test_set_predictions_adam_all']= predictions_testset_all
bigmart_model_predictions_test_results['test_set_model_evalutation_adam_all']= loss_test, metric_test

save_model['model_2_adam_all'] = bigmart_model_predictions_test_results

model.save("save_model['model_adam_all'].keras")
save_model.to_csv('bigmart_outlet_sales_prediction_minmax_adam_rmse_model_adam_all_test.csv', index=False)

# Evaluate the model
#loss_test, metric_test, metric_test_1 = model.evaluate(data_test_minmax_selected_X, y_train_a_sh, verbose=2)
# Save Model Data-test (Model_1)
# bigmart_model_predictions_test_results['model_definition_1']="optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),loss='mse', metrics=[keras.metrics.RootMeanSquaredError()]"
# bigmart_model_predictions_test_results['model_summary_1_testset'] = model.summary()
# bigmart_model_predictions_test_results['test_set_predictions_1']= predictions_testset_all
# bigmart_model_predictions_test_results['test_set_model_evalutation_1']= loss, metric

# save_model['model_1'] = bigmart_model_predictions_test_results

# model.save("save_model['model_1'].keras")
# save_model.to_csv('bigmart_outlet_sales_prediction_minmax_rmprop_rmse_model_1_test.csv', index=False)

# MODEL_2
# bigmart_model_predictions_test_results['model_definition_test_adam_selected']="optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate,beta_1=0.9, beta_2=0.999,epsilon=1e-8,decay=0.0), loss='mse', metrics=[keras.metrics.RootMeanSquaredError()]]"
# bigmart_model_predictions_test_results['model_summary_2_testset_adam_selected'] = model.summary()
# bigmart_model_predictions_test_results['test_set_predictions_adam_selected']= predictions_testset_all
# bigmart_model_predictions_test_results['test_set_model_evalutation_adam_selected']= loss_test, metric_test

# save_model['model_2_adam_selected'] = bigmart_model_predictions_test_results

# model.save("save_model['model_adam_selected'].keras")
# save_model.to_csv('bigmart_outlet_sales_prediction_minmax_adam_rmse_model_adam_selected_test.csv', index=False)

# Train Data Set Item_Outlet_Sales
#y_test
plot_metric(model_history,"Test","root_mean_squared_error", "RMSE")
plot_metric(model_history,"Test","loss", "Loss")
plot_metric(model_history,"Test","r2_score", "R2 Score")
# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
## 8. Generation of Submission Data For Evaluation
#from sklearn import
from sklearn import preprocessing
scaler_te = preprocessing.MinMaxScaler()
minmax_item_outlet_sales = scaler_te.fit_transform(data[['Item_Outlet_Sales']][:5681])


minmax_item_outlet_sales = minmax_item_outlet_sales.astype('float32')
minmax_item_outlet_sales.shape
predictions_testset_all
predictions_testset_all.mean()
predictions_testset_all.shape
# creating a list of index names
index_values = ['Item_Outlet_Sales']

# creating a list of column names
column_values = ['Item_Outlet_Sales']

# creating the dataframe
df_predictions_testset_all = pd.DataFrame(data = predictions_testset_all,
                  columns = column_values)

# displaying the dataframe
print(df_predictions_testset_all)
#df_predictions_testset_all =pd.DataFrame({'Item_Outlet_Sales':predictions_testset_all})

#df_predictions_testset_all.shape
submission_data_output_scaled = pd.DataFrame()

submission_data_output_scaled.insert(0, 'Item_Identifier', bkp_data_test_minmax_Item_Identifier)
submission_data_output_scaled.insert(1, 'Outlet_Identifier', bkp_data_test_minmax_Outlet_Identifier)
submission_data_output_scaled.insert(2, 'Item_Outlet_Sales', predictions_testset_all)



submission_data_output_unscaled_all = scaler_te.inverse_transform(df_predictions_testset_all)
submission_data_output_unscaled_all
# summarize history for accuracy
plt.plot(data['Item_Outlet_Sales'][:5681])
plt.plot(submission_data_output_unscaled_all)
plt.title('Item Outlet Sales')
plt.ylabel('Predicted Item Outlet Sales')
plt.xlabel('Actual Item Outlet Sales')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
submission_data_output_unscaled = pd.DataFrame()

submission_data_output_unscaled.insert(0, 'Item_Identifier', bkp_data_test_minmax_Item_Identifier)
submission_data_output_unscaled.insert(1, 'Outlet_Identifier', bkp_data_test_minmax_Outlet_Identifier)
submission_data_output_unscaled.insert(2, 'Item_Outlet_Sales', submission_data_output_unscaled_all)
#Exporting a Data File for Submission.
#submission_data_unscaled_output = submission_data_output.to_csv('bigmart_submission_minmax_rmsprop_rmse_submission_data_1.csv')
submission_data_output_scaled_1 = submission_data_output_scaled.to_csv('bigmart_submission_minmax_adam_lr0.005_b16-e-30loss_rmse_submission_data_all_scaled.csv')

submission_data_output_unscaled_1 = submission_data_output_unscaled.to_csv('bigmart_submission_minmax_adam_lr0.005_b16-e-30loss_rmse_submission_data_all_unscaled.csv')


# **Summary of Results using LR, XGBoost, and Neural Network Model.**
- Using different Learning Rates.
- Using All and Features of Significance.
- With and Without Outliers
#**Learning Rate: 0.0005**

## **ALL Features**
### **TRAIN SET**
	- Actual Train Mean = 0.1642970077485156

  - **LR:**
  	- Training Mean Absolute Error 0.06297834494117917
	  - Training Root Mean Squared Error: 0.1613450605731818
	  - Training R2 Score 0.5709287225489612

  -**XGB:**
	- Accuracy : 1.0
	- RMSE = 0.125
	- MSE = 0.015639718704262152
	- MAE = 0.09902603083559106
	- R2 = 0.06279628783670987
	- Adjusted R2 = 0.05659648378711801


### **TEST SET**
  - **LR:**
 	- Test Mean Absolute Error     0.1278423066099372
 	- Test R2 Score -0.5561919388693397

  - **XGB:**
  	- Accuracy : 0.7480227791850307
  	- RMSE = 0.143
  	- MSE = 0.020395422400204342
  	- MAE = 0.11135060476335988
  	- R2 = -0.19879435209649343
  	- Adjusted R2 = -0.20772471087408362
	  - **Predicted_Test_Mean: 0.16353457**
  

  - **NN:Adam:**    
   - loss: 0.0172
	- r2_score: -8.3355e-03
	- root_mean_squared_error: 0.1310
  
   - **Predicted_Test_Mean: 0.16386086**

---------------------------------------------------------

##**Features of SIGNIFICANCE**

###- **TRAINING**

 	- Actual Train Mean: 0.1642970077485156

   - **LR:**
 	- Training Mean Absolute Error 0.07204315868387007
 	- Root Mean Squared Error: 0.0982837111471889
 	- Training R2 Score 0.4211471761273704

   - **XGB:**
  	- Accuracy : 1.0
  	- RMSE = 0.126
	  - MSE = 0.015848539847458695
	  - MAE = 0.0996559838962222
	  - R2 = 0.050282766699744186
	  - Adjusted R2 = 0.045803672689524455
	  - **Predicted_Test_Mean: 0.16428576**


### **TEST**

 - **LR:**
   - Test Mean Absolute Error     0.12248642228048164
- Root Mean Squared Error: 0.15649191351304714
- Test R2 Score -0.4394473543883126
  - **Predicted_Test_Mean: 0.1638754354650507**

- **XGB:**
  - Accuracy : 1.0
  - RMSE = 0.131
 - MSE = 0.01704826743318137
 - MAE = 0.10279519278942896
 - R2 = -0.002056555186797704
 - Adjusted R2 = -0.007377209462125833
 - **Predicted_Test_Mean:0.16428071**

- **NN-Adam**
	- loss: 0.0171
	- r2_score: -3.0015e-03
	- root_mean_squared_error: 0.1306
	- **Predicted_Test_Mean:0.16428071**



-------------------------------------------------------------------
------------------------------------------------------------------

#*Learning Rate: 0.0054*

## **ALL Features**
### **TRAIN SET**
- Actual Train Mean = 0.1642970077485156

- **LR:**
    - Training Mean Absolute Error 0.06297834494117917
	- Training Root Mean Squared Error: 0.1613450605731818
	- Training R2 Score 0.5709287225489612

  - **XGB:**
	- Accuracy : 1.0
	- RMSE = 0.086
	- MSE = 0.007380457359082421
	- MAE = 0.06598655286937646
	- R2 = 0.5577291276658389
	- Adjusted R2 = 0.554803410759549


### **TEST SET**
  - **LR:**
 	- Test Mean Absolute Error     0.1278423066099372
 	- Test R2 Score -0.5561919388693397

  - **XGB:**
  	- Accuracy : 0.7480227791850307
  	- RMSE = 0.143
  	- MSE = 0.020395422400204342
  	- MAE = 0.11135060476335988
  	- R2 = -0.19879435209649343
  	- Adjusted R2 = -0.20772471087408362
   
    - **Predicted_Test_Mean: 0.16353457**
  

  - **NN:Adam:**
  	- loss: 0.0172
	  - r2_score: -8.3355e-03
	  - root_mean_squared_error: 0.1310
  	- **Predicted_Test_Mean: 0.16386086**

---------------------------------------------------------

##**Features of SIGNIFICANCE**

###- **TRAINING**

 	  - Actual Train Mean: 0.1642970077485156

   - **LR:**
 	- Training Mean Absolute Error 0.07204315868387007
 	- Root Mean Squared Error: 0.0982837111471889
 	- Training R2 Score 0.4211471761273704

   - **XGB:**
  	- Accuracy : 1.0
  	- RMSE = 0.095
  	- MSE = 0.009024183096083665
  	- MAE = 0.07267235988527049
   - R2 = 0.45922953879049133
  - Adjusted R2 = 0.4566791357349521
    -  **Predicted_Test_Mean: 0.16353457**


### **TEST**

 -**LR:**
  - Test Mean Absolute Error     0.12248642228048164
  	- Root Mean Squared Error: 0.15649191351304714
  	- Test R2 Score -0.4394473543883126
  	- **Predicted_Test_Mean: 0.1638754354650507**

- **XGB:**
  - Accuracy : 1.0
  - RMSE = 0.14
  - MSE = 0.01965676138470838
  - MAE = 0.10942817219097604
  - R2 = -0.15537761690391672
  - Adjusted R2 = -0.16151236531225614
  - **Predicted_Test_Mean:0.16386086**

#**Learning Rate: 0.008**
## **ALL Features**
###**TRAIN SET**
	- Actual Mean = 0.1642970077485156

  - **LR:**
	- Training Mean Absolute Error 0.06297834494117917
	- Training Root Mean Squared Error: 0.1613450605731818
	- Training R2 Score 0.5709287225489612

  -**XGB:**
	  - Accuracy : 1.0
  - RMSE = 0.086
	- MSE = 0.007380457359082421
	- MAE = 0.06598655286937646
	- R2 = 0.5577291276658389
	- Adjusted R2 = 0.554803410759549


### **TEST SET**

  **LR:**
  - Test Mean Absolute Error     0.1278423066099372
	- Root Mean Squared Error: 0.1613450605731818
- Test R2 Score -0.5561919388693397
	- **Predicted_Test_Mean: 0.16431755625935135**

  - **XGB:**
  	- Accuracy : 0.7480227791850307
	  - RMSE = 0.143
  	- MSE = 0.020395422400204342
  	- MAE = 0.11135060476335988
  	- R2 = -0.19879435209649343
  	- Adjusted R2 = -0.20772471087408362
	  - **Predicted_Test_Mean: 0.16353457**

- **NN:Adam:**
  - loss: 0.0172
 - r2_score: -8.3355e-03
 - root_mean_squared_error: 0.1310
 - **Predicted_Test_Mean: 0.16386086**

------------------------------------------------


## **Features of SIGNIFICANCE**

### **TRAINING**

 - **LR:**
  - Training Mean Absolute Error 0.07204315868387007
- Root Mean Squared Error: 0.0982837111471889
- Training R2 Score 0.4211471761273704
- **Predicted_Train_Mean: 0.1642970077485156**

- **XGB:**
 - Accuracy : 1.0
 - RMSE = 0.095
 - MSE = 0.009024183096083665
 - MAE = 0.07267235988527049
 - R2 = 0.45922953879049133
 - Adjusted R2 = 0.4566791357349521
 - **Predicted_Test_Mean: 0.16353457**


### **TEST**

 - **LR:**
  - Test Mean Absolute Error     0.12248642228048164
  - Root Mean Squared Error: 0.15649191351304714
  - Test R2 Score -0.4394473543883126
- **Predicted_Test_Mean: 0.1638754354650507**

- **XGB:**
  - Accuracy : 1.0
  - RMSE = 0.14
  - MSE = 0.01965676138470838
  - MAE = 0.10942817219097604
  - R2 = -0.15537761690391672
  - Adjusted R2 = -0.16151236531225614
 - **Predicted_Test_Mean:0.16386086**
