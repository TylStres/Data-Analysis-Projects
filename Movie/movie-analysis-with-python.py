# Importing libraries and setting up project

import pandas as pd 
from pandas.api.types import CategoricalDtype
import numpy as np 
import seaborn as sns

from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

# % matplotlib inline
matplotlib.rcParams['figure.figsize'] = (12,8)

pd.options.mode.chained_assignment = None 

# Read data from the csv file

df = pd.read_csv(r'C:\Users\tstre\OneDrive\Python Programs\Movie\imdb_top_1000.csv')

# Looking at the data

print(df.head())

#Check for any missing data as a function so that it can be called later to verify changes
def missing_data(df):
    for col in df.columns:
        percent_missing = np.mean(df[col].isnull())
        print('{} - {}%'.format(col, round(percent_missing*100)))

# Drop any missing data
df = df.dropna()

# Drop any duplicates
df.drop_duplicates()

# Check cleaned data calling the missing_data function from earlier
missing_data(df)
print(df)

# Check data types of the columns
print(df.dtypes)

# Ordering the Data
print(df.sort_values(by=['Gross'], inplace=False, ascending=False))

# Checking for outliers
sns.boxplot(
    x = "Gross",
    showmeans=True,
    data=df
)
plt.show()

# Remove outliers based on zscore of 3- standard
z_scores = stats.zscore(df['Gross'])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3)
df = df[filtered_entries]

sns.boxplot(
    x="Gross",
    showmeans=True,
    data=df
)
plt.show()

# Data exploration
sns.regplot(x="IMDB_Rating", y="Gross", data=df, scatter_kws = {"color": "red"},line_kws = 
            {"color": "blue"}).set_title('Gross Earnings vs IMDB Rating')

plt.show()

# Correlation Matrix

df_numerized =df.copy()

for col_name in df_numerized.columns:
    if df_numerized[col_name].dtype == 'object':
        cat_dtype = CategoricalDtype(categories=df_numerized[col_name].unique())
        df_numerized[col_name] = df_numerized[col_name].astype(cat_dtype)


print(df_numerized)

df_corr = df_numerized.select_dtypes(include=np.number)

corr_mat = df_corr.corr()

corr_pairs = corr_mat.unstack()

sorted_pairs = corr_pairs.sort_values()

high_corr = sorted_pairs[(sorted_pairs) > 0.5]

print(high_corr)

correlation_matrix = df_corr.corr(method = 'pearson')

sns.heatmap(correlation_matrix, annot = True)
plt.title("Correlation Matrix for Movies")
plt.xlabel("Movie Features")
plt.ylabel("Movie Features")
plt.show()


