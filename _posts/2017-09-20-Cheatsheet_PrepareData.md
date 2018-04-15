---
title: Prepare Datasets cheatsheet
date: 2017-09-20
tags: cheatsheet pandas
category: cheatsheet
layout: post
comments: true
excerpt_separator: <!--more-->
---

This is a *Cheatsheet*, meaning it contains code snippets. Based on [Data Exploration](https://www.dataquest.io/course/data-exploration) course from [Dataquest](https://www.dataquest.io).

Often Dataset Preparation consists of combining several messy data sets into a single clean one to make further analysis easier.
It is always good to get acquainted with the topic on the domain the data represents. For example, reading wikipedia (for public data) - thus getting insights and understanding on what's relevant (in practice we can understand which rows or columns are relevant for us).

<!--more-->

Workflow may include:
 - Handle files with different formats and columns
 - Prepare to merge multiple files
 - Handle (format) columns values
 - Merge columns to acquire aggregated data
 - Use text processing to extract coordinates from a string
 - Convert columns from strings to numbers

So, useful `Code snippets` can be found on this page!

#### Reading txt files
Following code read two files of identical structure and concatenates it (stacking on each other) into a dataframe.


```python
all_survey = pd.read_csv('directory/survey.txt',
                         delimiter='\t',
                         encoding='windows-1252')
another_survey = pd.read_csv('directory/survey2.txt',
                         delimiter='\t',
                         encoding='windows-1252')

survey = pd.concat([all_survey,another_survey],axis=0)

survey.head()
df.info
```

#### Rename columns
For example ID columns should have identical names across datasets if we want to merge them eventually.


```python
survey.rename(columns={'dbn':'DBN'},inplace=True)
```

#### Choose relevant columns (filter)
Datasets might have huge number of columns, while only some of them are required for the project.


```python
cols = ["DBN", "rr_s", "rr_t", "rr_p", "N_s", "N_t", "N_p", "saf_p_11",
        "com_p_11", "eng_p_11", "aca_p_11", "saf_t_11", "com_t_11",
        "eng_t_11", "aca_t_11", "saf_s_11", "com_s_11", "eng_s_11",
        "aca_s_11", "saf_tot_11", "com_tot_11", "eng_tot_11", "aca_tot_11"]

survey = survey[cols]
```

#### Handle (format) data in DataFrame columns
Say, we have a column of digits: [1,4,3,2,5 ... 2]. And we want to transform it to padded values: [01,04,03,02,05 ... 02]


```python
# Defining a function to add padding (width of two)
def pad_to_two_digits(val):
    value_str = str(val)
    padded_value_str = value_str.zfill(2) # padding width of two
    return padded_value_str

# Pass the function to apply() method and assign to a new column
data["class_size"]['padded_csd'] = data["class_size"]["CSD"].apply(pad_to_two_digits)
```

#### Merging string-data columns
Merging columns with string data would result into new column with concatenated string as following: 'str1'+'str2' = 'str1str2'


```python
data['class_size']['DBN'] = data["class_size"]['padded_csd']+data["class_size"]['SCHOOL CODE']
```

#### Transform to numeric and sum columns
Using pd.to_numeric() with parameter errors='coerce' to handle missing values (transforms to NaN).
Then summing up the columns of choice using +.


```python
colnames=['SAT Math Avg. Score', 'SAT Critical Reading Avg. Score', 'SAT Writing Avg. Score']

for colname in colnames:
    data['sat_results'][colname] = pd.to_numeric(data['sat_results'][colname], errors='coerce')

data['sat_results']['sat_score'] = (data['sat_results'][colnames[0]] +
                                    data['sat_results'][colnames[1]] +
                                    data['sat_results'][colnames[2]])
```

#### Extract specific values from the column
Following example extracts lattitude value from a string: "1110 Boston Road\nBronx, NY 10456\n(40.8276026690005, -73.90447525699966)"
First a function is defined, which extracts lattitude
Then a function is applied to a column using pd.Series.apply() method and assigning the result to a new column.


```python
import re

# A function that extracts lattitude
def get_lat(val_str):
    lattitude_longitude = re.findall('\(.+\)', val_str)[0]
    lattitude = lattitude_longitude.split(',')[0].replace('(','')
    return lattitude

# Apply function row by row and assign the result to a new column
data['hs_directory']['lat'] = data['hs_directory']['Location 1'].apply(get_lat)
```

The lattitude column should be converted to numeric


```python
data['hs_directory']['lat'] = pd.to_numeric(data['hs_directory']['lat'], errors='coerce')
```

#### Normalize column values
Here is an example of how to perform normalization


```python
from sklearn import preprocessing
# Get series as arrays of data, reshaped to two dimensions.
sat_scores = districts_val_sat['sat_score'].values.reshape(-1,1)
prop_prices = districts_val_sat['FULLVAL'].values.reshape(-1,1)

# Create a minimum and maximum processor object
min_max_scaler = preprocessing.MinMaxScaler()

# Create an object to transform the data to fit minmax processor. And reshape back to 1 dimension arrays.
sat_scores_scaled = min_max_scaler.fit_transform(sat_scores).reshape(-1)
prop_prices_scaled = min_max_scaler.fit_transform(prop_prices).reshape(-1)

# Assign results to columns!
districts_val_sat['SAT scores normalized'] = pd.Series(sat_scores_scaled)
districts_val_sat['Property prices normalized'] = pd.Series(prop_prices_scaled)
```
