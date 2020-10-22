# Author: Kunal Adhia

import numpy as np
import pandas as pd
import json
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
import re
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from xpinyin import Pinyin

COMMON_PREFIXES = ['mr.', 'dr.', 'ms.', 'mrs.', 'mr', 'dr', 'ms', 'mrs', 'sir', 'ca', 'ca.']
# Load the raw sample data and extract names
# Load the raw sample data and extract names
def load_sample_dataset(file_name):
    df = None
    names = []
    countries = []
    with open(file_name) as f:
      for line in f:
        j = json.loads(line)
        names.append(str(j['name']).lower())
        countries.append(j['country'])
    df = pd.DataFrame()
    df['name'] = names
    df['country'] = countries
    return df

# Delete common titles
def delete_prefix(x):
  if x.count(' ') == 0 or x[:x.index(' ')] not in COMMON_PREFIXES:
    return x
  return x[x.index(' ') + 1:]

# Extract First Name
def find_first_name(x):
  if ' ' not in x:
    return x
  by_word = x.split(" ")
  for i in by_word:
    if len(i) > 2 and i[0] == '(' and i[-1] == ')' and i[1:-1] not in COMMON_PREFIXES and '.' not in i:
      return i[1:-1]
  if '-' in x[:x.index(' ')]:
    return x[:x.index('-')]
  if by_word[0][-1] == ',':
    return by_word[1]
  return x[:x.index(' ')]


def clean_and_extract_first_name(df):
    df['name'] = df['name'].apply(delete_prefix)
    df['first'] = df['name'].apply(find_first_name)
    df['filtered_first'] = df['first'].apply(lambda x: re.sub('[.,]+', '', x))
    df['cleaned_name'] = df['filtered_first'].apply(lambda x: re.sub('[^a-z]+', '?', x))
    return df

##### APPROACH 0 #####
# Process SSN data
def approach_0(df, training_data):
    class_0_names, class_0_genders = [], []
    with open("data/combined_ss_data.txt") as f:
      for line in f:
        name = line[:line.index(',')].lower()
        if name not in training_data:
          temp_line = line[line.index(',') + 1:]
          gender = temp_line[:temp_line.index(',')]
          training_data[name] = gender
          class_0_names.append(name)
          class_0_genders.append(gender)
    df['CLASSIFY0'] = df['filtered_first'].apply(lambda x: training_data.get(x, None))
    return df, training_data

##### APPROACH 1 #####
# Process male/female.txt
def approach_1(df, training_data):
    names_list_m, names_list_f = [], []

    with open('data/female.txt') as f:
      for line in f:
        if ' ' in line:
          name = line[:line.index(' ')]
          if name.lower() not in training_data.keys():
            training_data[name.lower()] = 'F'
            names_list_f.append(name.lower())

    with open('data/male.txt') as g:
      for line in g:
        if ' ' in line:
          name = line[:line.index(' ')]
          if name not in training_data.keys():
            if name.lower() in names_list_f:
              if len(names_list_m) < names_list_f.index(name.lower()):
                training_data[name.lower()] = 'M'
            names_list_m.append(name.lower())

    df['CLASSIFY1'] = df['filtered_first'].apply(lambda x: training_data.get(x, None))

    return df, training_data

##### APPROACH 2 #####
def extract_country_name(line):
  # strips the white spaces from the beginning and end of the line
  # we should be left with just the name of the country.
  while line[0] == ' ':
    line = line[1:]
  line = line[::-1]
  while line[0] == ' ':
    line = line[1:]
  return line[::-1]

def change_to(vals, v_from, v_to):
    try:
        vals[vals.index(v_from)] = v_to
    except:
        pass

def match_country_names(df, input_df, country_names):
    change_to(country_names, 'U.S.A.', 'United States')
    change_to(country_names, 'Great Britain', 'United Kingdom')
    change_to(country_names, 'Swiss', 'Switzerland')
    original_countries = df['country'].to_list()

    counter1 = 0
    raw_sample_data_mat = []
    for row in df.iterrows():
      raw_data = [row[1]['cleaned_name']]
      sample_country = row[1]['country']
      # Try to estimate a similar nationality
      if sample_country in ['Brazil', 'Mexico', 'Panama', 'Ecuador', 'Paraguay']:
        sample_country = 'Spain'
      elif sample_country in ['Canada', 'Virgin Islands', 'Australia', 'South Africa', 'Chile', 'New Zealand', 'Peru', 'Colombia']:
        sample_country = 'United States'
      elif sample_country in ['Taiwan', 'Phillipines', 'Hong Kong', 'Singapore', 'Malaysia', 'Thailand']:
        sample_country = 'China'
      elif sample_country in ['Bangladesh', 'Pakistan']:
        sample_country = 'India'

      found_in_list = False
      for check_c in country_names[:-1]:
        if sample_country in check_c or check_c in sample_country:
          found_in_list = True
          raw_data.append(10)
        else:
          raw_data.append(0)


      if not found_in_list:
        counter1 += 1
        raw_data.append(10)
      else:
        raw_data.append(0)
      raw_data.append('?')
      raw_sample_data_mat.append(raw_data)

      input_df.columns = ['name'] + country_names + ['gender']
      sample_country_data = pd.DataFrame(raw_sample_data_mat)
      sample_country_data.columns = ['name'] + country_names + ['gender']

      return input_df, sample_country_data

# Process nam_dict.txt
def approach_2(df, training_data):
    nam_dict_names, nam_dict_genders = [], []
    country_names, country_columns = [], []
    line_counter = 0
    popularity_by_country = [[] for i in range(55)]
    with open('data/nam_dict.txt') as f:
      for line in f:
        if line_counter in range(177, 342, 3):
          processed_line = line.replace('#', '').replace('$', '').replace('\n', '')
          country_names.append(extract_country_name(processed_line))
        elif line_counter in range(178, 343, 3):
          country_columns.append(line.index('|'))

        if line[0] != '#':
          if 'M' in line[:2]:
            k = line[3:3+line[3:].index(" ")].lower()
            if k not in training_data.keys():
              training_data[k] = 'M'
            nam_dict_names.append(k)
            nam_dict_genders.append('M')
          elif 'F' in line[:2]:
            k = line[3:3+line[3:].index(" ")].lower()
            if k not in training_data.keys():
              training_data[k] = 'F'
            nam_dict_names.append(k)
            nam_dict_genders.append('F')
          else:
            continue

          for i in range(30, 85):
            popularity_by_country[i - 30].append(0 if line[i] == ' ' else int(line[i], 16))

        line_counter += 1

    input_df = pd.DataFrame()
    input_df['name'] = nam_dict_names
    for i in range(55):
      input_df[country_names[i]] = popularity_by_country[i]
    input_df['gender'] = nam_dict_genders

    input_df, sample_country_data = match_country_names(df, input_df, country_names)

    model_evaluator = input_df.copy()

    input_df = pd.concat([input_df, sample_country_data], axis=0)

    df['CLASSIFY2'] = df['filtered_first'].apply(lambda x: training_data.get(x, None))
    return df, input_df, model_evaluator, training_data

##### APPROACH 3 #####
def approach_3(df, training_data):
    # Process our outside data source
    outside_data = pd.read_csv('data/name_gender.csv')
    outside_data['name'] = outside_data['name'].apply(lambda x: x.lower())

    for row in outside_data.iterrows():
      if row[1]['name'] not in training_data.keys():
        training_data[row[1]['name']] = row[1]['gender']
    df['CLASSIFY3'] = df['filtered_first'].apply(lambda x: training_data.get(x, None))

    return df, training_data


def num_to_gender(val):
  if val in [1, '1']:
    return 'M'
  elif val in [0, '0']:
    return 'F'

##### APPROACH 4 #####
def approach_4(df, training_data):
    indian_names_df = pd.read_csv('data/gender_refine-csv.csv')
    for row in indian_names_df.iterrows():
        if row[1]["name"].lower() not in training_data.keys():
            training_data[row[1]["name"].lower()] = num_to_gender(row[1]["gender"])
    df['CLASSIFY4'] = df['filtered_first'].apply(lambda x: training_data.get(x, None))
    return df, training_data

##### APPROACH 5 #####
def approach_5(df, training_data_permanant):
    training_data = training_data_permanant.copy() # We don't want to use this for training the model
    female_chinese, male_chinese = [], []
    with open('data/female_c.txt') as f:
        for line in f:
            line = line.replace('\n', '').replace('\t', ' ')
            vals = line.split(' ')
            cleaned_vals = [i for i in vals if i]
            female_chinese += cleaned_vals
    p = Pinyin()
    converted_female_chinese = [p.get_pinyin(i).replace('、', ' ').replace('-', '').replace('\n', '') for i in female_chinese]
    for i in converted_female_chinese:
        if i not in training_data.keys():
            training_data[i] = 'F'
        if i + i not in training_data.keys():
            training_data[i + i] = 'F'

    with open('data/male_c.txt') as f:
        for line in f:
            line = line.replace('\n', '').replace('\t', ' ')
            vals = line.split(' ')
            cleaned_vals = [i for i in vals if i]
            male_chinese += cleaned_vals

    converted_male_chinese = [p.get_pinyin(i).replace('、', ' ').replace('-', '').replace('\n', '') for i in male_chinese]
    for i in converted_male_chinese:
        if i not in training_data.keys():
            training_data[i] = 'M'
        if i + i not in training_data.keys():
            training_data[i + i] = 'M'

    df['CLASSIFY5'] = df['filtered_first'].apply(lambda x: training_data.get(x, None))
    return df

def country_based_model(df, input_df, model_evaluator):
    input_df['-3'] = input_df['name'].str[-3]
    input_df['-2'] = input_df['name'].str[-2]
    input_df['-1'] = input_df['name'].str[-1]

    columns = ['-3','-2', '-1']
    vectorized_name = [pd.get_dummies(input_df[i]) for i in columns]
    input_df = pd.concat([vectorized_name[0], vectorized_name[1], vectorized_name[2], input_df], axis = 1)

    cY = input_df['gender'].head(39469)
    input_df = input_df.drop(columns = ['name', 'gender', '-3', '-2', '-1'])

    cX = input_df.head(39469)
    cX_train, cX_test, cy_train, cy_test = train_test_split(cX, cY, test_size=0.2, random_state=42)

    model = RidgeClassifier(fit_intercept = False, solver = 'lsqr')
    model.fit(cX_train, cy_train)

    training_predictions = model.predict(input_df.head(39469))
    model_evaluator['MODEL_PREDICTION'] = training_predictions
    country_model_predictions = model.predict(input_df.tail(1000))
    df['COUNTRY_MODEL'] = country_model_predictions

    return df, model_evaluator

def general_model(df, training_data):
    ml_names, ml_genders = [], []
    for k, v in training_data.items():
      k = re.sub('[^a-z]+', '?', k)
      ml_names.append(k)
      ml_genders.append(v)

    training_df = pd.DataFrame()
    training_df['name'] = ml_names
    training_df['gender'] = ml_genders
    training_df['-3'] = training_df['name'].str[-3]
    training_df['-2'] = training_df['name'].str[-2]
    training_df['-1'] = training_df['name'].str[-1]

    columns = ['-3','-2', '-1']
    X = [pd.get_dummies(training_df[i]) for i in columns]

    modelX = pd.concat([i for i in X], axis=1)

    def map_genders(x):
      if x == 'F':
        return 1
      return 0

    modelY = training_df['gender'].apply(map_genders)

    X_train, X_test, y_train, y_test = train_test_split(modelX, modelY, test_size=0.2, random_state=42)

    def to_gender_class(x):
      if x:
        return 'F'
      return 'M'

    model = RidgeClassifier(fit_intercept = False, solver = 'lsqr')
    model.fit(X_train, y_train)
    training_df['pred'] = model.predict(modelX)
    training_df['pred'] = training_df['pred'].apply(to_gender_class)
    print("Model Scores (Train/Test):")
    print(model.score(X_train, y_train))
    print(model.score(X_test, y_test))
    # training_df[training_df['gender'] != training_df['pred']]


    # Predict using model
    vectorized_df = pd.DataFrame()
    vectorized_df['cleaned_name'] = df['cleaned_name']

    vectorized_df['-3'] = df['cleaned_name'].str[-3]
    vectorized_df['-2'] = df['cleaned_name'].str[-2]
    vectorized_df['-1'] = df['cleaned_name'].str[-1]

    columns = ['-3','-2', '-1']
    tempX = [pd.get_dummies(vectorized_df[i], ) for i in columns]

    necessary_columns = ['?'] + [chr(i) for i in range(97, 123)]
    for i in tempX:
      for j in necessary_columns:
        if j not in i.columns:
          i[j] = [0 for k in range(1000)]

    actualX = pd.concat([i for i in tempX], axis=1)

    def map_to_gender(x):
      if x:
        return 'F'
      return 'M'

    predicted_values = model.predict(actualX)
    df['model_prediction'] = predicted_values
    df['model_prediction'] = df['model_prediction'].apply(map_to_gender)

    return df

def create_final_prediction(row):
  if row['CLASSIFY5'] == 'M' or row['CLASSIFY5'] == 'F':
    return row['CLASSIFY5']
  return row['model_prediction']

def map_to_full(x):
  if x == 'M':
    return "male"
  return "female"

def populate_fields_with_final_predictions(df):
    classified_vals = df['final_prediction'].to_list()

    counter = 0
    g = open("output/gender_sample_final.jsonl", 'w')
    with open("input/gender_sample.jsonl") as f:
      for line in f:
        j = json.loads(line)
        j["gender"] = classified_vals[counter]
        g.write(json.dumps(j) + "\n")
        counter += 1
    g.close()

df = load_sample_dataset("input/gender_sample.jsonl")
df = clean_and_extract_first_name(df)
training_data = {}
df, training_data = approach_0(df, training_data)
df, training_data = approach_1(df, training_data)
df, input_df, model_evaluator, training_data = approach_2(df, training_data)
df, training_data = approach_3(df, training_data)
df, training_data = approach_4(df, training_data)
df = approach_5(df, training_data)

# Train and predict using the two types of models
df, model_evaluator = country_based_model(df, input_df, model_evaluator)
df = general_model(df, training_data)

df['final_prediction'] = df.apply(create_final_prediction, axis = 1)
df['final_prediction'] = df['final_prediction'].apply(map_to_full)

# Save data
df.to_csv('output/full_classifier_predictions_new.csv')
model_evaluator.to_csv('output/country_based_model_training_data_results_new.csv')

# Final Predictions
populate_fields_with_final_predictions(df)
df.to_csv("output/final_results.csv")
