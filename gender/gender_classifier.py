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

class GenderClassifier:
    def __init__(self, string_input = True):
        self.match_data = [None for i in range(6)]
        self.string_input = string_input

        # Process SSN dataset
        self.match_data[0] = process_ssn_data()
        self.match_data[1] = process_male_female_popular()
        self.match_data[2] = process_name_country_data()
        self.match_data[3] = process_name_gender_likelihoods()
        self.match_data[4] = process_expanded_dataset()
        self.match_data[5] = process_east_asian_dataset()

        self.training_data = {}
        for i in self.match_data:
            for k in i.keys():
                if k not in self.training_data.keys():
                    matches = [self.match_data[e].get(k, 0) for e in range(len(self.match_data))]
                    if matches.count('M') > matches.count('F'):
                        self.training_data[k] = 'M'
                    elif matches.count('M') < matches.count('F'):
                        self.training_data[k] = 'F'
                    else:
                        self.training_data[k] = [x for x in matches if x][0]
        
        self.model = create_model(self.training_data)
    
    def predict(self, name):
        first_name = find_first_name(delete_prefix(name)).lower()
        filtered_name = re.sub('[.,]+', '', first_name)
        cleaned_name = re.sub('[^a-z]+', '?', filtered_name)

        matches = [self.match_data[e].get(filtered_name, 0) for e in range(len(self.match_data))]
        if matches.count('M') + matches.count('F') == 0:
            vectorized_df = pd.DataFrame()
            vectorized_df['cleaned_name'] = [cleaned_name]

            vectorized_df['-3'] = vectorized_df['cleaned_name'].str[-3]
            vectorized_df['-2'] = vectorized_df['cleaned_name'].str[-2]
            vectorized_df['-1'] = vectorized_df['cleaned_name'].str[-1]

            columns = ['-3','-2', '-1']
            tempX = [pd.get_dummies(vectorized_df[i], ) for i in columns]

            necessary_columns = ['?'] + [chr(i) for i in range(97, 123)]
            for i in tempX:
                for j in necessary_columns:
                    if j not in i.columns:
                        i[j] = [0]

            actualX = pd.concat([i for i in tempX], axis=1)
            
            predicted_values = self.model.predict(actualX)
            return map_to_gender(predicted_values[0])

        if matches.count('M') > matches.count('F'):
            return 'M'
        elif matches.count('M') < matches.count('F'):
            return 'F'
        else:
            if matches.index('F') < matches.index('M'):
                return 'F'
            else:
                return 'M'


################## Functions to process the sample JSON file ######################

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

def clean_and_extract_first_name(df):
    df['name'] = df['name'].apply(delete_prefix)
    df['first'] = df['name'].apply(find_first_name)
    df['filtered_first'] = df['first'].apply(lambda x: re.sub('[.,]+', '', x))
    df['cleaned_name'] = df['filtered_first'].apply(lambda x: re.sub('[^a-z]+', '?', x))
    return df

def create_final_prediction(row):
  if row['CLASSIFY5'] == 'M' or row['CLASSIFY5'] == 'F':
    return row['CLASSIFY5']
  return row['model_prediction']

def map_to_full(x):
  if x == 'M':
    return "male"
  return "female"

def map_to_gender(x):
    if x:
        return 'F'
    return 'M'
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

###################################################################################

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


##### APPROACH 0 #####
# Process SSN data
def process_ssn_data():
    names_genders = {}
    with open("data/combined_ss_data.txt") as f:
      for line in f:
        name = line[:line.index(',')].lower()
        temp_line = line[line.index(',') + 1:]
        gender = temp_line[:temp_line.index(',')]
        if name in names_genders.keys():
            names_genders[name][gender] += 1
        else:
            names_genders[name] = {'M': 0, 'F': 0, 'First': gender} # M/F/First
            names_genders[name][gender] += 1

    data = {}
    for name in names_genders.keys():
        if names_genders[name]['F'] > names_genders[name]['M']:
            data[name] = 'F'
        elif names_genders[name]['F'] < names_genders[name]['M']:
            data[name] = 'M'
        else:
            data[name] = names_genders[name]['First']

    return data


##### APPROACH 1 #####
# Process male/female.txt
def process_male_female_popular():
    data = {}
    names_list_m, names_list_f = [], []

    with open('data/female.txt') as f:
      for line in f:
        if ' ' in line:
          name = line[:line.index(' ')]
          if name.lower() not in data.keys():
            data[name.lower()] = 'F'
            names_list_f.append(name.lower())

    with open('data/male.txt') as g:
      for line in g:
        if ' ' in line:
          name = line[:line.index(' ')]
          if name not in data.keys():
            if name.lower() in names_list_f:
              if len(names_list_m) < names_list_f.index(name.lower()):
                data[name.lower()] = 'M'
            names_list_m.append(name.lower())

    return data

##### APPROACH 2 #####
# Process nam_dict.txt
def process_name_country_data():
    data = {}
    with open('data/nam_dict.txt') as f:
        for line in f:
            if line[0] != '#':
                if 'M' in line[:2]:
                    k = line[3:3+line[3:].index(" ")].lower()
                    if k not in data.keys():
                        data[k] = 'M'
                elif 'F' in line[:2]:
                    k = line[3:3+line[3:].index(" ")].lower()
                    if k not in data.keys():
                        data[k] = 'F'
                else:
                    continue
    return data

##### APPROACH 3 #####
# Process name_gender_likelihoods file
def process_name_gender_likelihoods():
    # Process our outside data source
    data = {}
    outside_data = pd.read_csv('data/name_gender.csv')
    outside_data['name'] = outside_data['name'].apply(lambda x: x.lower())

    for row in outside_data.iterrows():
      if row[1]['name'] not in data.keys():
        data[row[1]['name']] = row[1]['gender']

    return data


def num_to_gender(val):
    if val in [1, '1']:
        return 'M'
    elif val in [0, '0']:
        return 'F'

##### APPROACH 4 #####
# Process more thorough classifications
def process_expanded_dataset():
    data = {}
    names_df = pd.read_csv('data/gender_refine-csv.csv')
    for row in names_df.iterrows():
        if row[1]["name"].lower() not in data.keys() and row[1]["gender"] in [1, '1', 0, '0']:
            data[row[1]["name"].lower()] = num_to_gender(row[1]["gender"])
    return data

##### APPROACH 5 #####
# East Asian names
def process_east_asian_dataset():
    data = {}
    p = Pinyin()
    female_chinese, male_chinese = [], []

    with open('data/female_c.txt') as f:
        for line in f:
            line = line.replace('\n', '').replace('\t', ' ')
            vals = line.split(' ')
            cleaned_vals = [i for i in vals if i]
            female_chinese += cleaned_vals

    converted_female_chinese = [p.get_pinyin(i).replace('、', ' ').replace('-', '').replace('\n', '') for i in female_chinese]
    for i in converted_female_chinese:
        if i not in data.keys():
            data[i] = 'F'
        if i + i not in data.keys():
            data[i + i] = 'F'

    with open('data/male_c.txt') as f:
        for line in f:
            line = line.replace('\n', '').replace('\t', ' ')
            vals = line.split(' ')
            cleaned_vals = [i for i in vals if i]
            male_chinese += cleaned_vals

    converted_male_chinese = [p.get_pinyin(i).replace('、', ' ').replace('-', '').replace('\n', '') for i in male_chinese]
    for i in converted_male_chinese:
        if i not in data.keys():
            data[i] = 'M'
        if i + i not in data.keys():
            data[i + i] = 'M'

    return data

def create_model(training_data):
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

    model = RidgeClassifier(fit_intercept = True, solver = 'lsqr')
    model.fit(X_train, y_train)
    training_df['pred'] = model.predict(modelX)
    training_df['pred'] = training_df['pred'].apply(to_gender_class)
    # print(model.score(X_train, y_train))
    # print(model.score(X_test, y_test))
    print("Ready")
    return model
