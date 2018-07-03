#Numpy — a package for scientific computing.
#Pandas — a library providing high-performance, easy-to-use data structures and data analysis tools for the Python
#scikit-learn — a tool for data mining and data analysis.
#NLTK — a platform to work with natural language.
import text_prepare as preprocess

from ast import literal_eval
#ast.literal_eval raises an exception if the input isn't a valid Python datatype, so the code won't be executed if it's not.
import pandas as pd
import numpy as np

def read_data_with_tag(filename):
    #change the data format as a csv type format
    data = pd.read_csv(filename,sep = '\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data

def read_test_data(filename):
    data = pd.read_csv(filename,sep = '\t')
    return data


train = read_data_with_tag('data/train.tsv')
validation = read_data_with_tag('data/validation.tsv')
test = read_test_data('data/test.tsv')

train.head()

# Difference between train['title'] and train['title'].values
# with ID and without ID
x_train = train['title'].values
y_train = train['tags'].values

#Question: Why we need validation set?
x_validataion = validation['title'].values
y_validation = validation['tags'].values

x_test = test['title'].values


preprocessed_questions = []
# read the file and split each sentence and stored in a list
with open('data/text_prepare_tests.tsv',encoding = 'utf-8') as f:
    questions = f.read().split('\n')
for question in questions:
    question = preprocess.text_prepare(question)
    preprocessed_questions.append(question)
# change the list type to string
preprocessed_results = '\n'.join(preprocessed_questions)



