#Numpy — a package for scientific computing.
#Pandas — a library providing high-performance, easy-to-use data structures and data analysis tools for the Python
#scikit-learn — a tool for data mining and data analysis.
#NLTK — a platform to work with natural language.
import text_prepare as preprocess



##############################  Read the data from the file as csv format #############################
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

#train.head()

###########################  Split the dataset into "Condition" and "Label" ##########################

# Difference between train['title'] and train['title'].values
# with ID and without ID
x_train = train['title'].values
y_train = train['tags'].values

#Question: Why we need validation set?
x_validataion = validation['title'].values
y_validation = validation['tags'].values

x_test = test['title'].values


#################################  A TEST FUNCTION FOR PREPROCESSING #################################
preprocessed_questions = []
# read the file and split each sentence and stored in a list
with open('data/text_prepare_tests.tsv',encoding = 'utf-8') as f:
    questions = f.read().split('\n')
for question in questions:
    question = preprocess.text_prepare(question)
    preprocessed_questions.append(question)
# change the list type to string
preprocessed_results = '\n'.join(preprocessed_questions)

#prepared_questions = []
#for line in open('data/text_prepare_tests.tsv', encoding='utf-8'):
#    line = text_prepare(line.strip())
#    prepared_questions.append(line)
#text_prepare_results = '\n'.join(prepared_questions)

###################################  PREPROCESS THE CONDITION TEXT ###################################

x_train = [preprocess.text_prepare(question) for question in x_train]
x_validataion = [preprocess.text_prepare(question) for question in x_validataion]
x_test = [preprocess.text_prepare(question) for question in x_test]

#y_train = [preprocess.text_prepare(question) for question in y_train]


"""#############  Find 3 MOST POPULAR TAGS and 3 MOST POPULAR WORDS in the train data ##############"""
####################### Count the frequency of the words and tags in train data ######################
# Dictionary of all tags from train corpus with their counts.
tags_counts = {}
# Dictionary of all words from train corpus with their counts.
words_counts = {}

all_tags = []
for tags in y_train:
    for tag in tags:
        all_tags.append(tag)

words = []
for question in x_train:
    for word in question.split():
        words.append(word)

# Count the frequency of words and tags
import word_counting
tags_counts = word_counting.count(all_tags)
words_counts = word_counting.count_words(words)

# Get the top three popular tags and words

most_common_tags = sorted(tags_counts.items(), key=lambda x: x[1], reverse=True)[:3]
most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:3]
#[('javascript', 19078), ('c#', 19077), ('java', 18661)]
#[('using', 8278), ('php', 5614), ('java', 5501)]

