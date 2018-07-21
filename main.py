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

###########################  Split the dataset into "Question" and "Label" ##########################

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

most_common_tags = sorted(tags_counts.items(), key=lambda x: x[1], reverse=True)[:3] # x is {'key':'value'}
most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:3]
#[('javascript', 19078), ('c#', 19077), ('java', 18661)]
#[('using', 8278), ('php', 5614), ('java', 5501)]


######################################### Bag of words ########################################
import bag_of_words  # import the file of bag of words and call the bag of words function

dict_size = 5000
index_to_words = sorted(words_counts.keys(),key=lambda x : words_counts[x],reverse = True)[:dict_size] # word(key) is the index, frequency(value), x is the key
words_to_index = {word: i for i, word in enumerate(index_to_words)} # enumerate() gives the word an index
ALL_WORDS = words_to_index.keys()


#Apply the implemented function to all samples (this might take up to a minute):
#Transform the data to sparse representation is to store the useful information efficiently. There are many types of such representations, however slkearn algorithms can work only with csr matrix, so we will use this one.
from scipy import sparse as sp_sparse
x_train_bag = sp_sparse.vstack([sp_sparse.csr_matrix(bag_of_words.bag_of_words(text, words_to_index, dict_size)) for text in x_train])
x_validation_bag = sp_sparse.vstack([sp_sparse.csr_matrix(bag_of_words.bag_of_words(text, words_to_index, dict_size)) for text in x_validataion])
x_test_bag = sp_sparse.vstack([sp_sparse.csr_matrix(bag_of_words.bag_of_words(text, words_to_index, dict_size)) for text in x_test])
print('x_train shape ', x_train_bag.shape)
print('x_validation shape ', x_validation_bag.shape)
print('x_test shape ', x_test_bag.shape)

#x_train shape  (100000, 5000)
#x_validation shape  (30000, 5000)
#x_test shape  (20000, 5000)




