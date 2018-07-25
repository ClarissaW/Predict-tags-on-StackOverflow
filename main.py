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
x_validation = validation['title'].values
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
x_validation = [preprocess.text_prepare(question) for question in x_validation]
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
x_validation_bag = sp_sparse.vstack([sp_sparse.csr_matrix(bag_of_words.bag_of_words(text, words_to_index, dict_size)) for text in x_validation])
x_test_bag = sp_sparse.vstack([sp_sparse.csr_matrix(bag_of_words.bag_of_words(text, words_to_index, dict_size)) for text in x_test])
#print('x_train shape ', x_train_bag.shape)
#print('x_validation shape ', x_validation_bag.shape)
#print('x_test shape ', x_test_bag.shape)

"""
#x_train shape  (100000, 5000)
#x_validation shape  (30000, 5000)
#x_test shape  (20000, 5000)
"""

#For the 10th row in X_train_mybag find how many non-zero elements it has. In this task the answer (variable non_zero_elements_count) should be a number, e.g. 20.
row = x_train_bag[10].toarray()[0]
non_zero_elements_count = (row > 0).sum() #len([i for i in row if i > 0])
#print(non_zero_elements_count)
"""
    7
"""

######################################### TF-IDF ########################################
import tf_idf

x_train_tfidf, x_validation_tfidf, x_test_tfidf, tfidf_vocab = tf_idf.tfidf_features(x_train, x_validation, x_test)
tfidf_reversed_vocab = {i:word for word,i in tfidf_vocab.items()}

#print(tfidf_vocab['c++'])
#print(tfidf_reversed_vocab[1976])

"""
    1976
    c++
"""
################################ MultiLabel Classifier ###################################
#Multiple tags, transform labels in a binary form and the prediction will be a mask of 0s and 1s. For this purpose it is convenient to use [MultiLabelBinarizer]

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
#mlb.fit_transform(data) means that generate a vector and the length will be equal to classes' length, 0 and 1 represents whether the data has those tags.
y_train = mlb.fit_transform(y_train)
y_validation = mlb.fit_transform(y_validation)
#print(mlb)

######################################## Train Classifier ###########################################
import train_classifier
classifier_bag = train_classifier.train_classifier(x_train_bag, y_train)
classifier_tfidf = train_classifier.train_classifier(x_train_tfidf, y_train)

########### Apply the classifier on the validation data to predict and get the score ###########

y_val_predicted_labels_bag = classifier_bag.predict(x_validation_bag)
y_val_predicted_scores_bag = classifier_bag.decision_function(x_validation_bag)

y_val_predicted_labels_tfidf = classifier_tfidf.predict(x_validation_tfidf)
y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(x_validation_tfidf)

# Take a look at the performance of the classifier
y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_tfidf)
y_val_inversed = mlb.inverse_transform(y_validation)
for i in range(3):
    print('Title:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(x_validation[i], ','.join(y_val_inversed[i]), ','.join(y_val_pred_inversed[i])))

"""
    Title:    odbc_exec always fail
    True labels:    php,sql
    Predicted labels:
    
    
    Title:    access base classes variable within child class
    True labels:    javascript
    Predicted labels:
    
    
    Title:    contenttype application json required rails
    True labels:    ruby,ruby-on-rails
    Predicted labels:    json,ruby-on-rails

"""
