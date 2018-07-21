import numpy as np

def bag_of_words(question, words_to_index, dict_size):
    #Generate 5000 0's vector [0,0,0,...,0]
    result_vector = np.zeros(dict_size)
    for word in question.split():
        if word in words_to_index:
            result_vector[words_to_index[word]] += 1
    return result_vector

#####################################################################################################
#########################################  A TEST FUNCTION ##########################################
#####################################################################################################
def test_bag_of_words():
    words_to_index = {'hi': 0, 'you': 1, 'me': 2, 'are': 3}
    examples = ['hi how are you']
    answers = [[1, 1, 0, 1]]
    for ex, ans in zip(examples, answers):
        if (bag_of_words(ex, words_to_index, 4) != ans).any():
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'

#print(test_bag_of_words())

