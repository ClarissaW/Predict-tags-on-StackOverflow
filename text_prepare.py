import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
tokenizer = nltk.tokenize.WhitespaceTokenizer()


import re
RE_REPLACE_BY_SPACE = re.compile('[/(){}\[\]\|@,;]') # [] and | have special meaning in re, so use \[ \] \| replace these three characters
RE_BAD_SYMBOLS = re.compile('[^0-9a-z #+_]')

def text_prepare(text):
    
    text = text.lower() #lowercase text #text[0].lower() + text[1:]
    text = re.sub(RE_REPLACE_BY_SPACE,' ', text) #replace RE_REPLACE_BY_SPACE symbols by space in text
    text = re.sub(RE_BAD_SYMBOLS,'',text) # delete symbols which are in RE_BAD_SYMBOLS from text
    words = tokenizer.tokenize(text) # tokenize the text by space
    text = ' '.join(word for word in words if word not in STOPWORDS) # delete stopwords from text
    return text

######################################################################################################
##########################################  A TEST FUNCTION ##########################################
######################################################################################################
#def test_text_prepare():
#    examples = ["SQL Server - any equivalent of Excel's CHOOSE function?",
#                "How to free c++ memory vector<int> * arr?"]
#    answers = ["sql server equivalent excels choose function",
#                "free c++ memory vectorint arr"]
#    for ex, ans in zip(examples, answers):
#        if text_prepare(ex) != ans:
#            return "Wrong answer for the case: '%s'" % ex
#    return 'Basic tests are passed.'
#
#print(test_text_prepare())



