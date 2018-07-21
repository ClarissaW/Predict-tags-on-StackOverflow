#Parameters
#max_df = 0.50 means "ignore terms that appear in more than 50% of the documents".
#max_df = 25 means "ignore terms that appear in more than 25 documents".
#min_df = 0.01 means "ignore terms that appear in less than 1% of the documents".
#min_df = 5 means "ignore terms that appear in less than 5 documents".


from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_features(x_train, x_validation, x_test):
    """
        X_train, X_val, X_test — samples
        return TF-IDF vectorized representation of each sample and vocabulary
        """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result
    
    # Add token_pattern = '(\S+)' when you cannot find c++ or c#
    tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df = 0.95,ngram_range=(1,2),token_pattern='(\S+)')
    
    # fit_transform is a shortening for fit and transform, has to use fit_transform when applying them both, otherwise there will be an error.
    x_train = tfidf_vectorizer.fit_transform(x_train)
    x_validation = tfidf_vectorizer.transform(x_validation)
    x_test = tfidf_vectorizer.transform(x_test)
    
    return x_train, x_validation, x_test, tfidf_vectorizer.vocabulary_
