from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier

#As a basic classifier, use LogisticRegression. It is one of the simplest methods, but often it performs good enough in text classification tasks.
def train_classifier(x_train, y_train):
    # Create and fit LogisticRegression wraped into OneVsRestClassifier.
    clf = OneVsRestClassifier(LogisticRegression())
    # clf = OneVsRestClassifier(RidgeClassifier(normalize=True))
    clf = clf.fit(x_train,y_train)
    
    return clf
