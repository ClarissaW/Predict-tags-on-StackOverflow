
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
# accuracy_scores: return the correctly classified samples. The set of labels predicted for a sample must exactly match the corresponding set of labels

# f1_score: F1 = 2 * (precision * recall) / (precision + recall), average could be micro, macro, weighted

# roc_auc_score: Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores. Note: this implementation is restricted to the binary classification task or multilabel classification task in label indicator format.

# average_precision_score: Compute average precision (AP) from prediction scores

# recall_score
#'micro': Calculate metrics globally by considering each element of the label indicator matrix as a label.
#'macro': Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
#'weighted': Calculate metrics for each label, and find their average, weighted by support (the number of true instances for each label).


# accuracy, f1 score, and recall use y_true and y_pred
# roc_auc_score and average_precision_score use y_true and y_score

def evaluation_true_pred(y_validation, y_predicted):
    accuracy = accuracy_score(y_validation, y_predicted)
    f1 = f1_score(y_validation, y_predicted, average = 'weighted')
    recall = recall_score(y_validation, y_predicted, average = 'weighted')
    print("Accuray is {}".format(accuracy))
    print("F1 is {}".format(f1))
    print("Recall is {}".format(recall))

def evaluation_true_scores(y_validation, y_scores):
    roc = roc_auc_score(y_validation, y_scores)
    average_precision = average_precision_score(y_validation, y_scores)
    print("ROC is {}".format(roc))
    print("Average Precision is {}".format(average_precision))

