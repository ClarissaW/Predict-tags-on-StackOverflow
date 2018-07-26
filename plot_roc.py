
############################# In jupyter notebook, use inline matplotlib  ############################
from metrics import roc_auc
%matplotlib inline

n_classes = len(tags_counts)
roc_auc(y_val, y_val_predicted_scores_mybag, n_classes)

n_classes = len(tags_counts)
roc_auc(y_val, y_val_predicted_scores_tfidf, n_classes)



######################## Not in IPython, iteratively plot the charts in one figure.  ########################

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr = dict()
tpr = dict()
roc_auc = dict()

# n_classes = len(tags_counts)

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_val[:, i], y_val_predicted_scores_mybag[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
print(n_classes)
print(len(roc_auc))

# Plot of a ROC curve for a specific class
plt.figure()
for i in range(n_classes):
    
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
plt.show()

