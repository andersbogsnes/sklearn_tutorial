from sklearn.metrics import roc_auc_score, roc_curve

def plot_roc_curve(clf, test_x, test_y):
    pred_proba = clf.predict_proba(test_x)[:, 1] # Gives probability per label - we want the second class

    score = roc_auc_score(test_y, pred_proba)

    fpr, tpr, _ = roc_curve(test_y, pred_proba)

    plt.plot(fpr, tpr, label='Model ROC')
    plt.plot([0, 1], '--', label='Baseline ROC')
    plt.title(f'ROC Curve - score {score:.2f}');
    plt.ylabel('True Positive Rate');
    plt.xlabel('False Positive Rate');
    plt.legend(loc='best')
    
plot_roc_curve(clf, test_x, test_y)
