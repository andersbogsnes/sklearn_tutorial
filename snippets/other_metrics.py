from sklearn.metrics import f1_score, precision_score, recall_score
pred = clf.predict(test_x)

precision = precision_score(test_y, pred) # Out of the predicted positives, how many were actually positive? How sure are we, when the model guesses positive?
recall = recall_score(test_y, pred) # Out of all the positives, how many did the model find? How sure are we that the model has found all the positive cases?
f1 = f1_score(test_y, pred) # The harmonic mean of precision and recall - Tries to summarise the precision-recall tradeoff into one number
print(f"F1 score: {f1}, Precision: {precision}, Recall: {recall}")
