from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=10) # Create a 10-fold StratifiedKFold cross validator
model = RandomForestClassifier(n_estimators=10, max_depth=5) # Create a RandomForest model with some hyperparameters set
scores = [] # Ready to accumulate scores
for train_idx, test_idx in cv.split(train_x, train_y): # cv.split gives us indexes to use per split
    train_split_X, train_split_y = train_x.values[train_idx], train_y[train_idx] # index into our train data
    test_split_X, test_split_y = train_x.values[test_idx], train_y[test_idx] # index into our test data
    
    model.fit(train_split_X, train_split_y) # Train the model on train data
    predict_proba = model.predict_proba(test_split_X)[:, 1] # Calculate ROC score on test data
    roc_score = roc_auc_score(test_split_y, predict_proba) 
    print(f"Score: {roc_score:.2f}")
    scores.append(roc_score) # Append scores to our list
print(f"Average: {np.mean(scores):.2f} Std: {np.std(scores):.2f}") # Print average and standard deviation of our scores
