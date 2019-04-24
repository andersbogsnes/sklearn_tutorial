from sklearn.model_selection import cross_val_score # The magic function
model = GradientBoostingClassifier(learning_rate=0.1, max_depth=4, n_estimators=40) # Create our model with some parameters
cv = StratifiedKFold(n_splits=10) # Create a 10-fold StratifiedKFold cross validator

scores = cross_val_score(model, # The model we want to score
                         train_x, # Our data - remember our test_x and test_y are separate!
                         train_y, 
                         scoring='roc_auc', # Still want to score by roc_auc
                         cv=cv, # Our cross-validation object - could also pass 10 here
                         n_jobs=4, # Free parallellization! 
                         verbose=1)

print(f"Average: {np.mean(scores):.2f} Std: {np.std(scores):.2f}")
