from sklearn.model_selection import GridSearchCV

param_grid = {'learning_rate': [0.001, 0.01, 0.1], 'max_depth': [2, 4, 6, 8]} # Choose the parameters I want to tune and what values to try

grid = GridSearchCV(GradientBoostingClassifier(n_estimators=40), # setting n_estimators low to make it go a bit faster
                    param_grid=param_grid, # Our search space - 4 * 3 = 12 models
                    n_jobs=4, # Simple parallelization
                    verbose=1, # Get some text output
                    scoring='roc_auc') # What scoring function to use to compare models

grid.fit(train_x, train_y) # Train all combinations of parameters

print(f"Best score: {grid.best_score_:.3f}") # Get the score of the best-scoring model
print(f"Best params: {grid.best_params_}") # Get the params of the best-scoring model
pd.DataFrame(grid.cv_results_).sort_values(by='rank_test_score') # Pretty print our results
