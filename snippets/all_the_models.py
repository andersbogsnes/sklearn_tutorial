# Import some more models to try
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict

# Instantiate all the models we want to try
models = [LogisticRegression(), GradientBoostingClassifier(), GaussianNB(), RandomForestClassifier(), KNeighborsClassifier(), DummyClassifier()]

results = {} # Setup a dictionary to save our results
for model in models: # Iterate over our models
    model_name = model.__class__.__name__ # Get the model name
    model.fit(train_x, train_y) # Fit the model
    pred_proba = model.predict_proba(test_x)[:, 1] # Get probability of class 1
    results[model_name] = roc_auc_score(test_y, pred_proba) # Calculate ROC AUC score
pd.Series(results).sort_values(ascending=False) # Convert to Series for sorting and prettyprinting
