gb = GradientBoostingClassifier()

gb.fit(train_x, train_y)

pred_proba = gb.predict_proba(test_x)[:, 1]
roc_auc_score(test_y, pred_proba)
