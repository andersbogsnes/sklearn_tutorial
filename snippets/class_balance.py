pred = clf.predict(test_x)
print(f"Predicted positive: {pred.sum()} Actually positive: {test_y.sum()}")
print(f"Proportion of Claims: {y.mean():.2%} Non-claims: {1 - y.mean():.2%}")
