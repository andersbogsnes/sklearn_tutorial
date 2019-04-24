from sklearn.model_selection import KFold

cv = KFold(n_splits=3)

example_X = X[:21]
example_y = y[:21]

for i, (train_idx, test_idx) in enumerate(cv.split(example_X, example_y), start=1):
    print(f"Split nr: {i}")
    print('='*79)
    print("Train ids:")
    print(f"Number of rows: {len(train_idx)}")
    print(train_idx)
    print('-'*79)
    print(example_X.loc[train_idx, :'Age'])
    print()
    print("Test ids:")
    print(f"Number of rows: {len(test_idx)}")
    print(test_idx)
    print('-'*79)
    print(example_X.loc[test_idx, :'Age'])
    print()
    input()
