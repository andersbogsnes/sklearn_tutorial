from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(X, # Our features
                                                    y, # Our target
                                                    stratify=y, # Make sure that there are equal amounts of our target in train and test data
                                                    random_state=seed # train_test_split splits data randomly - by passing our seed, we can get reproducible results
                                                   )

clf.fit(train_x, train_y)
score = clf.score(test_x, test_y)
print(f"Accuracy: {score:.2%}")
