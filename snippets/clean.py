def clean_travel_insurance(df):
    y = np.where(df['Claim'] == 'Yes', 1, 0)  # Set our y to be claims
    
    df['Gender'] = df['Gender'].fillna('Unknown') # Replace all NaN with 'Unknown'
    X = df.drop(columns=['Product Name', 'Destination', 'Claim']) # Get rid of our target from our features + simplify our data a bit
    X = pd.get_dummies(X) # Do One Hot Encoding of our data
    return X, y

X, y = clean_travel_insurance(df)
X.sample(10)
