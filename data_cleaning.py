import numpy as np
import pandas as pd
import re
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
pd.options.display.max_columns = 50
np.random.seed(42)
plt.style.use('almbrand')


def clean_travel_insurance(df):
    df['Gender'] = df['Gender'].fillna('Unknown')
    df['Claim'] = np.where(df['Claim'] == 'Yes', 1, 0)
    y = df['Claim']
    X = df.drop(columns=['Product Name', 'Destination', 'Claim'])
    X = pd.get_dummies(X)
    return X, y


def load_titanic():
    df = pd.read_csv('data/raw/train.csv').drop(columns='PassengerId')
    return create_features(df)


def create_features(df):
    df['missing_age'] = np.where(df.Age.isna(), 1, 0)
    df['Age'] = df['Age'].fillna(df.Age.median())
    df['Sex'] = np.where(df['Sex'] == 'male', 1, 0)

    match = re.compile(r'([A-Z][a-z]*\.)')

    title_map = {
        "Mr.": "Mr",
        "Miss.": "Miss",
        "Mrs.": "Mrs",
        "Master.": "Master",
        "Dr.": "Other",
        "Rev.": "Other",
        "Major.": "Other",
        "Col.": "Other",
        "Mlle.": "Miss",
        "Lady.": "Other",
        "Jonkheer.": "Other",
        "Ms.": "Miss",
        "Capt.": "Other",
        "Don.": "Other",
        "Sir.": "Other",
        "Countess": "Other",
        "Mme.": "Mrs"
    }

    df['title'] = df.Name.str.extract(match, expand=False).map(title_map)
    df['family_size'] = df[['SibSp', 'Parch']].sum(axis=1)

    def create_ticket_letters(series):
        return (series.str.extract(r'(.*\s)', expand=False)
                .str.replace('.', '')
                .str.replace(' ', '')
                .str.strip()
                .fillna('No letters'))

    df['ticket_letters'] = create_ticket_letters(df.Ticket).where(
        lambda x: x.isin(['FCC', 'PC']), 'Other')

    df['missing_cabin'] = np.where(df.Cabin.isna(), 1, 0)

    df['num_cabins'] = df.Cabin.str.split(' ').apply(
        lambda x: len(x) if x is not np.nan else 0)

    df['cabin_letters'] = df.Cabin.str.split(' ').apply(
        lambda x: x[0][0] if x is not np.nan else 'X')

    df['Embarked'] = df.Embarked.fillna('Unknown')

    y = df.Survived
    X = pd.get_dummies(df.drop(columns='Survived'))

    return X, y
