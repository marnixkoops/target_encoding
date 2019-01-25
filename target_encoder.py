import pandas as pd
import numpy as np

from custom_code import timefold
from sklearn import preprocessing


def target_encoder(df, column, target, index=None, method='mean'):
    """
    Target-based encoding is numerization of a categorical variables via the target variable. Main purpose is to deal
    with high cardinality categorical features without exploding dimensionality. This replaces the categorical variable
    with just one new numerical variable. Each category or level of the categorical variable is represented by a
    summary statistic of the target for that level.

    Args:
        df (pandas df): Pandas DataFrame containing the categorical column and target.
        column (str): Categorical variable column to be encoded.
        target (str): Target on which to encode.
        index (arr): Can be supplied to use targets only from the train index. Avoids data leakage from the test fold
        method (str): Summary statistic of the target. Mean, median or std. deviation.

    Returns:
        arr: Encoded categorical column.

    """

    index = df.index if index is None else index # Encode the entire input df if no specific indices is supplied

    if method == 'mean':
        encoded_column = df[column].map(df.iloc[index].groupby(column)[target].mean())
    elif method == 'median':
        encoded_column = df[column].map(df.iloc[index].groupby(column)[target].median())
    elif method == 'std':
        encoded_column = df[column].map(df.iloc[index].groupby(column)[target].std())
    else:
        raise ValueError("Incorrect method supplied: '{}'. Must be one of 'mean', 'median', 'std'".format(method))

    return encoded_column


# Create some dummy data
df = pd.DataFrame({
    'product_id': ['a'] * 4 + ['c'] * 1 + ['b'] * 5 + ['a'] * 1 + ['c'] * 3 + ['b'] * 1,
    'product_type_id': [111] * 7 + [999] * 3 + [000] * 4 + [999] * 1,
    'actual': [1, 3, 7, 4, 0, 1, 0, 1, 0, 0, 0, 1, 2, 3, 10]})

df


labelencoder = preprocessing.LabelEncoder()
df['product_id_encoded'] = labelencoder.fit_transform(df['product_id'])
df['product_type_id'] = labelencoder.fit_transform(df['product_type_id'])
df

# Cross-validation setup using timefold
timefolds = timefold.timefold(method='window', folds=2)

for fold, (train_idx, test_idx) in enumerate(timefolds.split(df)):
    train_idx = train_idx

train_idx, test_idx

df['product_id_encoded'] = target_encoder(df, column='product_id', target='actual', method='mean')
df

df['product_type_id'] = target_encoder(df, column='product_type_id', target='actual', index=train_idx, method='mean')
df

encoded_column = target_encoder(df, column='product_id', target='actual', index=train_idx, method='mean')




def smoothing_target_encoder(df, column, target, weight=100):
    """
    Target-based encoding is numerization of a categorical variables via the target variable. This replaces the
    categorical variable with just one new numerical variable. Each category or level of the categorical variable
    is represented by it's summary statistic of the target. Main purpose is to deal with high cardinality categorical
    features.

    Smoothing adds the requirement that there must be at least m values for the sample mean to replace the global mean.
    Source: https://www.wikiwand.com/en/Additive_smoothing

    Args:
        df (pandas df): Pandas DataFrame containing the categorical column and target.
        column (string): Categorical variable column to be encoded.
        target (string): Target on which to encode.
        method (string): Summary statistic of the target.
        weight (int): Weight of the overall mean.

    Returns:
        array: Encoded categorical variable column.

    """
    # Compute the global mean
    mean = df[target].mean()

    # Compute the number of values and the mean of each group
    agg = df.groupby(column)[target].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # Compute the 'smoothed' means
    smooth = (counts * means + weight * mean) / (counts + weight)

    # Replace each value by the according smoothed mean
    return df[column].map(smooth)
