from custom_code import timefold
from sklearn import preprocessing


df = pd.read_hdf('./local_data/features.h5')
df = df.tail(10000).copy().reset_index(drop=True)

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

df['product_id_encoded'] = target_encoder(df, column='product_id', target='actual', index=train_idx, method='mean')
df

df['product_type_id_encoded'] = target_encoder(df, column='product_type_id', target='actual', index=train_idx, method='mean')
df



def target_encoder(df, column, target, index, method='mean'):
    """
    Target-based encoding is numerization of a categorical variables via the target variable. This replaces the
    categorical variable with just one new numerical variable. Each category or level of the categorical variable
    is represented by it's summary statistic of the target. Main purpose is to deal with high cardinality categorical
    features.

    Source: A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems
    http://helios.mm.di.uoa.gr/~rouvas/ssi/sigkdd/sigkdd.vol3.1/barreca.ps

    Args:
        df (pandas df): Pandas DataFrame containing the categorical column and target.
        column (str): Categorical variable column to be encoded.
        target (str): Target on which to encode.
        index (arr): Use targets only from the train index to avoid data leakage from the test fold
        method (str): Summary statistic of the target.

    Returns:
        arr: Encoded categorical variable column.

    """

    if method == 'mean':
        encoded_column = df[column].map(df.iloc[index].groupby(column)[target].mean())
    elif method == 'median':
        encoded_column = df[column].map(df.iloc[index].groupby(column)[target].median())
    else:
        raise ValueError("Incorrect method supplied: '{}'. Must be one of 'mean', 'median'".format(method))

    return encoded_column

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
