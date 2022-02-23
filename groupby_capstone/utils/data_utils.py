"""
Functions we can import
"""
import csv
import random
from pathlib import Path

import joblib
import numpy as np
import pandas
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision
from sklearn.metrics import f1_score

from groupby_capstone.settings import DATA_PATH


def convert_to_dict(filename):
    """
    Convert a CSV file to a list of Python dictionaries.
    """
    # open a CSV file - note - must have column headings in top row
    datafile = open(filename, newline='')

    # create list of OrderedDicts as of Python 3.6
    my_reader = csv.DictReader(datafile)

    # write it all out to a new list
    list_of_dicts = []
    for row in my_reader:
        # we convert each row to a string and add a newline
        list_of_dicts.append(dict(row))

    # close original csv file
    datafile.close()
    # return the list
    return list_of_dicts


def make_ordinal(num):
    """
    Create an ordinal (1st, 2nd, etc.) from a number.
    """
    base = num % 10
    if base in [0, 4, 5, 6, 7, 8, 9] or num in [11, 12, 13]:
        ext = "th"
    elif base == 1:
        ext = "st"
    elif base == 2:
        ext = "nd"
    else:
        ext = "rd"
    return str(num) + ext


def user_session_viewed(df):
    session_viewed = (
        df
            .loc[df.event_type == 'view']
            .groupby(['user_id', 'user_session'])
            .agg(
            num_viewed_items=('ones', sum),
            total_amt_viewed=('price', sum),
            products_viewed=('product_id', lambda x: len(x.unique())),
            categories_viewed=('category_code', lambda x: len(x.unique())),
            brands_viewed=('brand', lambda x: len(x.unique()))
        )
    )
    session_viewed['avg_amt_viewed'] = session_viewed.total_amt_viewed / session_viewed.num_viewed_items

    return session_viewed


def user_session_carted(df):
    session_carted = (
        df
            .loc[df.event_type == 'cart']
            .groupby(['user_id', 'user_session'])
            .agg(
            num_carted_items=('ones', sum),
            total_amt_carted=('price', sum),
            products_carted=('product_id', lambda x: len(x.unique())),
            categories_carted=('category_code', lambda x: len(x.unique())),
            brands_carted=('brand', lambda x: len(x.unique()))
        )
    )
    session_carted['avg_amt_carted'] = session_carted.total_amt_carted / session_carted.num_carted_items

    return session_carted


def user_session_overall(df):
    session = (
        df
            .groupby(['user_id', 'user_session'])
            .agg(
            num_events=('ones', sum),
            num_purchased_items=('purchase', sum),
            min_price=('price', min),
            max_price=('price', max),
            min_time=('event_time', min),
            max_time=('event_time', max)
        )
    )
    session['session_duration'] = (session.max_time - session.min_time).dt.total_seconds()
    session['session_purchase'] = np.where(session.num_purchased_items > 0, 1, 0)
    session['day'] = session.min_time.dt.day
    session['weekday'] = session.min_time.dt.weekday
    session['month'] = session.min_time.dt.month
    session['year'] = session.min_time.dt.year

    session = session.drop(['num_purchased_items'], axis=1)

    return session


def lag_time(df):
    shifted = df.groupby('user_id').shift(1)

    cols_to_keep = df.columns.tolist()
    cols_to_keep.append('min_time_lag')

    df_with_lag = df.join(shifted.rename(columns=lambda x: x + "_lag"))
    df_with_lag = df_with_lag[cols_to_keep]
    df_with_lag['time_since_last_visit_days'] = (
        (df_with_lag['min_time'] - df_with_lag['min_time_lag']).dt.total_seconds()).div(86400).round(0)
    df_with_lag.rename(columns={'min_time': 'session_timestamp'}, inplace=True)
    df_with_lag.drop(columns=['max_time', 'min_time_lag'], axis=1, inplace=True)
    df_with_lag.fillna(0, inplace=True)

    return df_with_lag


def make_pipeline(numeric_features, categorical_features, one_hot_features=None, scale=False):
    numeric_transformer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    scaler = StandardScaler()

    pipe_a_1 = Pipeline(steps=[('imp', numeric_transformer)])
    pipe_a_2 = Pipeline(steps=[('imp', numeric_transformer), ('scale', scaler)])
    pipe_b = Pipeline(steps=[('cat', categorical_transformer)])
    pipe_c = Pipeline(steps=[('imp', numeric_transformer)])

    if one_hot_features is None:
        if not scale:
            preprocess = ColumnTransformer(
                transformers=[
                    ('a1', pipe_a_1, numeric_features),
                    ('b', pipe_b, categorical_features)
                ]
            )
        else:
            preprocess = ColumnTransformer(
                transformers=[
                    ('a2', pipe_a_2, numeric_features),
                    ('b', pipe_b, categorical_features)
                ]
            )
    else:
        if not scale:
            preprocess = ColumnTransformer(
                transformers=[
                    ('a1', pipe_a_1, numeric_features),
                    ('b', pipe_b, categorical_features),
                    ('c', pipe_c, one_hot_features)
                ]
            )
        else:
            preprocess = ColumnTransformer(
                transformers=[
                    ('a2', pipe_a_2, numeric_features),
                    ('b', pipe_b, categorical_features),
                    ('c', pipe_c, one_hot_features)
                ]
            )

    return Pipeline(steps=[('preprocess', preprocess)])


def create_session_level_data(df, users, first_sessions):
    df_to_group = df.copy()
    df_to_group['ones'] = 1
    df_to_group['purchase'] = np.where(df_to_group.event_type == 'purchase', 1, 0)

    session_viewed = user_session_viewed(df_to_group)
    session_carted = user_session_carted(df_to_group)
    session_overall = user_session_overall(df_to_group)

    session_df = (
        session_overall
            .merge(session_viewed, left_index=True, right_index=True, how='left')
            .merge(session_carted, left_index=True, right_index=True, how='left')
    )
    session_df.reset_index(inplace=True)
    session_df.fillna(0, inplace=True)
    session_df.sort_values(['user_id', 'min_time'], inplace=True)

    session_df_with_lag = lag_time(session_df)
    session_df_with_lag.reset_index(drop=True)

    sessions = (
        session_df_with_lag
            .merge(users, left_on='user_id', right_on='user_id', how='left')
            .merge(first_sessions, left_on='user_session', right_on='user_session', how='left')
    )
    sessions.first_session.fillna(0.0, inplace=True)

    return sessions


def save_pickle_object(obj_to_pickle, pickle_file_path):
    joblib.dump(obj_to_pickle, pickle_file_path)


def load_pickled_object(pickle_file_path):
    return joblib.load(pickle_file_path)


def transform_x(x, pipeline, save_pickle=False, pickle_name=None):
    print(f'The shape of the X matrix before preprocessing: {x.shape}')
    pipeline.fit(x)

    if save_pickle:
        print(f'Saving off the fit pipeline: {pickle_name}')
        save_pickle_object(pipeline, pickle_name)

    x_transformed = pipeline.transform(x)
    print(f'The shape of the X matrix after preprocessing: {x_transformed.shape}')
    return x_transformed


def classification_results(y_test, preds):
    """returns accuracy, precision, recall, and f1 metrics"""
    cm = confusion_matrix(y_test, preds)
    acc = accuracy(y_test, preds)
    rec = recall(y_test, preds)
    prec = precision(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f'Accuracy = {acc}, Recall = {rec}, Precision = {prec}, F1-score = {f1}')
    print('Confusion Matrix is:')
    print(cm)


def top_k_viewed_products(df, k):
    products_viewed = df.loc[df.event_type == 'view']
    top_k_products_viewed = products_viewed.product_id.value_counts(normalize=True).index.tolist()[0:k]

    products = products_viewed.loc[products_viewed.product_id.isin(top_k_products_viewed)][
        ['user_session', 'product_id']].drop_duplicates()
    products['ones'] = 1

    user_products = products.pivot_table(index='user_session',
                                         columns='product_id',
                                         values='ones',
                                         fill_value=0).add_prefix('viewed_product_id_').reset_index()
    return user_products


def top_k_carted_products(df, k):
    products_carted = df.loc[df.event_type == 'cart']
    top_k_products_carted = products_carted.product_id.value_counts(normalize=True).index.tolist()[0:k]

    products = products_carted.loc[products_carted.product_id.isin(top_k_products_carted)][
        ['user_session', 'product_id']].drop_duplicates()
    products['ones'] = 1

    user_products = products.pivot_table(index='user_session',
                                         columns='product_id',
                                         values='ones',
                                         fill_value=0).add_prefix('carted_product_id_').reset_index()
    return user_products


def get_products():
    products = load_pickled_object(f"{DATA_PATH}/product_data.pkl")
    products.reset_index(inplace=True)

    return products


def get_top_products():
    top_products_columns = load_pickled_object(f"{DATA_PATH}/top_100_products.pkl").columns.tolist()
    top_product_ids = []
    for product in top_products_columns:
        if "product_id_" in product:
            top_product_ids.append(product.replace("product_id_", ""))

    return top_product_ids


def get_random_products(products):

    return random.choices(products, k=25)


def get_users():
    users = pandas.read_parquet(f"{DATA_PATH}/users.parquet")

    return users


def create_users_file():
    sample = load_pickled_object(f"{DATA_PATH}/ten_percent_sample.pkl")
    users_data = sample.user_id.unique()
    users = pandas.DataFrame(data=users_data, columns=["user_id"])
    users.to_parquet(f"{DATA_PATH}/users.parquet")


def get_random(df):
    random_value = random.choice(df)

    return random_value


def get_product_recommendations():
    product_recommendations = pandas.read_parquet(f"{DATA_PATH}/product_recommendations.parquet")

    return product_recommendations


def create_recommendations_file():
    data_dir = Path(f'{DATA_PATH}/product_recs.parquet')
    full_df = pandas.concat(
        pandas.read_parquet(parquet_file)
        for parquet_file in data_dir.glob('*.parquet')
    )
    full_df.to_parquet(f"{DATA_PATH}/product_recommendations.parquet")

# tests
def test_make_ordinal():
    for i in range(1, 46):
        print(make_ordinal(i))
