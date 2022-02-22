import requests
import pandas as pd

from groupby_capstone.enums.event_type import EventType
from groupby_capstone.settings import FLASK_HOST_NAME, FLASK_PORT, DATA_PATH, MODEL_PATH
from groupby_capstone.utils.data_utils import load_pickled_object, create_session_level_data, get_products, \
    get_product_recommendations, get_users, get_random, get_random_products, get_top_products
from groupby_capstone.utils.decorators.profile import profile
from groupby_capstone.utils.flask_form_classes.session_event import SessionEvent


gradient_boost_model = load_pickled_object(f"{MODEL_PATH}/gradient_boost_model.pkl")
products = get_products()
users = get_users()
pipeline = load_pickled_object(f"{DATA_PATH}/ten_percent_transformer_pipeline.pkl")
top_100_products = load_pickled_object(f"{DATA_PATH}/top_100_products.pkl")
top_products = get_top_products()
product_recommendations = get_product_recommendations()
user_types = load_pickled_object(f"{DATA_PATH}/user_types.pkl")
first_sessions = load_pickled_object(f"{DATA_PATH}/first_sessions.pkl")
ten_percent_sample = load_pickled_object(f"{DATA_PATH}/ten_percent_sample.pkl")


def score_new_data(df):
    list_products = df.product_id.to_list()
    new_session_agg = create_session_level_data(df, user_types, first_sessions)
    top_products_new_session = top_100_products.iloc[[0]].copy()
    top_products_new_session.user_session = new_session_agg.user_session
    top_product_ids = [x.split('_')[-1] for x in top_100_products.columns[1:]]

    for product in list_products:
        if product in top_product_ids:
            top_products_new_session[f'product_id_{product}'] = 1
        else:
            top_products_new_session[f'product_id_{product}'] = 0

    session = new_session_agg.merge(top_products_new_session, left_on='user_session', right_on='user_session',
                                    how='left')
    x = session.drop(['session_purchase', 'user_id', 'user_session', 'session_timestamp'], axis=1)

    x_transformed = pipeline.transform(x)
    preds = gradient_boost_model.predict_proba(x_transformed)
    likelyhood = preds.flatten()[-1]

    # print(f'The probablity of conversion after {len(df)} events is: {likelyhood}')

    return likelyhood


def run_prediction(input_data: dict):
    res = requests.post(f'http://{FLASK_HOST_NAME}:{FLASK_PORT}/api/v1/predict', json=input_data)
    if res.ok:
        print(res.json())


def get_product_recs(product_id: str):
    related_product_ids = []
    product_recs = product_recommendations[product_recommendations.product_id == int(product_id)].recommendations

    for product_recommendation in product_recs:
        for rec in product_recommendation:
            related_product_ids.append(rec["related_product_id"])

    if not related_product_ids:
        related_product_ids = top_products[:10]

    return related_product_ids


def predict(input_data: dict):
    session_events = input_data["session_events"]
    df = pd.DataFrame(session_events)
    pred = score_new_data(df)

    return pred

@profile
def test():
    product_id = "3762"

    related_product_ids = []
    product_recs = product_recommendations[product_recommendations.product_id == 3762].recommendations

    for product_recommendation in product_recs:
        for rec in product_recommendation:
            related_product_ids.append(rec["related_product_id"])

    return related_product_ids


def test_predict():
    df = ten_percent_sample

    one_session = df.loc[df.user_session == 'e161dc53-6f81-4d28-ac14-28fe9d6e8830']
    one_session = one_session.sort_values('event_time')
    session_events = []

    top_products_columns = top_100_products.columns.tolist()
    top_product_ids = []
    for product in top_products_columns:
        if "product_id_" in product:
            top_product_ids.append(product.replace("product_id_", ""))

    for id, one_sess in one_session.iterrows():
        if one_sess.event_type != EventType.PURCHASE.value:
            product_id = one_sess.product_id
            print(f"New session event {one_sess.event_type}: product_id: {product_id}")
            session_event = SessionEvent(
                event_type=EventType(one_sess.event_type),
                product_id=product_id,
                category_id=one_sess.category_id,
                category_code=one_sess.category_code,
                brand=one_sess.brand,
                price=one_sess.price,
                user_id=one_sess.user_id,
                user_session=one_sess.user_session
            )

            session_events.append(session_event.__dict__)
            session_df = pd.DataFrame(session_events)
            session_df = session_df.sort_values('event_time')
            score_new_data(session_df)
