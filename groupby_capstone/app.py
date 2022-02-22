import uuid
from time import sleep

from flask import Flask, render_template, request, jsonify, session

from groupby_capstone.enums.event_type import EventType
from groupby_capstone.settings import FLASK_HOST_NAME, FLASK_PORT, APP_FERNET_KEY, APP_ENV
from groupby_capstone.utils.data_utils import get_products, get_random_products, get_random
from groupby_capstone.utils.flask_form_classes.session_event import SessionEvent
from groupby_capstone.utils.grouby_capstone import run_prediction, predict, test_predict, products, test, users, \
    top_products, get_product_recs
from groupby_capstone.utils.structured_logger import StructuredLogger

logger = StructuredLogger()
flask_app = Flask(__name__)
flask_app.secret_key = APP_FERNET_KEY


def set_sess(session):
    if not session:
        session["user_session"] = str(uuid.uuid4())
        session["user_id"] = get_random(users.user_id)
        session["conversion_percentage"] = "unknown"

    return session


def update_session_events(event: SessionEvent):

    if session.get("session_events", False):
        session["session_events"].append(event.__dict__)

    else:
        session["session_events"] = [event.__dict__]

    if event.event_type == EventType.CART.value:
        if session.get("carted", False):
            session["carted"].append(event.__dict__)

        else:
            session["carted"] = [event.__dict__]

    elif event.event_type == EventType.REMOVE.value:
        updated_carted = []
        for carted_event in session.get("carted", []):
            if carted_event['product_id'] != event.product_id:
                updated_carted.append(carted_event)

        session["carted"] = updated_carted


    session["conversion_percentage"] = round(predict(session) * 100, 2)


# first route
@flask_app.route('/')
def index():
    if not session:
        session["user_session"] = str(uuid.uuid4())
        session["user_id"] = get_random(users.user_id)
        session["conversion_percentage"] = "unknown"


    product_ids = get_random_products(products.product_id)

    template = render_template(
        'index.html',
        the_title="groupby capstone project demo",
        user_session=session["user_session"],
        conversion_percentage=session["conversion_percentage"],
        product_ids=product_ids,
        top_products=top_products
    )

    return template


@flask_app.route('/view_product', methods=['POST'])
def view_product():
    if not session:
        session["user_session"] = str(uuid.uuid4())
        session["user_id"] = get_random(users.user_id)
        session["conversion_percentage"] = "unknown"

    product_id = request.form.to_dict()["product_id"]
    product = products.loc[products.product_id == product_id]

    session_event = SessionEvent(
        EventType.VIEW,
        product.product_id.iloc[0],
        product.category_id.iloc[0],
        product.category_code.iloc[0],
        product.brand.iloc[0],
        product.price.iloc[0],
        session["user_id"],
        session["user_session"]
    )
    update_session_events(session_event)
    product_recs = get_product_recs(product_id)

    template = render_template(
        'product.html',
        the_title="groupby capstone project demo",
        user_session=session["user_session"],
        conversion_percentage=session["conversion_percentage"],
        product_id=product_id,
        product_recs=product_recs
    )

    return template


@flask_app.route('/cart_product', methods=['POST'])
def cart_product():
    if not session:
        session["user_session"] = str(uuid.uuid4())
        session["user_id"] = get_random(users.user_id)
        session["conversion_percentage"] = "unknown"

    product_id = request.form.to_dict()["product_id"]
    product = products.loc[products.product_id == product_id]

    session_event = SessionEvent(
        EventType.CART,
        product.product_id.iloc[0],
        product.category_id.iloc[0],
        product.category_code.iloc[0],
        product.brand.iloc[0],
        product.price.iloc[0],
        session["user_id"],
        session["user_session"]
    )
    update_session_events(session_event)

    template = render_template(
        'cart.html',
        the_title="groupby capstone project demo",
        user_session=session["user_session"],
        conversion_percentage=session["conversion_percentage"],
        products=session.get("carted", [])
    )

    return template


@flask_app.route('/remove_product', methods=['POST'])
def remove_product():
    if not session:
        session["user_session"] = str(uuid.uuid4())
        session["user_id"] = get_random(users.user_id)
        session["conversion_percentage"] = "unknown"

    product_id = request.form.to_dict()["product_id"]
    product = products.loc[products.product_id == product_id]

    session_event = SessionEvent(
        EventType.REMOVE,
        product.product_id.iloc[0],
        product.category_id.iloc[0],
        product.category_code.iloc[0],
        product.brand.iloc[0],
        product.price.iloc[0],
        session["user_id"],
        session["user_session"]
    )
    update_session_events(session_event)

    template = render_template(
        'cart.html',
        the_title="groupby capstone project demo",
        user_session=session["user_session"],
        conversion_percentage=session["conversion_percentage"],
        products=session.get("carted", [])
    )

    return template


@flask_app.route('/api/v1/predict', methods=['GET', 'POST'])
def model_predict():
    content = request.json
    return jsonify({"content": content})


def test_model_predict():
    test_predict("gradient_boost_model")


def run(command):
    if command == 'webserver':
        if APP_ENV == "DEV":
            flask_app.run(host=FLASK_HOST_NAME, port=FLASK_PORT, debug=True)

        else:
            flask_app.run(host=FLASK_HOST_NAME, port=FLASK_PORT)

    elif command == 'test_predict':
        # test_model_predict({"test": "Prediction test"})
        test_model_predict()

    elif command == 'test':
        # test_model_predict({"test": "Prediction test"})
        test()
