from time import sleep

from flask import Flask, render_template, request, jsonify

from groupby_capstone.settings import FLASK_HOST_NAME, FLASK_PORT
from groupby_capstone.utils.grouby_capstone import run_prediction
from groupby_capstone.utils.structured_logger import StructuredLogger

logger = StructuredLogger()
flask_app = Flask(__name__)


# first route
@flask_app.route('/')
def index():
    return render_template('index.html', the_title="groupby capstone project")


@flask_app.route('/api/v1/predict', methods=['GET', 'POST'])
def model_predict():
    content = request.json
    print(content)
    return jsonify({"content": content})


def run(command):
    print(command)
    if command == 'webserver':
        flask_app.run(host=FLASK_HOST_NAME, port=FLASK_PORT, debug=True)

    elif command == 'test_predict':
        run_prediction({"test": "Prediction test"})
