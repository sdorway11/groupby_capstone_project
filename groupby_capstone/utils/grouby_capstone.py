import pandas as pd
import requests

from groupby_capstone.settings import FLASK_HOST_NAME, FLASK_PORT


def run_prediction(input_data):
    res = requests.post(f'http://{FLASK_HOST_NAME}:{FLASK_PORT}/api/v1/predict', json=input_data)
    if res.ok:
        print(res.json())