import os

from groupby_capstone.utils.settings import get

'''These are the most common settings used. Please update to include any environment variables needed
proper usage:
#At top of file
from .settings import *

#within file
with psycopg2.connect(
    host=REDSHIFT_HOST,
    user=REDSHIFT_USER,
    #.... etc
'''
# Flask Settings
FLASK_HOST_NAME = get("FLASK_HOST_NAME", "127.0.0.1")
FLASK_PORT = get("FLASK_PORT", "8080")
APP_FERNET_KEY = get("APP_FERNET_KEY", "YODVCbNyDm5jbl-SX9p4CktRBlxsaXSEKD-LG1hr3vQ=")

# Directory settings
WORKING_DIR = os.getcwd()
DATA_PATH = get("DATA_PATH", f"{WORKING_DIR}/data")
MODEL_PATH = get("MODEL_PATH", f"{WORKING_DIR}/models")
APP_ENV = get("APP_ENV", f"DEV")
