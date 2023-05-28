import json
from datetime import datetime

def read_config(path):
    with open(path) as f:
        data = json.load(f)
    return data

def get_current_time():
    return "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
