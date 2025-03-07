import json
import os

def load_json(file_path):
    """
    Load the X2XBenchmark.json file
    """
    with open(file_path) as f:
        data = json.load(f)
    return data