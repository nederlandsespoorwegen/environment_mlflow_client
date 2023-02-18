import json
import pickle
from pathlib import Path


def _load_pyfunc(path: str):
    """
    This function is required by mlflow for loading the model, and returns an object with a 'predict' function
    """
    table_path = Path(path) / "table.json"

    with open(table_path, mode="r", encoding="utf-8") as table_file:
        lookup_table = json.load(table_file)

    model_path = Path(path) / "model.p"
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    model.lookup_table = lookup_table

    return model
