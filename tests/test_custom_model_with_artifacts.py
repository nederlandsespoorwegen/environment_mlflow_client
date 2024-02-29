import json
import pickle
import sys
import tempfile
from pathlib import Path

import mlflow
import pytest

from environment_mlflow_client import EnvMlflowClient

sys.path.append("./tests/model")  # in order find model_loader module

TEST_MODEL_NAME = "test_custom_model_with_artifacts"
ENV_NAME = "local"
LOOKUP_TABLE = {"column_1": [1, 2, 3]}


class FakeModel(mlflow.pyfunc.PythonModel):
    """Fake model to log"""

    def predict(self, context, model_input):  # pylint: disable=unused-argument
        return None


@pytest.fixture(autouse=True, scope="module")
def log_model_with_artifacts():
    """Log a custom pyfunc model and its artifacts"""
    model_flavor = mlflow.pyfunc
    client = EnvMlflowClient(env_name=ENV_NAME)

    experiment_id = client.create_experiment_if_not_exists("unittest")

    with mlflow.start_run(run_name="unittest_training", experiment_id=experiment_id):
        with tempfile.TemporaryDirectory() as model_dir:
            table_path = Path(model_dir) / "table.json"

            with open(table_path, mode="w", encoding="utf-8") as table_file:
                table = LOOKUP_TABLE
                json.dump(table, table_file)

            model_path = Path(model_dir) / "model.p"
            with open(model_path, "wb") as model_file:
                model = FakeModel()
                pickle.dump(model, model_file)

            model_code_path = Path(__file__).parent / "model"

            model_version, model_info = client.log_model_helper(
                model_flavor=model_flavor,
                registered_model_name=TEST_MODEL_NAME,
                loader_module="model_loader",
                artifact_path=TEST_MODEL_NAME,
                data_path=model_dir,
                code_path=[model_code_path],
            )
    assert model_version.name == client.get_env_model_name(TEST_MODEL_NAME)
    assert model_info.artifact_path == client.get_env_model_name(TEST_MODEL_NAME)
    assert model_version.current_stage == "Staging"


def test_custom_model_with_artifacts():
    """test loading of underlying model with artifacts with custom lookup table"""
    client = EnvMlflowClient(env_name="local")
    model = client.load_latest_model(
        model_flavor=mlflow.pyfunc, name=TEST_MODEL_NAME, unwrap_model=True
    )
    assert model.lookup_table == LOOKUP_TABLE
    assert hasattr(model, "predict")


def test_wrapped_model_with_artifacts():
    """test loading of PyFuncModel with artifacts"""
    client = EnvMlflowClient(env_name="local")
    model = client.load_latest_model(model_flavor=mlflow.pyfunc, name=TEST_MODEL_NAME)
    assert hasattr(model, "lookup_table") is False
    assert hasattr(model, "predict")
