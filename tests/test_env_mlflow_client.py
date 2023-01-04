import mlflow
import pytest

from environment_mlflow_client import EnvMlflowClient

TEST_MODEL_NAME = "test_model_name"
ENV_NAME = "local"


class FakeModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        return None


@pytest.fixture(autouse=True, scope="module")
def log_model(run_mlflow):
    model_flavor = mlflow.pyfunc
    client = EnvMlflowClient(env_name=ENV_NAME)

    experiment_id = client.create_experiment_if_not_exists("unittest")

    with mlflow.start_run(run_name="unittest_training", experiment_id=experiment_id):
        model_version, model_info = client.log_model_helper(
            model_flavor=model_flavor,
            registered_model_name=TEST_MODEL_NAME,
            artifact_path=TEST_MODEL_NAME,
            python_model=FakeModel(),
        )
    assert model_version.name == client.get_env_model_name(TEST_MODEL_NAME)
    assert model_info.artifact_path == client.get_env_model_name(TEST_MODEL_NAME)
    assert model_version.current_stage == "Staging"


def test_get_env_model_name():
    """test the environment specific model name"""
    client = EnvMlflowClient(env_name=ENV_NAME)
    model_name_env = client.get_env_model_name(TEST_MODEL_NAME)
    assert model_name_env == f"{TEST_MODEL_NAME}_{ENV_NAME}"


def test_load_model_version():
    """test load model version"""
    model_flavor = mlflow.pyfunc
    client = EnvMlflowClient(env_name=ENV_NAME)
    model = client.load_model_version(model_flavor, TEST_MODEL_NAME, version="1")

    assert hasattr(model, "predict")


def test_get_latest_versions():
    """tes get the latest model versions"""
    client = EnvMlflowClient(env_name=ENV_NAME)
    versions = client.get_latest_versions(TEST_MODEL_NAME)
    assert str(versions[0].version) == "1"


def test_get_model_version():
    """test get specific model version"""
    client = EnvMlflowClient(env_name=ENV_NAME)
    model_version = client.get_model_version(name=TEST_MODEL_NAME, version="1")
    assert model_version.name == client.get_env_model_name(TEST_MODEL_NAME)


def test_set_model_version_tag():
    """test set model version tag"""
    client = EnvMlflowClient(env_name=ENV_NAME)
    client.set_model_version_tag(
        name=TEST_MODEL_NAME, version="1", key="dorst", value="bier"
    )

    model_version = client.get_model_version(name=TEST_MODEL_NAME, version="1")
    assert model_version.tags["dorst"] == "bier"


def test_set_registered_model_tag():
    """test set registered model tag"""
    client = EnvMlflowClient(env_name=ENV_NAME)
    client.set_registered_model_tag(name=TEST_MODEL_NAME, key="olie", value="bollen")
    registered_model = client.get_registered_model(name=TEST_MODEL_NAME)

    assert registered_model.tags["olie"] == "bollen"


def test_create_registered_model():
    "test create registered model"
    test_registered_model_name = "create_registered_model_test"
    client = EnvMlflowClient(env_name=ENV_NAME)
    registered_model = client.create_registered_model(name=test_registered_model_name)
    assert registered_model.name == client.get_env_model_name(
        test_registered_model_name
    )


def test_create_experiment_if_not_exists():
    """test experiment creation"""
    client = EnvMlflowClient(env_name=ENV_NAME)
    experiment_id = client.create_experiment_if_not_exists("experiment1")
    assert (
        experiment_id is not None
    ), "create_experiment_if_not_exists does not return an experiment id"


def test_create_experiment_if_already_exists():
    """test experiment retrieval"""
    client = EnvMlflowClient(env_name=ENV_NAME)
    first_id = client.create_experiment_if_not_exists("experiment1")
    _ = client.create_experiment_if_not_exists("other")
    second_id = client.create_experiment_if_not_exists("experiment1")
    assert (
        first_id == second_id
    ), "create_experiment_if_not_exists does not return existing experiment id"


def test_get_env_experiment_name():
    """test get experiment environment name"""
    client = EnvMlflowClient(env_name="acc")
    assert "/experiments/acc/experiment1" == client.get_env_experiment_name(
        "experiment1"
    )
    client = EnvMlflowClient(env_name="prod")
    assert "/experiments/prod/experiment1" == client.get_env_experiment_name(
        "experiment1"
    )
