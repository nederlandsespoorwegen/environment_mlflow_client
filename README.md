# Environment MLFlow Client

Here we extend on the standard MLFlow client to manage different environments with the same MLFlow instance, which mainly involves the model registry and experiment management.
Our goal is to run multiple logical environments (acc, preprod, prod) in the same databricks workspace with proper permission controls. We wrote a blog about the combination of our MLFlow client and the basic permission structure that is available with the terraform Databricks provider.

## Features

1. abstraction for environment scoped model names
1. helper function for logging a model and registering a model version
1. automatic model stage assignment based on the environment
1. abstraction for environment scoped experiment folders
1. methods for common usage patterns (f.i. load latest model version of any model flavor)

## Usage

```
>>pip install environment-mlflow-client
```

Python:

```
from environment_mlflow_client import EnvMlflowClient

model_name = "deepar"

mlflow_client = EnvMlflowClient(env_name="test")

model_versions = mlflow_client.get_latest_versions(name=model_name)
```

## Compatibility

Compatible with MLFlow 1.27.0 and higher as there was some reshuffling of the MLFlow entities, such as ModelVersion.

## Unit tests

A fixture is included that starts a local MLFlow instance and cleans it up after the testing session is finished.
The unit tests are thus conducted against the MLFlow API to validate our client.

## Pipeline

Github actions are triggered on pull requests to validate the code change against the unit tests.
When a commit is tagged on main a Python wheel is build and published to pypi and github releases.
