"""mlflow client that is aware of the application environment, main target is databricks mlflow"""
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import mlflow
from mlflow.entities.model_registry import ModelVersion, RegisteredModel
from mlflow.models.model import ModelInfo


class EnvMlflowClient(mlflow.tracking.MlflowClient):
    """
    Class inherits from mlflow client and contextualizes methods to the current logical environment
    Custom functionality is added to:
    * load a (latest) model version
    * log and register a model version and set the stage property

    """

    ENVIRONMENT_KEY = "MLFLOW_ENV"

    def __init__(
        self,
        env_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None,
    ):
        """
        For databricks we do not provide these arguments but rely on environment variables.
        Args:
            env_name: environment name
            tracking_uri: Address of local or remote tracking server.
            registry_uri: Address of local or remote model registry server.
        """
        super().__init__(tracking_uri, registry_uri)
        if env_name is None and not self.ENVIRONMENT_KEY in os.environ:
            raise ValueError(f"pass env_name or set {self.ENVIRONMENT_KEY} in env.")
        self.env_name = env_name if env_name else os.environ[self.ENVIRONMENT_KEY]
        self.stage_lookup = defaultdict(
            lambda: "Staging"
        )  # staging for all envs except production
        self.stage_lookup["production"] = "Production"

    def get_env_model_name(self, name: str) -> str:
        """postfix model names with the environment"""
        return f"{name}_{self.env_name}"

    def get_latest_versions(self, name: str) -> List[ModelVersion]:
        """
        Get latest model version.
        The stage parameter is not supported as we set its value.
        """
        name = self.get_env_model_name(name)
        return super().get_latest_versions(
            name, stages=[self.stage_lookup[self.env_name]]
        )

    def get_latest_model_version(self, name: str) -> ModelVersion:
        """
        Retrieve the latest model version

        Args:
            name: Name of the model
        Returns:
            Latest ModelVersion
        """
        return self.get_latest_versions(name)[0]

    def set_model_version_tag(
        self, name: str, version: str, key: str, value: Any
    ) -> None:
        """
        Set a tag on a model version
        The stage parameter is not supported as we set its value.
        """
        name = self.get_env_model_name(name)
        super().set_model_version_tag(name, version, key, value)

    def set_registered_model_tag(self, name: str, key: str, value: Any) -> None:
        """Set a tag on a registered model"""
        name = self.get_env_model_name(name)
        super().set_registered_model_tag(name, key, value)

    def create_model_version(
        self,
        name: str,
        source: str,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        run_link: Optional[str] = None,
        description: Optional[str] = None,
        await_creation_for: int = ...,
    ) -> ModelVersion:

        name = self.get_env_model_name(name)
        return super().create_model_version(
            name, source, run_id, tags, run_link, description, await_creation_for
        )

    def create_registered_model(
        self,
        name: str,
        tags: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> RegisteredModel:
        name = self.get_env_model_name(name)
        return super().create_registered_model(name, tags, description)

    def get_model_version_download_uri(self, name: str, version: str) -> str:
        name = self.get_env_model_name(name)
        return super().get_model_version_download_uri(name, version)

    def get_registered_model(self, name: str) -> RegisteredModel:
        """Get a registered model by name"""
        name = self.get_env_model_name(name)
        return super().get_registered_model(name)

    def transition_model_version_stage(self, name: str, version: str) -> ModelVersion:
        """
        Set the stage of a registered model.
        More than one model can be in one stage.
        We do not support the stage and archive_existing_versions paramters,
        as we set to those.
        """
        name = self.get_env_model_name(name)
        return super().transition_model_version_stage(
            name=name,
            version=version,
            stage=self.stage_lookup[self.env_name],
            archive_existing_versions=False,
        )

    def get_model_version(self, name: str, version: str) -> ModelVersion:
        """Get a specific ModelVersion object"""
        name = self.get_env_model_name(name)
        return super().get_model_version(name=name, version=version)

    def load_model_version(
        self, model_flavor, name: str, version: str, unwrap_model: bool = False
    ) -> Any:
        """Load a model version within the specified stage"""
        model_version = self.get_model_version(
            name=name,
            version=version,
        )
        model = model_flavor.load_model(model_uri=model_version.source)
        if unwrap_model:
            model = model._model_impl  # retrieve custom model implementation
        return model

    def load_latest_model(
        self, model_flavor, name: str, unwrap_model: bool = False
    ) -> Any:
        """
        Load the latest version of the pyfunc model with the given name.

        Args:
            model_flavor: i.e. mlflow.pyfunc or mlflow.spark
            name: Name of the model

        Returns:
            The loaded model
        """
        latest_versions = self.get_latest_versions(name)
        model = model_flavor.load_model(latest_versions[0].source)
        if unwrap_model:
            model = model._model_impl  # retrieve custom model implementation
        return model

    def log_model_helper(
        self, model_flavor: Any, registered_model_name: str, **kwargs
    ) -> Tuple[ModelVersion, ModelInfo]:
        """
        Standardize model logging setting an environment aware name
        and stage attribute.
        All parameters are passed to the log_model of the model_flavor.

        Args:
            model_flavor: i.e. mlflow.pyfunc
            registered_model_name: base model name
            kwargs: parameters for log_model minus registered_model_name, see documentation of model_flavor
            i.e. mlflow.sklearn.log_model
        Returns:
            Tuple of a mlflow.entities.model_registry.ModelVersion object and mlflow.models.model.ModelInfo object.
        """
        if "artifact_path" in kwargs:
            kwargs["artifact_path"] = self.get_env_model_name(kwargs["artifact_path"])
        # log model
        model_info = model_flavor.log_model(**kwargs)
        # register model
        registered_model_name_env = self.get_env_model_name(registered_model_name)
        model_version = mlflow.register_model(
            model_uri=model_info.model_uri, name=registered_model_name_env
        )
        # set stage attribute on model version
        model_version = self.transition_model_version_stage(
            registered_model_name, model_version.version
        )
        return model_version, model_info

    def get_env_experiment_name(self, name: str) -> str:
        """Get environment specific experiment name."""
        return f"/experiments/{self.env_name}/{name}"

    def create_experiment_if_not_exists(self, name: str) -> str:
        """Create MLflow experiment if not exists."""
        name = self.get_env_experiment_name(name)
        try:
            experiment_id = mlflow.create_experiment(name=name)
        except mlflow.exceptions.MlflowException:
            experiment = mlflow.get_experiment_by_name(name)
            experiment_id = experiment.experiment_id
        return experiment_id
