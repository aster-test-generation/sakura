"""
Config Module
"""

import json
import os
import re
from typing import Any
from pathlib import Path

import toml
from nltest.utils.exceptions import ConfigurationException
from nltest.utils.llm.model import Provider


def init_config(
    project_name: str,
    *,
    base_project_dir: str,
    output_dir: str,
    llm_provider: Provider,
    llm_model: str,
    emb_provider: Provider = None,
    emb_model: str = None,
    llm_api_url: str = None,  # Base URL, assuming OpenAI-API compatible endpoint
    emb_api_url: str = None,
    llm_api_key: str = None,
    emb_api_key: str = None,
    localization_max_iters: int = 40,
    composition_max_iters: int = 40,
    reuse_config: bool = False,
) -> "Config":
    config = Config(None, reuse=reuse_config)

    # Decided temperature assignments
    summarization_temp = 0.4
    code_gen_temp = 0.5
    decision_temp = 0.3
    structured_temp = 0.3

    config.set("llm", "provider", llm_provider.value)
    if emb_provider is not None:
        config.set("emb", "provider", emb_provider.value)
    else:
        config.set("emb", "provider", None)

    config.set("llm", "model", llm_model)
    config.set("llm", "summarization_temp", summarization_temp)
    config.set("llm", "code_gen_temp", code_gen_temp)
    config.set("llm", "decision_temp", decision_temp)
    config.set("llm", "structured_temp", structured_temp)
    # TODO: Decide of output tokens need to be set as config

    # Assign embedding settings
    config.set("emb", "model", emb_model)

    # Configure API URLs based on the provider
    if llm_api_url is not None:
        config.set("llm", "api_url", val=llm_api_url)
    elif llm_provider == Provider.OPENROUTER:
        config.set("llm", "api_url", val="https://openrouter.ai/api/v1")
    elif llm_provider == Provider.VLLM:
        config.set("llm", "api_url", val="http://localhost:8000/v1")

    config.set("llm", "api_key", val=llm_api_key)

    if emb_api_url is not None:
        config.set("emb", "api_url", val=emb_api_url)
    elif emb_provider == Provider.OPENROUTER:
        config.set("emb", "api_url", val="https://openrouter.ai/api/v1")
    elif emb_provider == Provider.VLLM:
        config.set("emb", "api_url", val="http://localhost:8000/v1")
    config.set("emb", "api_key", val=emb_api_key)

    # Assign localization agent settings
    # NOTE: Max iters should be higher for open-source models that do not support parallel tool calling
    config.set("localization", "max_iters", val=localization_max_iters or 20)

    # Assign composition agent settings
    config.set("composition", "max_iters", val=composition_max_iters or 20)

    # Assign project settings
    config.set("project", "base_project_dir", val=base_project_dir)
    config.set("project", "output_dir", val=output_dir)

    return config


class Config:
    """This is singleton class to hold the configuration information.

    By virtue of it's singleton nature, it can be configured once and used anywhere. Every time a new instance is created,
    we have overridden the `__new__` magic method to return a pre-existing instance of this class.
    """

    _instance = None

    # Save the configuration information across sessions.
    _LOCK_FILE = Path.cwd().joinpath(".aster.lock")

    def __new__(cls, conf_file: Path = None, reuse: bool = True):
        """Create a new instance of the config class

        Args:
            conf_file (Path, optional): Path to the configuration file. Defaults to None.
            reuse (bool, optional): True to reuse the lock file. Defaults to True.

        Returns:
            _type_: _description_
        """
        if not cls._instance:
            cls._instance = super(Config, cls).__new__(cls)

            # Initial setup
            cls._instance._conf_file = conf_file
            cls._instance._last_modified = None

            # Initial load
            cls._instance._load_config(reuse)

        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the singleton instance to None"""
        # pylint: disable=protected-access
        if cls._instance:
            cls._instance._conf_file = None
            cls._instance._last_modified = None
            cls._instance.config = {}
            cls._instance = None

    @classmethod
    def destroy(cls):
        """Remove the lock file and reset the singleton instance"""
        if cls._instance:
            if os.path.exists(cls._LOCK_FILE):
                os.remove(cls._LOCK_FILE)
            cls.reset()

    # pylint: disable=no-member,attribute-defined-outside-init)
    def _load_config(self, reuse: bool):
        """Load the configurations from the TOML file. If the lock file exists, then load that.
        If the file does not exist, return a FileNotFound error.
        If a file wasn't specified, then return an empty dict.

        Args:
            reuse (bool): Reuse forces the use of the lock file.

        Exceptions:
            FileNotFound: Thrown if config file is not found
        """
        self.config = {}

        # Check to see if the config file has been modified, if so, reload the file.
        if os.path.exists(self._LOCK_FILE) and reuse is True:
            with open(self._LOCK_FILE, "r", encoding="utf8") as lock_file:
                self.config = json.load(lock_file)
                return

        if self._conf_file:
            try:
                current_modified = os.path.getmtime(self._conf_file)
                if (
                    self._last_modified is None
                    or current_modified != self._last_modified
                    or reuse is False
                ):
                    self.config = toml.load(self._conf_file)
                    self.last_modified = current_modified
                    self._save_config_state()
            except FileNotFoundError as exc:
                raise ConfigurationException(
                    "",
                    message=f"Configuration file '{self._conf_file}' could not be found.",
                ) from exc

    def _save_config_state(self):
        pass
        # with open(self._LOCK_FILE, "w", encoding="utf8") as f:
        #     json.dump(self.config, f, indent=4)

    def _expand_env_variables(self, value):
        """Expand environment variables in a given value."""
        if isinstance(value, str):
            # Using regex for various formats like $VAR, env:VAR, ${VAR}
            return re.sub(
                r"(?i)\$(\w+)|env:(\w+)|\$\{(\w+)\}",
                lambda match: os.environ.get(
                    match.group(1) or match.group(2) or match.group(3), match.group(0)
                ),
                value,
            )
        return value

    def get(self, section: str, key: str) -> Any:
        """Get any value in a given section.

        Args:
            section (str): Configuration section.
            key (str): Configuration key.

        Returns:
            Any: Value associated with that section.
        """
        if section not in self.config:
            error_message = f'Group "{section}" is not found in config.'
            raise ConfigurationException("", message=error_message)

        if key not in self.config[section]:
            error_message = (
                f'Parameter "{key}" in group "{section}" is not found in config.'
            )
            raise ConfigurationException("", message=error_message)

        value = self._expand_env_variables(self.config[section][key])

        return value

    def set(self, section: str, key: str, val: Any) -> None:
        """Set any value in a given section.

        Args:
            section (str): Configuration section.
            key (str): Configuration key.
            value (str): Configuration value to be set.
        """
        if section not in self.config:
            self.config[section] = {}

        self.config[section][key] = val
        self._save_config_state()

    @property
    def LOCK_FILE(self):
        return self._LOCK_FILE
