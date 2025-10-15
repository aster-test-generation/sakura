"""
Exceptions Package
"""


class CustomBaseException(Exception):
    """Base exception"""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ConfigurationException(CustomBaseException):
    """Error thrown when configuration parameters are not set or when agent policy is violated."""

    def __init__(self, config_var, message=None, *, details=None) -> None:
        if not message:
            message = (
                f"The configuration parameter {config_var} has not been set. "
                f"Please set the configuration parameter {config_var} before proceeding."
            )
        super().__init__(message)
        self.config_var = config_var
        self.details = details
