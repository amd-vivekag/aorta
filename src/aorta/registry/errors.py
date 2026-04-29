"""Exceptions raised by the registry."""


class RegistryError(Exception):
    """Base class for all registry-related errors."""


class UnknownMitigationError(KeyError):
    """Raised when a mitigation name is not in the merged registry."""


class UnknownEnvironmentError(KeyError):
    """Raised when an environment name is not in the merged registry."""


class RegistryCollisionError(RegistryError):
    """Raised when two contributors register the same name in any registry."""
