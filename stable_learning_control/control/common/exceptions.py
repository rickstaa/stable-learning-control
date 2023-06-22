"""Module containing several control related exceptions."""


class EnvLoadError(Exception):
    """Custom exception that is raised when the saved environment could not be loaded.

    Attributes:
        log_message (str): The full log message.
        details (dict): Dictionary containing extra Exception information.
    """

    def __init__(self, message="", log_message="", **details):
        """Initializes the EePoseLookupError exception object.

        Args:
            message (str, optional): Exception message specifying whether the exception
                occurred. Defaults to ``""``.
            log_message (str, optional): Full log message. Defaults to ``""``.
            details (dict): Additional dictionary that can be used to supply the user
                with more details about why the exception occurred.
        """
        super().__init__(message)

        self.log_message = log_message
        self.details = details


class PolicyLoadError(Exception):
    """Custom exception that is raised when the saved policy could not be loaded.

    Attributes:
        log_message (str): The full log message.
        details (dict): Dictionary containing extra Exception information.
    """

    def __init__(self, message="", log_message="", **details):
        """Initializes the EePoseLookupError exception object.

        Args:
            message (str, optional): Exception message specifying whether the exception
                occurred. Defaults to ``""``.
            log_message (str, optional): Full log message. Defaults to ``""``.
            details (dict): Additional dictionary that can be used to supply the user
                with more details about why the exception occurred.
        """
        super().__init__(message)

        self.log_message = log_message
        self.details = details
