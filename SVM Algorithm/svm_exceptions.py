class IsNotSVCException(Exception):
    def __init__(self, message):
        self.message = message


class PathExistsException(Exception):
    def __init__(self, message):
        self.message = message


class WrongExtensionException(Exception):
    def __init__(self, message):
        self.message = message


class PathNotExistsException(Exception):
    def __init__(self, message):
        self.message = message