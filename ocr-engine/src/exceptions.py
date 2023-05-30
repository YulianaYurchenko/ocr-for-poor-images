class SRException(BaseException):
    error_code = None

    def __str__(self):
        return f'[E{self.error_code}] {self.message}'


class UnableToLoadModel(SRException):
    error_code = 1

    def __init__(self, reason):
        self.message = f'Unable to load model: {reason}'


class ModelIsNotLoaded(SRException):
    error_code = 2
    message = 'Model is not yet loaded. Call model.load() before inference'
