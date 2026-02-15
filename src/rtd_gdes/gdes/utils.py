# Utilities for GDES
# e.g. exceptions

class MixedPrecisionSelectionError(Exception):
    def __init__(self, msg):
        self.msg = msg
        print("Cannot select multiple mixed precision settings at once")
