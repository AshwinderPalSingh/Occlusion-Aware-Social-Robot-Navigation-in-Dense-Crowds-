def warn(msg):
    print('WARNING:', msg)
def log(msg):
    print(msg)
class _ScopedConfigure:
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False
def scoped_configure(*args, **kwargs):
    return _ScopedConfigure(*args, **kwargs)
