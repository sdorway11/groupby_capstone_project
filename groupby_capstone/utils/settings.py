import os


def get_required(var_name):
    var = os.environ.get(var_name)
    if var is None:
        raise Exception(f'Required parameter {var_name} has no value')
    return var


def get(var_name, default=None):
    var = os.environ.get(var_name)
    if var is None:
        var = default
    return var
