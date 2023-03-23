import importlib


def get_model(model_name):
    model_file_name = model_name.lower()

    module_path = '.'.join(['model', model_file_name])
    print(module_path)
    if importlib.util.find_spec(module_path, __name__):
        model_module = importlib.import_module(module_path, __name__)
    else:
        raise ValueError('`model_name` [{}] is not the name of an existing model.'.format(model_name))

    model_class = getattr(model_module, model_name)
    return model_class


def is_denoise_model(model_name):
    return 'Denoise' in model_name
