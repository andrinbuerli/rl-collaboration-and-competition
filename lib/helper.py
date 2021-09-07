import argparse


def extract_properties_from(obj):
    return {
        key: value for key, value in obj.__dict__.items()
        if type(value) == int or type(value) == float or type(value) == str
    }


def extract_config_from(*config_objects):
    config = dict()
    for obj in config_objects:
        if type(obj) == dict:
            config.update(obj)
        else:
            properties = extract_properties_from(obj)
            config.update(properties)

    return config


def parse_config_for(program_name, config_objects: dict):
    parser = argparse.ArgumentParser(prog=program_name)

    for key, value in config_objects.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)

    return parser.parse_args()
