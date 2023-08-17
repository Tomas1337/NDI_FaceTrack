import configparser
import os

DEFAULT_CONFIG_FILE = 'config.ini'

def get_config_file():
    return os.environ.get('CONFIG_FILE', DEFAULT_CONFIG_FILE)

CONFIG_FILE = get_config_file()

def create_config(config_file=None):
    parser = configparser.ConfigParser()
    parser.read(config_file or CONFIG_FILE)
    return parser

CONFIG = create_config()
TRACK_TYPE_DICT = {
    0: 'face',
    1: 'body',
    2: 'fallback'
    }