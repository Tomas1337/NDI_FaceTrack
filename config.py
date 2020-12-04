import configparser
import os

my_path = os.path.abspath(os.path.dirname(__file__))
DEFAULT_CONFIG_FILE = 'config.ini'
config_path = os.path.join(my_path, DEFAULT_CONFIG_FILE)

def get_config_file():
    return os.environ.get('CONFIG_FILE', config_path)

CONFIG_FILE = get_config_file()

def create_config(config_file=None):
    parser = configparser.ConfigParser()
    parser.read(CONFIG_FILE)
    return parser

CONFIG = create_config()