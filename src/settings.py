import os

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
RESOURCE_DIR = os.path.join(PROJECT_DIR, '.resources')
CXR8_DIR = os.environ.get("CXR8_DIR", "/datasets/CXR8")
RSNA_DIR = os.environ.get('RSNA_DIR', '/datasets/RSNA')
WANDB_PROJECT = os.environ.get('WANDBPROJECT', 'WSRPN')
WANDB_ENTITY = os.environ.get('WANDBENTITY')
MODELS_DIR = os.environ.get('LOG_DIR', os.path.join(PROJECT_DIR, 'logs', 'models'))
