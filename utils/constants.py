import os
from datetime import datetime 

date_time = datetime.now().strftime("%m-%d-%Y")

CWD = os.getcwd()
MP4_PATH = os.path.join(CWD, 'data')
OUTPUT_PATH = os.path.join(CWD, 'results', date_time) 

GENERATED_DATA_PATH = os.path.join(CWD, 'generated')
PREPROCESSOR_DICT = {
    'depth_zoe': 'lllyasviel/control_v11f1p_sd15_depth',
    'optical_flow': 'raft_from_torchvision',
}

MODEL_IDS = {
    'SD 1.5': 'None'
}

