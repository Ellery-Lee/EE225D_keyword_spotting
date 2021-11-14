from audio import audio_extractor as aex
from visual import visual_extractor as vex
from config.attention_config import get_config

from visual.dataset_lrw1000 import LRW1000_Dataset as Dataset
from config.visual_config import getArgs
from visual.video_model import VideoModel
from visual.main_visual import feature_extractor
from torch.utils.data import DataLoader


# audio read data + feature extraction
config = get_config()
audio_data = aex.read_dataset(config)
stager, stage_op, train_filequeue_enqueue_op, melspec = audio_data.batch_input_queue()

# visual extractopm
feat = feature_extractor(True)

# 

