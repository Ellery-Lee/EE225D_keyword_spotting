import sys
sys.path.append("..")
import audio_extractor as aex
from config.attention_config import get_config

# audio read data
config = get_config()
audio_data = aex.read_dataset(config)

# audio extraction
stager, stage_op, train_filequeue_enqueue_op, melspec = audio_data.batch_input_queue()

