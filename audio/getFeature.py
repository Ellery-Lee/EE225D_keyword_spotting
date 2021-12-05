import sys
sys.path.append('../')
from config.attention_config import get_config
from audio import audio_extractor as aex


# audio read data
config = get_config()
audio_data = aex.read_dataset(config)

# audio extraction
stager, stage_op, train_filequeue_enqueue_op, melspec = audio_data.batch_input_queue()

if __name__ == '__main__':
    print(melspec)