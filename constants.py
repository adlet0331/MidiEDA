FPS = 30 # Video FPS
FRAME_TIME = 1 / FPS # 0.0333..
SAMPLE_RATE = 16020
HOP_LENGTH = round(SAMPLE_RATE * FRAME_TIME) # 534 # 32 * SAMPLE_RATE // 1000 # 32 ms 
ONSET_LENGTH = HOP_LENGTH # 534
OFFSET_LENGTH = HOP_LENGTH # 534
HOPS_IN_ONSET = ONSET_LENGTH // HOP_LENGTH # 1
HOPS_IN_OFFSET = OFFSET_LENGTH // HOP_LENGTH # 1

MIN_MIDI = 21
MAX_MIDI = 108

N_MELS = 229 # 256
MEL_FMIN = 30
MEL_FMAX = SAMPLE_RATE // 2 # 8010
WINDOW_LENGTH = round(HOP_LENGTH * 4) # 2136

RANDOM_SEED = 42
TRAIN_TEST_VAL_SPLIT = [0.8, 0.1, 0.1]

CIPI_PATH = '/Users/simhyeongju/AVAPT/data/CIPI/'
CIPI_METADATA_FILE = '/Users/simhyeongju/AVAPT/data/CIPI/index.json'

MIKROKOSMOS_PATH = '/Users/simhyeongju/AVAPT/data/Mikrokosmos-difficulty/'
MIKROKOSMOS_MIDI_FOLDER = '/Users/simhyeongju/AVAPT/data/Mikrokosmos-difficulty/midi/'
MIKROKOSMOS_METADATA_FILE = '/Users/simhyeongju/AVAPT/data/Mikrokosmos-difficulty/metadata/henle_mikrokosmos.json'

MODEL_SAVE_PATH = "/Users/simhyeongju/AVAPT/EDA/pretrained_model/best_model-53900step-valLoss0.053890.pt"

PIANOVAM_AUDIO_INPUT_FOLDER = '/Users/simhyeongju/AVAPT/data/pianovam/audio/'
PIANOVAM_GROUND_TRUTH_FOLDER = '/Users/simhyeongju/AVAPT/data/pianovam/midi/'

AUDIO_TRANSCRIBE_OUTPUT_FOLDER = '/Users/simhyeongju/AVAPT/EDA/_transcribed_MIDI/'
AUDIO_TRANSCRIBE_BATCH_SIZE = 24
AUDIO_TRANSCRIBE_SEGMENT_LENGTH_SEC = 5

SEGMENT_LEN_SEC = 5

PREDICTED_FOLDER = "/Users/simhyeongju/AVAPT/EDA/_transcribed_MIDI/OnsetsAndFrames_2_5sec"

SF2_PATH = '/Users/simhyeongju/AVAPT/EDA/FluidR3_GM.sf2'

import torch
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'