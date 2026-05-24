# Model Configuration

D_MODEL = 256
NUM_HEADS = 8
NUM_LAYERS = 4
D_FF = 1024
DROPOUT = 0.1

# Training Configuration

BATCH_SIZE = 32
EPOCHS = 25

LEARNING_RATE = 1e-4
WARMUP_STEPS = 4000

LABEL_SMOOTHING = 0.1
GRAD_CLIP = 1.0

# Special Tokens

UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

# Checkpoints

CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = "checkpoints/best_model.pt"

# W&B

WANDB_PROJECT = "da6401_transformer"
WANDB_RUN_NAME = "autograder_fix_final"