import torch

from dataset import (
    tokenize_de,
    text_to_tensor,
    src_vocab,
    tgt_vocab,
    EOS_IDX
)

from model import Transformer

from config import (
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS,
    D_FF,
    DROPOUT,
    BEST_MODEL_PATH
)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model = Transformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    d_ff=D_FF,
    dropout=DROPOUT
).to(device)

model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location=device
    )
)

model.eval()

sentence = "ein kleines mädchen spielt"

src_tensor = text_to_tensor(
    tokenize_de(sentence),
    src_vocab
).unsqueeze(0).to(device)

prediction = model.infer(src_tensor)

translated_words = []

for token in prediction:

    if token == EOS_IDX:
        break

    if token > 3:

        translated_words.append(
            tgt_vocab.lookup_token(token)
        )

translation = " ".join(translated_words)

print("\nGerman Input:")
print(sentence)

print("\nPredicted Translation:")
print(translation)