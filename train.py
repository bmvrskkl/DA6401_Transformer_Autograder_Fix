import torch
import torch.nn as nn
import wandb

from tqdm import tqdm
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu

from dataset import (
    train_data,
    valid_data,
    collate_fn,
    src_vocab,
    tgt_vocab,
    PAD_IDX,
    EOS_IDX
)

from model import Transformer
from lr_scheduler import NoamScheduler

from config import (
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS,
    D_FF,
    DROPOUT,
    BATCH_SIZE,
    EPOCHS,
    WARMUP_STEPS,
    LABEL_SMOOTHING,
    GRAD_CLIP,
    BEST_MODEL_PATH,
    WANDB_PROJECT,
    WANDB_RUN_NAME
)

wandb.init(
    project=WANDB_PROJECT,
    name=WANDB_RUN_NAME
)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

valid_loader = DataLoader(
    valid_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
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

criterion = nn.CrossEntropyLoss(
    ignore_index=PAD_IDX,
    label_smoothing=LABEL_SMOOTHING
)

optimizer = torch.optim.Adam(
    model.parameters(),
    betas=(0.9, 0.98),
    eps=1e-9
)

scheduler = NoamScheduler(
    optimizer,
    d_model=D_MODEL,
    warmup_steps=WARMUP_STEPS
)

best_bleu = 0

@torch.no_grad()
def evaluate():

    model.eval()

    total_loss = 0

    references = []
    hypotheses = []

    for batch in valid_loader:

        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        output = model(
            src,
            tgt_input
        )

        loss = criterion(
            output.reshape(-1, output.shape[-1]),
            tgt_output.reshape(-1)
        )

        total_loss += loss.item()

        for i in range(src.shape[0]):

            prediction = model.infer(
                src[i].unsqueeze(0)
            )

            pred_words = []

            for token in prediction:

                if token == EOS_IDX:
                    break

                if token > 3:

                    pred_words.append(
                        tgt_vocab.lookup_token(token)
                    )

            target_words = []

            for token in tgt_output[i]:

                token = token.item()

                if token == EOS_IDX:
                    break

                if token > 3:

                    target_words.append(
                        tgt_vocab.lookup_token(token)
                    )

            hypotheses.append(pred_words)
            references.append([target_words])

    bleu = corpus_bleu(
        references,
        hypotheses
    ) * 100

    return (
        total_loss / len(valid_loader),
        bleu
    )

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}"
    )

    for batch in progress_bar:

        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()

        output = model(
            src,
            tgt_input
        )

        loss = criterion(
            output.reshape(-1, output.shape[-1]),
            tgt_output.reshape(-1)
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        progress_bar.set_postfix(
            loss=loss.item()
        )

    val_loss, bleu = evaluate()

    train_loss = total_loss / len(train_loader)

    wandb.log({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "bleu": bleu
    })

    print(
        f"Epoch {epoch+1} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"BLEU: {bleu:.2f}"
    )

    if bleu > best_bleu:

        best_bleu = bleu

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print("Best model saved")

print("Training complete")