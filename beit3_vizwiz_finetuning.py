import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import XLMRobertaTokenizer

from Beit3_vizwiz import VizWizDataset, get_img_names_and_questions
from loading_beit3models import load_beit3_base_finetuned, load_beit3_large_finetuned, load_beit3_base, load_beit3_large


def getBeitVQAImgTransform():
    return transforms.Compose(
        [
            transforms.Resize((480, 480), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
            ),
        ]
    )


def freeze_until(model: nn.Module, param_name="head.0.weight"):
    """Freeze all parameters up to param_name. Freezes all parameters except for head, by default."""
    for name, param in model.named_parameters():
        if param_name in name:
            break
        param.requires_grad = False

def initOptimizerLoss(model, checkpoint_path: str = None, lr = 2e-5, opt_betas = (0.9, 0.98), weight_decay = 0.01):
    # Initialize optimizer
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    optim = torch.optim.AdamW(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=lr,
        betas=opt_betas,
        weight_decay=weight_decay,
    )
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        optim.load_state_dict(checkpoint["optimizer_state_dict"])
    return optim, criterion


def validate(
    tokenizer, criterion, model, val_loader, device='cpu'
):
    val_loss = 0.0
    for data in tqdm(val_loader):

        img = data["image"].to(device)
        q_tokens = data["language_tokens"].to(device)
        labels = data["labels"].to(device)

        logits = model(image=img, question=q_tokens)
        logits = logits.float()
        loss = criterion(input=logits, target=labels)
        val_loss += loss.item() * img.size(0)

    val_loss = val_loss / len(val_loader)
    print(f"Validation loss: {val_loss:.4f}")
    return val_loss


def getDataLoader(tokenizer, data_dir="./VizWiz/", batch_size=32, split="train"):
    if split not in ["train", "val", "test"]:
        raise ValueError(f"Split: {split} is invalid.")
    img_names, questions, ans = get_img_names_and_questions(
        data_dir=os.path.join(data_dir,"VizWizAnnotations"), split=split
    )
    dl = DataLoader(
        VizWizDataset(
            tokenizer=tokenizer,
            img_transform=getBeitVQAImgTransform(),
            img_names=img_names,
            questions=questions,
            answers=ans,
            split=split,
            viz_wiz_dir=data_dir,
            use_huggingface=False,
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    return dl


def initModel(checkpoint_path: str, model_type: str, is_compiled_model_checkpoint=False, is_pretrained=False):
    model = None
    if model_type not in ["base", "large"]:
        raise ValueError("model_type must be 'base' or 'large'")
    if model_type == "base":
        if is_pretrained:
            model = load_beit3_base_finetuned(checkpoint_path, is_compiled_model_checkpoint)
        else:
            model = load_beit3_base(checkpoint_path)
    elif model_type == "large":
        if is_pretrained:
            model = load_beit3_large_finetuned(checkpoint_path, is_compiled_model_checkpoint)
        else:
            model = load_beit3_large(checkpoint_path)
    if model is None:
        raise ValueError("Model not loaded correctly")
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device == torch.device("cuda"), "Make sure, cuda is available"
    embedding_model = "/teamspace/studios/this_studio/beit-distillation/models/beit3.spm"
    tokenizer = XLMRobertaTokenizer(embedding_model)

    epochs = 1
    # needs ≈ 20 GB VRAM
    batch_size = 32

    model_type = "large"
    epoch_of_checkpoint = 12
    checkpoint_path = f"/teamspace/studios/this_studio/beit-distillation/models/{model_type}/vizwiz_checkpoint_epoch{epoch_of_checkpoint}_{model_type}.tar"
    model = initModel(checkpoint_path, model_type, is_compiled_model_checkpoint=True)
    # train layers 20, 21, 22, 23 and head
    freeze_until(model, "beit3.encoder.layers.20")
    print("compiling model...")
    model = torch.compile(model)
    print("finished compiling")
    # Using TensorFloat32 Cores for better performance
    torch.set_float32_matmul_precision('high')

    # optim, criterion = initOptimizerLoss(model, checkpoint_path)
    optim, criterion = initOptimizerLoss(model)
    

    vizwiz_path = "/teamspace/studios/this_studio/VizWiz/"
    train_loader = getDataLoader(tokenizer=tokenizer, batch_size=batch_size, data_dir=vizwiz_path)
    val_loader = getDataLoader(tokenizer=tokenizer, batch_size=batch_size, data_dir=vizwiz_path, split='val')
    losses = []
    num_correct = 0

    save_checkpoint_folder = f"/teamspace/studios/this_studio/beit-distillation/models/{model_type}"
    print("starting training")
    for epoch in range(epochs):
        print(f"Starting epoch {epoch}")
        epoch_loss = 0.0

        model.train()
        for data in tqdm(train_loader):

            img = data["image"].to(device)
            q_tokens = data["language_tokens"].to(device)
            labels = data["labels"].to(device)

            optim.zero_grad()

            # Mixed precision training
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(image=img, question=q_tokens)
                logits = logits.float()
                loss = criterion(input=logits, target=labels)
                epoch_loss += loss.item() * img.size(0)

            loss.backward()
            optim.step()

        epoch_loss = epoch_loss / len(train_loader)
        losses.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            val_loss = validate(
                tokenizer=tokenizer,
                criterion=criterion,
                model=model,
                val_loader=val_loader
            )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "loss": epoch_loss,
                "val_loss": val_loss,
            },
            os.path.join(
                save_checkpoint_folder, f"vizwiz_checkpoint_epoch{epoch + 1 + epoch_of_checkpoint}_{model_type}.tar"
            ),
        )
        print(f"Epoch {epoch} loss: {epoch_loss}")
        epoch_loss = 0.0
