import json
import os
import random
from collections import Counter
from typing import List

import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BatchEncoding, XLMRobertaTokenizer

from loading_beit3models import (
    load_beit3_base_finetuned,
    load_beit3_large_finetuned
)


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

def get_img_names_and_questions(
    data_dir: str = "./VizWiz/VizWizAnnotations", split: str = "test"
):
    with open(os.path.join(data_dir, f"{split}.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    img_names = list()
    questions = list()
    ans = list()
    for d in data:
        img_names.append(d["image"])
        questions.append(d["question"])
        if split == "train" or split == "val":
            ans.append(d["answers"])
    if split == "train" or split == "val":
        return img_names, questions, ans
    return img_names, questions

def create_vizwiz_label2ans_ans2label(
    data_dir="./VizWiz/VizWizAnnotations",
):
    answers = []
    data_sets = ["train", "val"]
    for data_set in data_sets:
        with open(f"{data_dir}/{data_set}.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            for d in data:
                for a in d["answers"]:
                    answers.append(a["answer"])

    answers = Counter(answers)
    answers = answers.most_common(3129)
    label2ans = [a for (a, _) in answers]
    ans2label = {answer: i for i, answer in enumerate(label2ans)}
    return ans2label, label2ans


class VizWizDataset(Dataset):

    def __init__(
        self,
        tokenizer,
        img_transform,
        img_names: List[str] = None,
        questions: List[BatchEncoding] = None,
        viz_wiz_dir="./VizWiz",
        split="test",
        answers=None,
        num_max_bpe_tokens=64,
        mask_prob=0.15,
        use_huggingface=False,
    ):
        # self.ans2label, self.label2ans = create_vizwiz_label2ans_ans2label(
        #     data_dir="/teamspace/studios/this_studio/beit-distillation/VizWiz/VizWizAnnotations"
        # )
        self.ans2label, self.label2ans = create_vizwiz_label2ans_ans2label(os.path.join(viz_wiz_dir, "VizWizAnnotations"))
        valid_splits = ["train", "val", "test"]
        if split not in valid_splits:
            raise ValueError(
                f"{split} is not a valid split name. Split name must be in {valid_splits}."
            )
        self.split = split
        self.use_huggingface = use_huggingface
        if self.use_huggingface:
            self.dataset = load_dataset(
                f"Multimodal-Fatima/VizWiz_{split}", split=split
            )
        else:
            self.img_path = "VizWizTrainImages"
            if split == "val":
                self.img_path = "VizWizValImages"
            elif split == "test":
                self.img_path = "VizWizTestImages"

            self.viz_wiz_img_dir = os.path.join(viz_wiz_dir, self.img_path)
            self.img_names = img_names
            self.questions = questions
            if split == "train" or split == "val":
                if answers is None:
                    raise Exception("Answers for train/val split not provided.")
                self.labels = []
                self.scores = []
                for answer_list in answers:
                    answer_count = {}
                    for answer_dict in answer_list:
                        answer = answer_dict["answer"]
                        answer_count[answer] = answer_count.get(answer, 0) + 1

                    labels = []
                    scores = []
                    for answer in answer_count:
                        if answer not in self.ans2label:
                            continue
                        labels.append(self.ans2label[answer])
                        score = self.get_score(answer_count[answer])
                        scores.append(score)
                    self.labels.append(labels)
                    self.scores.append(scores)

        self.tokenizer = tokenizer
        self.img_transform = img_transform
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.mask_prob = mask_prob

        self.language_vocab_size = tokenizer.vocab_size
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id

    def get_score(self, occurences):
        if occurences == 0:
            return 0.0
        elif occurences == 1:
            return 0.3
        elif occurences == 2:
            return 0.6
        elif occurences == 3:
            return 0.9
        else:
            return 1.0

    def __getitem__(self, idx: int):
        """Returns the image and the question embedding at the given index"""
        data = dict()
        question = None

        if self.use_huggingface:
            question = self.dataset[idx]["question"]
            data["image"] = self.img_transform(self.dataset[idx]["image"])
            data["image_id"] = self.dataset[idx]["filename"]
            if self.split == "val" or self.split == "train":
                answer_count = {}
                for answer_ in self.dataset[idx]["answers"]:
                    answer_count[answer_] = answer_count.get(answer_, 0) + 1

                labels = []
                scores = []
                for answer in answer_count:
                    if answer not in self.ans2label:
                        continue
                    labels.append(self.ans2label[answer])
                    score = self.get_score(answer_count[answer])
                    scores.append(score)
                labels_tmp = [0.0] * len(self.label2ans)
                if len(labels) > 0:
                    for l, s in zip(labels, scores):
                        labels_tmp[l] = s
                labels_tmp = torch.FloatTensor(labels_tmp)
                data["labels"] = labels_tmp

        else:
            img_name = self.img_names[idx]
            img = os.path.join(self.viz_wiz_img_dir, img_name)
            img = Image.open(img)
            img = self.img_transform(img)
            question = self.questions[idx]
            data["image"] = img
            data["image_id"] = img_name
            if self.split != "test" and self.labels[idx] is not None:
                labels = [0.0] * len(self.label2ans)
                if len(self.labels[idx]) > 0:
                    for l, s in zip(self.labels[idx], self.scores[idx]):
                        labels[l] = s
                labels = torch.FloatTensor(labels)
                data["labels"] = labels

        language_tokens, padding_mask, num_tokens = self._get_text_segment(question)
        masked_tokens = language_tokens[:]
        masked_tokens, language_masked_pos = self._masking_on_text_tokens(
            masked_tokens, num_tokens, self.mask_prob
        )
        data["language_tokens"] = torch.IntTensor(language_tokens)
        data["masked_tokens"] = masked_tokens
        data["language_masked_pos"] = language_masked_pos
        data["padding_mask"] = padding_mask

        return data

    def _get_text_segment(self, text_segment, max_len=None):
        if isinstance(text_segment, str):
            tokens = self.tokenizer.tokenize(text_segment)
        else:
            tokens = text_segment[:]
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contain at least one token!")
        if max_len is None:
            max_len = self.num_max_bpe_tokens

        if len(tokens) > max_len - 2:
            tokens = tokens[: max_len - 2]

        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        return (
            tokens + [self.pad_token_id] * (max_len - num_tokens),
            padding_mask,
            num_tokens,
        )

    def _get_mask_token(self, token):
        p = random.random()
        if p < 0.8:
            return self.mask_token_id
        elif p < 0.9:
            return token
        else:
            return random.randint(3, self.language_vocab_size - 1)

    def _masking_on_text_tokens(self, tokens, num_tokens, mask_prob):
        bool_masked_pos = [0] * len(tokens)
        to_mask = min(int(num_tokens * mask_prob + 0.5), num_tokens - 1)
        to_mask = max(to_mask, 1)
        num_masked_tokens = 0
        while num_masked_tokens < to_mask:
            i = random.randint(1, num_tokens - 1)
            if bool_masked_pos[i] == 0:
                bool_masked_pos[i] = 1
                tokens[i] = self._get_mask_token(tokens[i])
                num_masked_tokens += 1

        return tokens, bool_masked_pos

    def __len__(self):
        return len(self.dataset) if self.use_huggingface else len(self.img_names)



def initModel(checkpoint_path: str, model_type: str, is_compiled_model_checkpoint=False):
    model = None
    if model_type not in ["base", "large"]:
        raise ValueError("model_type must be 'base' or 'large'")
    if model_type == "base":
        model = load_beit3_base_finetuned(checkpoint_path, is_compiled_model_checkpoint)
    elif model_type == "large":
        model = load_beit3_large_finetuned(checkpoint_path, is_compiled_model_checkpoint)
    if model is None:
        raise ValueError("Model not loaded correctly")
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model


if __name__ == "__main__":
    vizwiz_annotations_dir = "/teamspace/studios/this_studio/VizWiz/VizWizAnnotations"
    img_names, questions = get_img_names_and_questions(
        data_dir=vizwiz_annotations_dir,
        split="test",
    )
    embedding_model = "/teamspace/studios/this_studio/beit-distillation/models/beit3.spm"
    tokenizer = XLMRobertaTokenizer(embedding_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    num_max_bpe_tokens = 64
    batch_size = 64


    dataset = VizWizDataset(
        tokenizer=tokenizer,
        img_transform=getBeitVQAImgTransform(),
        split="test",
        img_names=img_names,
        questions=questions,
    )
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model_type = "large"
    checkpoint_epoch = 13
    checkpoint_path = f"/teamspace/studios/this_studio/beit-distillation/models/{model_type}/vizwiz_checkpoint_epoch{checkpoint_epoch}_{model_type}.tar"
    model = initModel(checkpoint_path, model_type, is_compiled_model_checkpoint=True)
    model = torch.compile(model)
    # Using TensorFloat32 Cores for better performance
    torch.set_float32_matmul_precision('high')

    eval_path = f"/teamspace/studios/this_studio/beit-distillation/Evaluation/{model_type}"
    # eval loop
    img_ids = []
    pred_classes = []
    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader):
            img = data["image"].to(device)
            qs = data["language_tokens"].to(device)
            logits = model(img, qs)
            predicted_class = logits.argmax(-1)
            img_ids += data["image_id"]
            pred_classes += predicted_class
    # save the results as a json
    with open(
        os.path.join(
            eval_path,
            f"Trained_{checkpoint_epoch}_Epochs_Beit3_VizWiz_results.json",
        ),
        "w",
    ) as file:
        json.dump(
            [
                {
                    "image": img_id,
                    "answer": dataset.label2ans[prd],
                }
                for img_id, prd in zip(img_ids, pred_classes)
            ],
            file,
            indent=4,
        )
