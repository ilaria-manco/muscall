import os
import numpy as np
from sklearn import metrics

import torch
from torch.utils.data import DataLoader
from torchaudio.datasets.gtzan import gtzan_genres
from transformers.models.clip.tokenization_clip import CLIPTokenizer


from muscall.tasks.retrieval import Retrieval
from muscall.datasets.tagging import MTTDataset
from muscall.datasets.gtzan import GTZAN


def prepare_labels(labels, prompt=None):
    """Convert class labels to tokenized text inputs that can be passed
    to a muscall model. Optionally wrap the text labels within prompts [ref].
    """
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    max_length = 77
    text_prompts = torch.zeros((len(labels), max_length), dtype=torch.long).cuda()

    for i, label in enumerate(labels):
        if prompt is None:
            text_to_tokenize = label
        else:
            text_to_tokenize = "A {} track".format(label)
        input_ids = tokenizer.encode(
            text_to_tokenize, max_length=max_length, truncation=True
        )
        while len(input_ids) < max_length:
            input_ids.append(0)
        text_prompts[i] = torch.tensor(input_ids, dtype=torch.long)

    return text_prompts


def get_metrics(predictions, ground_truth, dataset_name):
    results = {}
    
    if dataset_name == "mtt":
        results["ROC-AUC-macro"] = metrics.roc_auc_score(
            ground_truth, predictions, average="macro"
        )
        results["MAP-avg"] = np.mean(
            metrics.average_precision_score(ground_truth, predictions, average=None)
        )
    elif dataset_name == "gtzan":
        predictions = torch.argmax(predictions, dim=1)
        ground_truth = ground_truth[:, 0]
        results["accuracy"] = metrics.accuracy_score(ground_truth, predictions)

    return results


@torch.no_grad()
def compute_muscall_similarity_score(model, data_loader, text_prompts, device):
    dataset_size = data_loader.dataset.__len__()

    all_audio_features = torch.zeros(dataset_size, 512).to("cuda")
    ground_truth = torch.zeros(dataset_size, data_loader.dataset.num_classes()).to(
        "cuda"
    )

    all_text_features = model.encode_text(text_prompts, None)
    all_text_features = all_text_features / all_text_features.norm(dim=-1, keepdim=True)

    for i, batch in enumerate(data_loader):
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
        input_audio, labels = batch

        input_audio = input_audio.to(device=device)

        audio_features = model.encode_audio(input_audio)
        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)

        num_samples_in_batch = input_audio.size(0)

        all_audio_features[
            i * num_samples_in_batch : (i + 1) * num_samples_in_batch
        ] = audio_features

        ground_truth[i * num_samples_in_batch : (i + 1) * num_samples_in_batch] = labels

    logits_per_audio = all_audio_features @ all_text_features.t()

    return logits_per_audio, ground_truth


class Zeroshot(Retrieval):
    def __init__(self, pretrain_config, dataset_name):
        self.dataset_name = dataset_name
        super().__init__(pretrain_config)

    def load_dataset(self):
        data_root = os.path.join(
            self.muscall_config.env.data_root, "datasets", self.dataset_name
        )

        if self.dataset_name == "mtt":
            test_dataset = MTTDataset(data_root, subset="testing")
            self.tags = np.load(os.path.join(data_root, "tags.npy"))
        elif self.dataset_name == "gtzan":
            test_dataset = GTZAN(data_root, subset="testing")
            self.tags = gtzan_genres
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    def evaluate(self):
        text_prompts = prepare_labels(self.tags)
        score_matrix, ground_truth = compute_muscall_similarity_score(
            self.model, self.test_loader, text_prompts, self.device
        )

        metrics = get_metrics(score_matrix.cpu(), ground_truth.cpu(), self.dataset_name)
        print(metrics)
        return metrics
