import os
import time
import numpy as np

import torch

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from sentence_transformers import SentenceTransformer

from muscall.datasets.audiocaption import AudioCaptionDataset
from muscall.trainers.base_trainer import BaseTrainer
from muscall.models.muscall import MusCALL
from muscall.tasks.retrieval import run_retrieval
from muscall.utils.audio_utils import get_transform_chain


class MusCALLTrainer(BaseTrainer):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.bert_config = self.config.model_config.bert
        self.batch_size = self.config.training.dataloader.batch_size

        self.load()

        self.scaler = torch.cuda.amp.GradScaler()

        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.sbert_model.to(self.device)

    def load_dataset(self):
        self.logger.write("Loading dataset")
        dataset_name = self.config.dataset_config.dataset_name

        if dataset_name == "audiocaption":
            self.train_dataset = AudioCaptionDataset(self.config.dataset_config)
            self.val_dataset = AudioCaptionDataset(self.config.dataset_config, dataset_type="val")
        else:
            raise ValueError("{} dataset is not supported.".format(dataset_name))

        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            **self.config.training.dataloader,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            dataset=self.val_dataset,
            **self.config.training.dataloader,
            drop_last=True,
        )

        self.logger.write(
            "Number of training samples: {}".format(self.train_dataset.__len__())
        )

    def build_model(self):
        self.logger.write("Building model")
        model_name = self.config.model_config.model_name

        if model_name == "muscall":
            self.model = MusCALL(self.config.model_config)
        else:
            raise ValueError("{} model is not supported.".format(model_name))

        self.print_parameters()

        self.model.to(self.device)

    def build_optimizer(self):
        self.logger.write("Building optimizer")
        optimizer_config = self.config.training.optimizer
        self.optimizer = getattr(optim, optimizer_config.name, None)(
            self.model.parameters(), **optimizer_config.args
        )

        num_train_optimization_steps = (
            int(self.train_loader.dataset.__len__() / self.batch_size)
            * self.config.training.epochs
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=num_train_optimization_steps * 0.1
        )

    def get_retrieval_metrics(self):
        indices = torch.randperm(len(self.val_dataset))[:1000]
        random_val_subset = Subset(self.val_dataset, indices)
        val_subset_loader = DataLoader(
            random_val_subset,
            batch_size=self.batch_size,
        )
        retrieval_metrics = run_retrieval(
            model=self.model,
            data_loader=val_subset_loader,
            device=self.device,
        )
        return retrieval_metrics["R@10"].item()

    def get_sentence_similarities(self, data_loader, data_idx):
        raw_captions = [
            data_loader.dataset.get_raw_caption(idx.item()) for idx in data_idx
        ]
        sentence_embeddings = self.sbert_model.encode(
            raw_captions,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        return sentence_embeddings @ sentence_embeddings.t()

    def train(self):
        best_r10 = 0

        if os.path.exists(self.logger.checkpoint_path):
            self.logger.write(
                "Resumed training experiment with id {}".format(
                    self.logger.experiment_id
                )
            )
            self.load_ckp(self.logger.checkpoint_path)
        else:
            self.logger.write(
                "Started training experiment with id {}".format(
                    self.logger.experiment_id
                )
            )
            self.start_epoch = 0

        for epoch in range(self.start_epoch, self.config.training.epochs):
            epoch_start_time = time.time()

            train_loss = self.train_epoch(self.train_loader, is_training=True)
            val_loss = self.train_epoch_val(self.val_loader)

            track_retrieval_metrics = True
            if track_retrieval_metrics:
                r10 = self.get_retrieval_metrics()

            epoch_time = time.time() - epoch_start_time
            self.logger.update_training_log(
                epoch + 1,
                train_loss,
                val_loss,
                epoch_time,
                self.scheduler.get_last_lr()[0],
                r10,
            )

            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }

            is_best = r10 > best_r10
            if is_best:
                best_r10 = r10
            # save checkpoint in appropriate path (new or best)
            self.logger.save_checkpoint(state=checkpoint, is_best=is_best)

    def load_ckp(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.start_epoch = checkpoint["epoch"]

    def train_epoch(self, data_loader, is_training):
        running_loss = 0.0
        n_batches = 0

        if is_training:
            self.model.train()
        else:
            self.model.eval()

        for i, batch in enumerate(data_loader):
            batch = tuple(t.to(device=self.device, non_blocking=True) for t in batch)
            audio_id, input_audio, text_input_ids, _, _, data_idx = batch

            if self.config.model_config.loss == "weighted_clip":
                sentence_sim = self.get_sentence_similarities(data_loader, data_idx)
            else:
                sentence_sim = None

            original_audio = None
            audio_data_config = self.config.dataset_config.audio
            if is_training and audio_data_config.augment:
                original_audio = input_audio
                augment_chain = get_transform_chain(
                    p_polarity=0,
                    p_gain=0,
                    p_noise=audio_data_config.p_noise,
                    p_pitch_shift=audio_data_config.p_pitch_shift,
                    sample_rate=audio_data_config.sr,
                )
                input_audio = augment_chain(input_audio.unsqueeze(1), audio_data_config.sr).squeeze(1)

            # Cast operations to mixed precision
            with torch.cuda.amp.autocast(enabled=self.config.training.amp):
                loss = self.model(
                    input_audio,
                    text_input_ids,
                    original_audio=original_audio,
                    sentence_sim=sentence_sim,
                )

            if is_training:
                if self.config.training.amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                # clamp temperature scaling if over log(100)
                if self.model.logit_scale.item() > np.log(100):
                    self.model.logit_scale.data = torch.clamp(
                        self.model.logit_scale.data, max=np.log(100)
                    )

                self.scheduler.step()
                self.optimizer.zero_grad()

            running_loss += loss.item()
            n_batches += 1

        return running_loss / n_batches

    def train_epoch_val(self, data_loader):
        with torch.no_grad():
            loss = self.train_epoch(data_loader, is_training=False)
        return loss
