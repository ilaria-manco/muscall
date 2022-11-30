import os
import torchaudio
from pathlib import Path
from typing import Tuple, Optional, Union

import torch
from torch import Tensor
from torchaudio.datasets.gtzan import GTZAN, gtzan_genres

from muscall.utils.audio_utils import resample


class GTZAN(GTZAN):
    def __init__(
        self,
        root: Union[str, Path],
        download: bool = False,
        subset: Optional[str] = "training",
    ) -> None:
        super().__init__(root, download=download, subset=subset)

    def load_gtzan_item(
        self, fileid: str, path: str, ext_audio: str
    ) -> Tuple[Tensor, str]:
        """
        Loads a file from the dataset and returns the raw waveform
        as a Torch Tensor, its sample rate as an integer, and its
        genre as a string.
        """
        # Filenames are of the form label.id, e.g. blues.00078
        label, _ = fileid.split(".")

        # Read wav
        path_to_audio = os.path.join(path, label, fileid + ext_audio)

        waveform, sample_rate = torchaudio.load(path_to_audio)
        waveform = torch.mean(waveform, dim=0)
        if sample_rate != 16000:
            waveform = resample(waveform, sample_rate)

        return waveform, sample_rate, label

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, label)``
        """
        fileid = self._walker[n]
        item = self.load_gtzan_item(fileid, self._path, self._ext_audio)
        waveform, _, label = item
        label = gtzan_genres.index(label)
        length = 20 * 16000
        start = int((waveform.size(0) - length) / 2.0)
        return waveform[start : start + length], label

    @classmethod
    def num_classes(cls):
        return 10
