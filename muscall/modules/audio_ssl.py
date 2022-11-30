from torch import nn

from simclr import SimCLR
from simclr.modules import NT_Xent

from muscall.utils.audio_utils import get_transform_chain


class SimCLRAudio(nn.Module):
    def __init__(self, encoder, audio_config):
        super().__init__()
        self.config = audio_config
        projection_dim = self.config.ssl.ssl_projection_dim
        temperature = self.config.ssl.ssl_temperature
        n_features = self.config.hidden_size

        self.simclr = SimCLR(
            encoder=encoder,
            projection_dim=projection_dim,
            n_features=n_features,
        )
        self.nt_xent_loss = NT_Xent(
            batch_size=256,
            temperature=temperature,
            world_size=1,
        )

        self.transform = get_transform_chain(
            p_polarity=self.config.ssl.p_polarity,
            p_noise=self.config.ssl.p_noise,
            p_gain=self.config.ssl.p_gain,
            p_highpass=self.config.ssl.p_filter,
            p_lowpass=self.config.ssl.p_filter,
            p_reverb=self.config.ssl.p_reverb,
            p_pitch_shift=self.config.ssl.p_pitch_shift,
            sample_rate=16000,
        )

    def forward(self, x, original_x):
        if original_x is None:
            x_i = self.transform(x.unsqueeze(1), 16000).squeeze(1)
            x_j = self.transform(x.unsqueeze(1), 16000).squeeze(1)
        else:
            x_i = x
            x_j = self.transform(original_x.unsqueeze(1), 16000).squeeze(1)

        _, _, z_i, z_j = self.simclr(x_i, x_j)
        loss = self.nt_xent_loss(z_i, z_j)

        return loss
