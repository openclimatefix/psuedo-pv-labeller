import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin


class PsuedoIrradienceForecastor(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        input_channels: int = 3,
        input_size: int = 256,
        input_steps: int = 12,
        output_channels: int = 8,
        conv3d_channels: int = 256,
        hidden_dim: int = 8,
        kernel_size: int = 3,
        num_layers: int = 1,
        output_steps: int = 1,
        pv_meta_input_channels: int = 2,
        **kwargs
    ):
        super().__init__()
        config = locals()
        config.pop("self")
        config.pop("__class__")
        self.config = kwargs.pop("config", config)
        input_size = self.config.get("input_size", 256)
        input_steps = self.config.get("input_steps", 12)
        input_channels = self.config.get("input_channels", 3)
        output_channels = self.config.get("output_channels", 8)
        conv3d_channels = self.config.get("conv3d_channels", 256)
        kernel_size = self.config.get("kernel_size", 3)
        num_layers = self.config.get("num_layers", 1)
        output_steps = self.config.get("output_steps", 1)
        pv_meta_input_channels = self.config.get("pv_meta_input_channels", 2)
        hidden_dim = self.config.get("hidden_dim", 8)
        self.input_steps = input_steps
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Conv3d(
                in_channels=input_channels,
                out_channels=conv3d_channels,
                kernel_size=(kernel_size, kernel_size, kernel_size),
                padding="same",
            )
        )
        for i in range(0, num_layers):
            self.layers.append(
                nn.Conv3d(
                    in_channels=conv3d_channels,
                    out_channels=conv3d_channels,
                    kernel_size=(kernel_size, kernel_size, kernel_size),
                    padding="same",
                )
            )

        # Map to output latent variables, per timestep

        # Map the output to the latent variables
        self.latent_head = nn.Conv3d(
            conv3d_channels, out_channels=output_channels, kernel_size=(1, 1, 1), padding="same"
        )

        self.latent_output = nn.Conv2d(
            in_channels=input_steps * output_channels,
            out_channels=output_steps,
            kernel_size=(1, 1),
            padding="same",
        )


        # Small head model to convert from latent space to PV generation for training
        # Input is per-pixel input data, this will be reshaped to the same output steps as the latent head
        self.pv_meta_input = nn.Conv2d(
            pv_meta_input_channels, out_channels=hidden_dim, kernel_size=(1, 1)
        )

        # Output is forecast steps channels, each channel is a timestep
        # For labelling, this should be 1, forecasting the middle timestep, for forecasting, the number of steps
        # This is done by putting the meta inputs to each timestep
        self.pv_meta_output = nn.Conv2d(
            in_channels=input_steps * (output_channels + hidden_dim),
            out_channels=output_steps,
            kernel_size=(1, 1),
            padding="same",
        )

    def forward(self, x: torch.Tensor, pv_meta: torch.Tensor = None, output_latents: bool = True):
        for layer in self.layers:
            x = layer(x)
        x = self.latent_head(x)
        if output_latents:
            x = einops.rearrange(x, "b c t h w -> b (c t) h w")
            return self.latent_output(x)
        pv_meta = self.pv_meta_input(pv_meta)
        # Reshape to fit into 3DCNN
        pv_meta = einops.repeat(pv_meta, "b c h w -> b c t h w", t=self.input_steps)
        x = torch.cat([x, pv_meta], dim=1)
        # Get pv_meta_output
        x = einops.rearrange(x, "b c t h w -> b (c t) h w")
        x = F.relu(self.pv_meta_output(x))  # Generation can only be positive or 0, so ReLU
        return x
