import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from contextlib import contextmanager

os.environ["NNUNET_LOADER_USED"] = "True"
from nnunet.training.model_restore import restore_model


class nnUNet(nn.Module):
    def __init__(self, pretrained_dir: str, ckp_name: str = "model_best.model") -> None:
        """Loader for pre-trained nnunet

        Parameters
        ----------
        fold_dir : str
            Path to `nnUNetTrainerV2__nnUNetPlansv2.1/fold_i` dir
        """
        super().__init__()
        self.model = self._load_best_model_for_inference(pretrained_dir, ckp_name)

    def _load_best_model_for_inference(self, fold_dir, ckp_name):
        checkpoint = os.path.join(fold_dir, ckp_name)
        pkl_file = checkpoint + ".pkl"
        trainer = restore_model(pkl_file, checkpoint, False)
        return trainer.network

    @contextmanager
    def predict_state(self):
        try:
            self.eval()
            self.model.do_ds = False
            yield self
        finally:
            self.model.do_ds = True
            self.train()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def predict(self, x: np.ndarray):
        """Predicts one-hot mask for trained model

        Parameters
        ----------
        x : np.ndarray
            Shape: b c *

        Returns
        -------
        np.ndarray
            One-hot prediction of shape b c *
        """

        param = next(self.parameters()).data
        x = torch.tensor(x, dtype=param.dtype, device=param.device)

        with self.predict_state():
            with torch.no_grad():
                preds = self.model(x)

        preds = F.one_hot(preds.max(1).indices, preds.shape[1]).to(torch.uint8)
        preds = preds.transpose(-2, -1).transpose(-3, -2)  # "* h w c" -> "* c h w"
        if preds.ndim == 5:
            preds = preds.transpose(-4, -3)  # "* d c h w" -> "* c d h w"

        return preds.cpu().numpy()

    def export_torchscript(
        self,
        savepath: str = "nnunet_best.torchscript",
        sample_input: torch.Tensor = torch.rand(1, 1, 512, 512)
    ):
        with self.predict_state():
            traced_model = torch.jit.trace(self.model, sample_input)
            traced_model.save(savepath)

    def export_onnx(
        self,
        savepath: str = "nnunet_best.onnx",
        sample_input: torch.Tensor = torch.rand(1, 1, 512, 512)
    ):
        dynamic_axes = {'image' : {0: 'batch_size'}, 'mask' : {0: 'batch_size'}}
        with self.predict_state():
            traced_model = torch.jit.trace(self.model, sample_input)
            torch.onnx.export(
                traced_model.cpu(),
                sample_input.cpu(),
                savepath,
                input_names=["image"],
                output_names=["mask"],
                dynamic_axes=dynamic_axes,
            )


class FeatureHook(nn.Module):
    def __init__(self, module: nn.Module, layers: list[str]):
        super().__init__()
        self.module = module
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = self.module._modules[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str):
        def fn(module, input, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.module(x)
        return x, self._features


class UNet(nn.Module):
    def __init__(
        self,
        num_channels: int = 1,
        image_size: int = 512,
        pretrained: str = "/ws/fold_0",
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.image_size = image_size
        self.deep_supervision = deep_supervision

        fe_layers = nn.ModuleDict()

        nnunet = nnUNet(pretrained)
        for i, block in enumerate(nnunet.model.conv_blocks_context[:-1]):
            fe_layers[f"stack_{i}"] = block
            fe_layers[f"stack_{i}"].out_features = block.output_channels

        self.bottleneck = nnunet.model.conv_blocks_context[-1]

        self.fe_layers = nn.Sequential(OrderedDict(**fe_layers))
        self.fe_keys = list(fe_layers.keys())
        self.fe_hook = FeatureHook(self.fe_layers, self.fe_keys)

        # Localization and Upscale layers
        self.upscale_layers = nn.ModuleDict()
        self.loc_layers = nn.ModuleDict()
        self.seg_layers = nn.ModuleDict()
        for i, key in enumerate(self.fe_keys[::-1]):
            self.upscale_layers[key] = nnunet.model.tu[i]
            self.loc_layers[key] = nnunet.model.conv_blocks_localization[i]
            self.seg_layers[key] = nnunet.model.seg_outputs[i]


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        segmentation_outputs = []
        b, c, h, w = x.size()
        assert c == self.num_channels
        assert h == w == self.image_size

        x, downscale_features = self.fe_hook(x)

        x = self.bottleneck(x)

        for key in self.fe_keys[::-1]:
            x = torch.cat((self.upscale_layers[key](x), downscale_features[key]), dim=1)
            x = self.loc_layers[key](x)
            segmentation = self.seg_layers[key](x)
            segmentation_outputs.append(segmentation)

        if self.deep_supervision:
            return segmentation_outputs
        else:
            return segmentation_outputs[-1]


if __name__ == "__main__":

    dev = torch.device("cpu")
    model = UNet(pretrained="...Task512_MtSinaiBinBkg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0")
    input = torch.rand(4, 1, 512, 512).to(dev)
    model = model.to(dev)
    out = model(input)
    print(out.shape)
# "/Users/artem/pyproj/ws/tcbs/backbones/nnunet/Task510_BBOXSem/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0"

