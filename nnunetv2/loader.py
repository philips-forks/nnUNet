import os
import torch
import torch.nn as nn
from pathlib import Path
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from collections import OrderedDict
from contextlib import contextmanager


class nnUNet(nn.Module):
    def __init__(
        self,
        nnunet_trained_model_dir: str = "/home/nnUNet_results/Dataset001_Task",
        fold: int = 0,
        checkpoint_name: str = "checkpoint_best.pth",
    ):
        super().__init__()
        nnunet_trained_model_dir = Path(nnunet_trained_model_dir)
        model_dir = nnunet_trained_model_dir / "nnUNetTrainer__nnUNetPlans__2d"
        _compile = os.getenv("nnUNet_compile", '0')
        os.environ["nnUNet_compile"] = '0'
        self.predictor = nnUNetPredictor(device=torch.device("cpu"), perform_everything_on_gpu=False)
        self.predictor.initialize_from_trained_model_folder(model_dir, (fold,), checkpoint_name=checkpoint_name)
        self.predictor.network.load_state_dict(self.predictor.list_of_parameters[0])
        self.model = self.predictor.network
        self.name = f"{nnunet_trained_model_dir.stem}_fold{fold}_{Path(checkpoint_name).stem}"
        os.environ["nnUNet_compile"] = _compile

    def forward(self, x: torch.Tensor):
        return self.model(x)

    @contextmanager
    def predict_state(self):
        initial_ds = self.model.decoder.deep_supervision
        training_state = self.model.training
        try:
            self.model.train(mode=False)
            self.model.decoder.deep_supervision = False
            yield self
        finally:
            self.model.decoder.deep_supervision = initial_ds
            self.model.train(mode=training_state)
    
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
        savedir: Path = ".",
        sample_input: torch.Tensor = torch.rand(1, 1, 512, 512)
    ):
        dynamic_axes = {'image' : {0: 'batch_size'}, 'mask' : {0: 'batch_size'}}
        with self.predict_state():
            traced_model = torch.jit.trace(self.model, sample_input)
            torch.onnx.export(
                traced_model.cpu(),
                sample_input.cpu(),
                str(Path(savedir) / f"{self.name}.onnx"),
                input_names=["image"],
                output_names=["mask"],
                dynamic_axes=dynamic_axes,
            )

# Example of preparing the network for finetuning pre-trained nnUNet
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
        nnunet_trained_model_dir: str = "/home/nnUNet_results/Dataset001_Task",
        nnunet_fold: int = 0,
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.image_size = image_size
        self.deep_supervision = deep_supervision

        fe_layers = nn.ModuleDict()

        nnunet = nnUNet(nnunet_trained_model_dir, fold=nnunet_fold)
        for i, block in enumerate(nnunet.model.encoder.stages[:-1]):
            fe_layers[f"stage_{i}"] = block

        self.bottleneck = nnunet.model.encoder.stages[-1]

        self.fe_layers = nn.Sequential(OrderedDict(**fe_layers))
        self.fe_keys = list(fe_layers.keys())
        self.fe_hook = FeatureHook(self.fe_layers, self.fe_keys)

        # Localization and Upscale layers
        self.upscale_layers = nn.ModuleDict()
        self.loc_layers = nn.ModuleDict()
        self.seg_layers = nn.ModuleDict()
        for i, key in enumerate(self.fe_keys[::-1]):
            self.upscale_layers[key] = nnunet.model.decoder.transpconvs[i]
            self.loc_layers[key] = nnunet.model.decoder.stages[i]
            self.seg_layers[key] = nnunet.model.decoder.seg_layers[i]

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

    # Example of export trained model to ONNX.
    nnunet = nnUNet("nnUNet_results/Dataset001_Task", fold='all')
    nnunet.export_onnx("nnUNet_results/Dataset001_Task")

    from nnunetv2.imageio.nibabel_reader_writer import NibabelIO

    # Raw Inference example
    dev = torch.device("cpu")
    model = UNet(nnunet_trained_model_dir=os.path.join(os.environ["nnUNet_results"], "Dataset001_Task"), nnunet_fold=7)
    # img, props = NibabelIO().read_images(
    #     [os.path.join(os.environ["nnUNet_raw"], "Dataset001_Task/imagesTr/0000.nii.gz")]
    # )
    # input = torch.tensor(img).to(dev)
    input = torch.rand(1, 1, 512, 512).to(dev)
    model = model.to(dev)
    out = model(input)
    print(out.shape)

    # nnUNet predictor usage example (not always better, despite all your assumptions  ¯\_(ツ)_/¯ )
    import os
    import matplotlib.pyplot as plt
    preprocessed_dataset_folder_base = (
        os.path.join(os.environ["nnUNet_results"], "Dataset001_Task/nnUNetTrainer__nnUNetPlans__2d")
    )
    predictor = nnUNetPredictor()
    predictor.initialize_from_trained_model_folder(
        preprocessed_dataset_folder_base, (6, 7), checkpoint_name="checkpoint_best.pth"
    )
    predictor.network.load_state_dict(predictor.list_of_parameters[0])
    img, props = NibabelIO().read_images(
        [os.path.join(os.environ["nnUNet_raw"], "Dataset001_Task/imagesTr/0000.nii.gz")]
    )
    ret = predictor.predict_single_npy_array(img, props)

    plt.figure(dpi=240)
    plt.imshow(img[0, 0], cmap="gray")
    plt.imshow(ret[0], alpha=0.2)
    plt.savefig("imshow.png")
