import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


if __name__ == "__main__":
    nnunet_pretrained = nnUNet("/path/to/nnUNet_trained_models/nnUNet/2d/TaskXXX_NAME/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0")
