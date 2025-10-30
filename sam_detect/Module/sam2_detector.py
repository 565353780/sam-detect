import os
import torch
import numpy as np
from PIL import Image
from typing import Union
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from sam_detect.Config.model_config import MODEL_CONFIG_DICT


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class SAM2Detector(object):
    def __init__(
        self,
        model_file_path: Union[str, None] = None,
        device: str = "cpu",
    ) -> None:
        self.device = device

        self.updateDevice(self.device)
        print("[INFO][SAM2Detector::__init__]")
        print("\t using device:", self.device)

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def updateDevice(self, device: str) -> bool:
        if device == "cpu":
            self.device = "cpu"
            return True

        if device == "mps":
            if torch.backends.mps.is_available():
                self.device = "mps"
                print("[WARN][SAM2Detector::updateDevice]")
                print(
                    "\t Support for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                    "give numerically different outputs and sometimes degraded performance on MPS. "
                    "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
                )
                return True

            print("[WARN][SAM2Detector::updateDevice]")
            print("\t mps not available!")
            self.device = "cpu"
            return True

        if "cuda" in device:
            if torch.cuda.is_available():
                self.device = device
                # use bfloat16 for the entire notebook
                torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
                # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
                if torch.cuda.get_device_properties(0).major >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                return True

            print("[WARN][SAM2Detector::updateDevice]")
            print("\t cuda not available!")
            self.device = "cpu"
            return True

        print("[WARN][SAM2Detector::updateDevice]")
        print("\t device not valid!")
        self.device = "cpu"
        return True

    def loadModel(self, model_file_path: str) -> bool:
        if not os.path.exists(model_file_path):
            print("[ERROR][SAM2Detector]")
            print("\t model file not exist!")
            print("\t model_file_path:", model_file_path)
            return False

        model_cfg = None
        for model_type, config_file_path in MODEL_CONFIG_DICT.items():
            if model_type not in model_file_path:
                continue

            model_cfg = config_file_path
            break

        if model_cfg is None:
            print("[ERROR][SAM2Detector::loadModel]")
            print("\t model config not found!")
            print("\t model_file_path:", model_file_path)
            return False

        sam2 = build_sam2(
            model_cfg, model_file_path, device=self.device, apply_postprocessing=False
        )

        self.mask_generator = SAM2AutomaticMaskGenerator(sam2)
        return True

    def detect(self, image: np.ndarray) -> torch.Tensor:
        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        plt.axis("off")
        plt.show()
        return

    def detectImageFile(self, image_file_path: str) -> torch.Tensor:
        image = Image.open(image_file_path)
        image = np.array(image.convert("RGB"))
        return self.detect(image)
