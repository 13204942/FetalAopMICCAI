import os
from pathlib import Path
import logging

import numpy
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
import SimpleITK as sitk
from segmentation import MySegmentationModel
import segmentation_models_pytorch as smp

execute_in_docker = True

#root_path = '/Users/fangyijiewang/Desktop/PhD/MICCAI2023'

val_transforms = T.Compose(
    [
        T.Resize(256),
        T.ToTensor(),
        T.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
        ),
     ],
)


class NoduleSeg:
    def __init__(self):
        self.input_dir = Path(f"/input/images/pelvic-2d-ultrasound/") if execute_in_docker else Path("./test/")
        self.output_dir = Path(f"/output/images/symphysis-segmentation/") if execute_in_docker else Path("./output/")
        self.device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
        #self.device = torch.device('mps' if not torch.backends.mps.is_available() else 'cuda')
        self.batch_size = 10
        self.model_name = "unet_unfrozen_resnet18_model.pt"
        # todo Load the trained model
        if execute_in_docker:
            path_model = f"/opt/algorithm/model/{self.model_name}"
            #path_model = f"{root_path}/fetalaop_test/model/unet_frozen_timm-resnest14d_model.pt"
        else:
            path_model = f"./model/{self.model_name}"
        self.md = MySegmentationModel(path_model)
        load_success = self.md.load_model()
        if load_success:
            print(f"Successfully loaded model {path_model}")
        else:
            print(f"Failed to load model {path_model}")

    def load_image(self, img_path) -> numpy.ndarray:
        img = sitk.ReadImage(img_path)
        nda = sitk.GetArrayFromImage(img)
        img_array = np.transpose(nda, (1, 2, 0))
        return img_array

    def img_to_tensor(self, data):
        out = None
        try:
            img_rgb = Image.fromarray(data, 'RGB')
            image = val_transforms(img_rgb)
            out = image[None, :]
        except Exception as e:
            print(f"{e} - [img_to_tensor] failed to convert image to tensor object")
        return out

    def write_outputs(self, image_name, outputs):
        try:
            if not os.path.exists(f"/output/images/symphysis-segmentation"):
                os.makedirs(f"/output/images/symphysis-segmentation")
            trg_path = f"/output/images/symphysis-segmentation/" + image_name + '.mha'
            sitk.WriteImage(outputs, trg_path)
            print(f"[write_outputs] image was saved to the target path {trg_path}")
        except Exception as e:
            print(f"{e} - [write_outputs] failed to write image to target path")

    def predict(self, image_data):
        pred_img = None
        try:
            with torch.no_grad():
                # Put it into the network for processing
                pred = self.md.process_image(image_data)
                # Post-processing and saving of predicted images
                mask_g = pred[0][1, :, :].detach().cpu().numpy() * np.array([0])
                mask_pubic = pred[0][1, :, :].detach().cpu().numpy() * np.array([1])
                mask_head = pred[0][2, :, :].detach().cpu().numpy() * np.array([2])
                pred_img = sitk.GetImageFromArray(np.uint8(mask_g + mask_pubic + mask_head))
        except Exception as e:
            print(f"[predict] cannot do prediction with {e}")

        return pred_img

    def process(self):
        try:
            image_paths = list(self.input_dir.glob("*"))
            print(f"[process] Received {len(image_paths)} input images")
            for image_path in image_paths:
                image_name = os.path.basename(image_path).split('.')[0]
                print(image_path)
                image_data = self.load_image(image_path)
                print(f"[process] loaded {image_path}")
                image_tensor = self.img_to_tensor(image_data)
                print(f"[process] convert image to {image_tensor.shape} tensor")
                result = self.predict(image_tensor)
                print(f"[process] get prediction result: {result.GetSize()}")
                self.write_outputs(image_name, result)
            print("[process] Success hsiadhfjowiqjeoijfosdj9832049820sahfdi389u4903u409")
        except Exception as e:
            print(f"[process] program failed {e}")

if __name__ == "__main__":
    NoduleSeg().process()
