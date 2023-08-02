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

root_path = '/Users/fangyijiewang/Desktop/PhD/MICCAI2023'

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
        self.batch_size = 16
        # todo Load the trained model
        if execute_in_docker:
            path_model = "/opt/algorithm/model/unet_frozen_timm-resnest14d_model.pt"
            #path_model = f"{root_path}/fetalaop_test/model/unet_frozen_timm-resnest14d_model.pt"
        else:
            path_model = "./model/unet_frozen_timm-resnest14d_model.pt"
        self.md = MySegmentationModel(path_model)
        load_success = self.md.load_model()
        if load_success:
            print("Successfully loaded model.")
            logging.info("Successfully loaded model.")

    def load_image(self, img_path) -> numpy.ndarray:
        img = sitk.ReadImage(img_path)
        nda = sitk.GetArrayFromImage(img)

        img_array = np.transpose(nda, (1, 2, 0))
        return img_array

    def imge_to_tensor(self, data):
        img_rgb = Image.fromarray(data, 'RGB')
        image = val_transforms(img_rgb)
        return image[None, :]

    def write_outputs(self, image_name, outputs):
        if not os.path.exists(f"/output/images/symphysis-segmentation"):
            os.makedirs(f"/output/images/symphysis-segmentation")
        sitk.WriteImage(outputs, f"/output/images/symphysis-segmentation" + image_name + '.mha')

    def predict(self, image_data):
        with torch.no_grad():
            # Put it into the network for processing
            pred = self.md.process_image(image_data)
            # Post-processing and saving of predicted images
            mask_g = pred[0][1, :, :].detach().cpu().numpy() * np.array([0])
            mask_pubic = pred[0][1, :, :].detach().cpu().numpy() * np.array([1])
            mask_head = pred[0][2, :, :].detach().cpu().numpy() * np.array([2])
            pred_img = sitk.GetImageFromArray(mask_g + mask_pubic + mask_head)

            return pred_img

    def process(self):
        image_paths = list(self.input_dir.glob("*"))

        for image_path in image_paths:
            image_name = os.path.basename(image_path).split('.')[0]
            image_data = self.load_image(image_path)
            image_tensor = self.imge_to_tensor(image_data)
            result = self.predict(image_tensor)
            self.write_outputs(image_name, result)
        print("Success hsiadhfjowiqjeoijfosdj9832049820sahfdi389u4903u409")


if __name__ == "__main__":
    NoduleSeg().process()
