import logging
import torch
import segmentation_models_pytorch as smp


class MySegmentationModel:
    def __init__(self, path_checkpoint):
        # network parameters
        self.model = None
        self.path_checkpoint = path_checkpoint
        self.device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

    # load saved models
    def load_model(self):
        if torch.cuda.is_available():
            print("Model loaded on CUDA")
            self.model = torch.load(self.path_checkpoint)
        else:
            print("Model loaded on CPU")
            self.model = torch.load(self.path_checkpoint, map_location='cpu')

        self.model.to(self.device)
        logging.info("Model loaded.")
        return True

    def process_image(self, input_data):
        device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
        self.model.eval()
        # todo Image preprocessing
        image = input_data.to(device=device, dtype=torch.float32)

        # Putting images into the network
        output = self.model(image)
        output = (output > 0.5).float()

        # todo  Post processing of predicted images

        return output
