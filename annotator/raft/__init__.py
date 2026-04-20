import os
import torch
import torchvision
from PIL import Image
from torchvision.utils import flow_to_image
from annotator.annotator_path import models_path, DEVICE

from torchvision.models.optical_flow import raft_large, Raft_Large_Weights



class RaftDetector:
    model_dir = os.path.join(models_path, "raft")

    def __init__(self):
        self.model = None
        self.device = DEVICE
        self.weights = Raft_Large_Weights.DEFAULT

    def load_model(self):
        model = raft_large(weights=self.weights, progress=False)
        model.eval()
        self.model = model.to(self.device)

    def unload_model(self):
        if self.model is not None:
            self.model.cpu()

    def __call__(self, input_image, prev_image):
        if self.model is None:
            self.load_model()
        self.model.to(self.device)

        input_image = Image.fromarray(input_image)
        trans2tens = torchvision.transforms.ToTensor()
        input_tensor, prev_tensor = trans2tens(input_image), trans2tens(prev_image)
        img1_batch, img2_batch = torch.stack([input_tensor]), torch.stack([prev_tensor])

        trans2flow = self.weights.transforms()
        img1_batch, img2_batch = trans2flow(img1_batch, img2_batch)

        list_of_flows = self.model(img1_batch.to(self.device), img2_batch.to(self.device))
        predicted_flow = list_of_flows[-1][0]

        # visualize the resulted flow
        flow_imgs = flow_to_image(list_of_flows[-1])
        img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]
        grid = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch, flow_imgs)]
        pil_image = torchvision.transforms.ToPILImage()(grid[0][1])
        pil_image.save('results/flow_image.jpg')

        return predicted_flow
