import base64, io
from typing import List
from urllib.request import urlopen

import torch
from torchvision import transforms, models
from PIL import Image


class BaseServer:
    # All setup things are moved to here:
    def setup(self):
        self._model = models.resnet18(pretrained=True)
        self._model.eval()
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

        url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
        output = urlopen(url).read()
        self._categories = [s.strip() for s in output.decode('utf-8').split('\n')]

    # Prediction step just makes requests processing and prediction.
    def predict(self, request: List[str]) -> List[str]:
        results = []
        for image_request in request:
            image = base64.b64decode(image_request.encode("utf-8"))
            input_image = Image.open(io.BytesIO(image))
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

            input_batch = input_batch.to(input_batch)
            prediction = self._model(input_batch)
            probabilities = torch.nn.functional.softmax(prediction[0], dim=0)

            results.append(self._categories[prediction.argmax().item()])

        return results
