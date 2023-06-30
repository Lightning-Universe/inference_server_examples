# This is an example of inference server that classifies images according to ImageNet classes
# using pre-trained ResNet18 model from torchvision collection.
# Please refer https://pytorch.org/hub/pytorch_vision_resnet/ to see the model description on PyTorch website.
import base64, io
import requests
from typing import List
from urllib.request import urlopen

import torch
from torchvision import transforms, models
from PIL import Image


class BaseServer:
    """A class BaseServer contains two methods needed for inference server.

    Methods
    -------
    setup():
        Sets up your model.
    predict():
        Makes predictions.
    """
    def setup(self):
        """Setup is called once per inference server setup.
        Here we initialize the model and download ImageNet categories."""
        self._model = models.resnet18(pretrained=True)
        self._model.eval()
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

        url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
        output = urlopen(url).read()
        self._categories = [s.strip() for s in output.decode('utf-8').split('\n')]

    def predict(self, request: List[str]) -> List[str]:
        """In predict, we get encoded images, prepare them to get predictions from the model
        and getting their categories using the model we set up.

        Parameters
        ----------
        request : List[str]
            List of encoded images to detect their categories

        Returns
        -------
        List[str]
            List of detected categories
        """
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
            input_batch = input_tensor.unsqueeze(0)

            input_batch = input_batch.to(input_batch)
            prediction = self._model(input_batch)

            results.append(self._categories[prediction.argmax().item()])

        return results


# run server locally
if __name__ == "__main__":
    s = BaseServer()
    s.setup()

    img_url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    img_content = requests.get(img_url).content
    img = base64.b64encode(img_content).decode("UTF-8")

    print(s.predict([img]))
