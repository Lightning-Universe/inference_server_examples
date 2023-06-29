# Inference server file should contain the single class named BaseServer.
from typing import List


class BaseServer:
    """Class BaseServer should contain two methods."""
    def setup(self):
        """Setup happens once per inference server set up.
        Things like model load, pipeline initialization, etc. should happen here.
        """
        self._model = lambda x: 2*x

    def predict(self, request: List[str]) -> List[str]:
        """Predict makes requests processing and prediction.
        It should take a list of requests and return a list of prediction results.
        """
        results = []
        for request in request:
            results.append(self._model(request))
        return results


# You can try your server running it locally:
if __name__ == "__main__":
    s = BaseServer()
    s.setup()
    print(s.predict(["abc"]))
