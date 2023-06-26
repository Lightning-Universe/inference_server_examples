# Inference server file should contain the single class named BaseServer,
# with two methods defined:
#   - setup(): sets up your model;
#   - predict(): makes predictions.
class BaseServer:
    # Setup step happens once per inference server setup.
    # All setup things like model load, pipeline setup, etc. should happen here:
    def setup(self):
        class DummyModel:
            def predict(self, input: str) -> str:
                return f"Prediction for: {input}"
        self._model = DummyModel()

    # Prediction step makes requests processing and prediction.
    # It should take list of requests and return list of prediction results.
    # Call your model/pipeline that your set up in setup() to get the predictions.
    def predict(self, request: List[str]) -> List[str]:
        results = []
        for request in request:
            result = self._model.predict(request)
            results.append(result)
        return results
