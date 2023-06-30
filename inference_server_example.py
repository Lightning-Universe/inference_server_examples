from typing import List


class BaseServer:
    def setup(self):
        # setup your model pipeline here, this is called once
        self._model = lambda x: [2*y for y in x]

    def predict(self, batch: List[str]) -> List[str]:
        # make predictions using the request
        results = self._model(batch)
        return results


# run server locally
if __name__ == "__main__":
    s = BaseServer()
    s.setup()
    print(s.predict(["abc"]))
