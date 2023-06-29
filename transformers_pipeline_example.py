# This is an example of inference server that makes texts summarization
# using summarization pipeline from Huggingface transformers.
from typing import List
from transformers import pipeline


class BaseServer:
    """Class BaseServer should contain two methods."""
    def setup(self):
        """In setup step, we inilialize summarization pipeline.
        """
        self.pipeline = pipeline("summarization")

    def predict(self, request: List[str]) -> List[str]:
        """In predict step, we are getting summaries for texts in requests.
        """
        result = self.pipeline(*request)
        return [r["summary_text"] for r in result]


# You can try your server running it locally:
if __name__ == "__main__":
    s = BaseServer()
    s.setup()
    print(s.predict(["summarize this text please"]))
