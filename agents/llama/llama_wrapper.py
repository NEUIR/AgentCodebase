from transformers import pipeline, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
import torch

# Define the stopping criteria
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids:list):
        self.keywords = keywords_ids
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] in self.keywords:
            return True
        return False


class LLamaAgent():
    def __init__(self, model_path):
        """
        :param model: The model to use for completion.
        """
        self.model_path = model_path
        self.model = LlamaForCausalLM.from_pretrained(self.model_path, device_map="auto")
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        return self.generator


    def __call__(self,prompt,stop_words):
        stop_token_ids=[self.generator.tokenizer.encode(stop_word) for stop_word in stop_words]
        stopping_criteria = StoppingCriteriaList([KeywordsStoppingCriteria(stop_token_ids)])
        response = self.generator(prompt,
                                  num_return_sequences=1,
                                  return_full_text=False,
                                  handle_long_generation="hole",
                                  temperature=1.0,
                                  top_p=1.0,
                                  max_new_tokens=1024,
                                  do_sample=True,
                                  stopping_criteria=stopping_criteria)
        return response[0]["generated_text"].strip()


if __name__ == '__main__':
    model_path = ""
    generator = LLamaAgent(model_path)

    prompt = "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    "
    stop_words = ["def"]

    print("    User Prompt: ", prompt)

    response = generator(prompt,stop_words)

    print("    Response: ", response)



