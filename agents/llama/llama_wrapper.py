from transformers import pipeline, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import GenerationConfig
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
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)



    def __call__(self,prompt,stop_words,max_new_tokens=128,temperature=0.2):
        stop_token_ids=[self.tokenizer.encode(stop_word)[-1] for stop_word in stop_words]
        # print("stop_token_ids:",stop_token_ids)
        # conover ids to tokens
        # stop_tokens = [self.generator.tokenizer.convert_ids_to_tokens(stop_token_id) for stop_token_id in stop_token_ids]
        # print("stop tokens:",stop_tokens)
        stopping_criteria = StoppingCriteriaList([KeywordsStoppingCriteria(stop_token_ids)])

        eos_token_id = self.tokenizer.eos_token_id
        generation_config = GenerationConfig(
            eos_token_id=eos_token_id,
            pad_token_id=eos_token_id,
            do_sample=True,
            max_new_tokens=128,
            temperature=0.2
        )

        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        input_ids = input_ids.to('cuda')
        generated_ids = self.model.generate(input_ids,
                                            generation_config=generation_config,
                                            stopping_criteria=stopping_criteria)
        response = self.tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:-1], skip_special_tokens=True)[0]

        return response


if __name__ == '__main__':
    model_path = "/data4/codellama/CodeLlama-13b-Instruct-hf/"
    generator = LLamaAgent(model_path)

    prompt = "def sum(a,b) -> float:\n"
    stop_words = ["\ndef"]
    max_new_tokens = 128
    temperature = 0.2
    print("    User Prompt:\n", prompt)

    response = generator(prompt,stop_words,max_new_tokens,temperature)

    print("    Response:\n", response)



