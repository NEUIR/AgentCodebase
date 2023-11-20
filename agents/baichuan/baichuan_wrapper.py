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


class BaichuanAgent():
    def __init__(self, model_path):
        """
        :param model: The model to use for completion.
        """
        self.model_path = model_path
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,use_fast=False, trust_remote_code=True)



    def __call__(self,messages,stop_words,max_new_tokens=128,temperature=0.2):
        stop_token_ids=[self.tokenizer.encode(stop_word)[-1] for stop_word in stop_words]
        stopping_criteria = StoppingCriteriaList([KeywordsStoppingCriteria(stop_token_ids)])

        eos_token_id = self.tokenizer.eos_token_id
        generation_config = GenerationConfig(
            eos_token_id=eos_token_id,
            pad_token_id=eos_token_id,
            do_sample=True,
            max_new_tokens=128,
            temperature=0.2
        )

        # instruct model
        # input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        # input_ids = input_ids.to('cuda')
        # generated_ids = self.model.generate(input_ids,
        #                                     generation_config=generation_config,
        #                                     stopping_criteria=stopping_criteria)
        # response = self.tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:-1], skip_special_tokens=True)[0]


        # chat model
        response = self.model.chat(self.tokenizer, messages, generation_config=generation_config)
        return response


if __name__ == '__main__':
    model_path = "/data4/Baichuan2/Baichuan2-13B-Chat/"
    generator = BaichuanAgent(model_path)

    messages = []
    messages.append({"role": "user", "content": "解释一下“温故而知新”"})
    stop_words = ["\ndef"]
    max_new_tokens = 128
    temperature = 0.2

    print("    User Prompt:\n", messages)

    response = generator(messages,stop_words,max_new_tokens,temperature)

    print("    Response:\n", response)



