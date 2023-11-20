# author: wanghanbin
# Time: 2023/11/20
# desc: The main function of the project

import argparse
from agents.gpt.gpt_wrapper import GPTAgent


def load_generator(model_type):
    if model_type in ["gpt-4", "gpt-3.5-turbo",]:
        generator = API_Caller(model_type)
    else:
        ckpt = model_path[model_type]

        if model_type == "starchat":
            generator = pipeline("text-generation", model=ckpt, tokenizer=ckpt, torch_dtype=torch.bfloat16, device_map="auto")
        else: # llama-series
            if model_type in ["mpt-30b-chat", "falcon-40b-instruct"]:
                generator = pipeline(model=ckpt, tokenizer=ckpt, device_map="auto", trust_remote_code=True)
            else:
                model = LlamaForCausalLM.from_pretrained(ckpt, device_map="auto")
                tokenizer = LlamaTokenizer.from_pretrained(ckpt)
                generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print("model loaded")
    return generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="gpt-3.5-turbo-0613")
    args = parser.parse_args()

    generator = load_generator(model_type)


if __name__ == '__main__':
    main()