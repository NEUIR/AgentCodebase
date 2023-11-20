# author: wanghanbin
# Time: 2023/11/20
# desc: The main function of the project

import argparse
from agents.gpt.gpt_wrapper import GPTAgent
from agents.llama.llama_wrapper import LLamaAgent

# Model Description:https://platform.openai.com/docs/models/gpt-3-5
GPT_FAMILY=[
    "gpt-3.5-turbo-1106","gpt-3.5-turbo",
    "gpt-3.5-turbo-16k","gpt-3.5-turbo-instruct",
    "gpt-3.5-turbo-0613","gpt-3.5-turbo-16k-0613",
    "gpt-3.5-turbo-0301","text-davinci-003",
    "text-davinci-002","code-davinci-002",
    "gpt-4-1106-preview","gpt-4-vision-preview",
    "gpt-4","gpt-4-32k",
    "gpt-4-0613","gpt-4-32k-0613",
    "gpt-4-0314","gpt-4-32k-0314",
]

LLAMA_FAMILY=[
    # Llama-2
    "Llama-2-7b","Llama-2-7b-hf",
    "Llama-2-7-chatb","Llama-2-7b-chat-hf",
    "Llama-2-13b","Llama-2-13b-hf",
    "Llama-2-13b-chat","Llama-2-13b-chat-hf",
    "Llama-2-70b", "Llama-2-70b-hf",
    "Llama-2-70b-chat", "Llama-2-70b-chat-hf",
    # CodeLlama
    "CodeLlama-7b-hf","CodeLlama-7b-Python-hf","CodeLlama-7b-Instruct-hf"
    "CodeLlama-13b-hf","CodeLlama-13b-Python-hf","CodeLlama-13b-Instruct-hf"
    "CodeLlama-34b-hf","CodeLlama-34b-Python-hf","CodeLlama-34b-Instruct-hf"
]



def load_generator(model):
    """
    :param model: The model to use for completion.
    :return: Return the generator.
    """
    if model in GPT_FAMILY:
        generator = GPTAgent(model)
    elif model in LLAMA_FAMILY:
        generator = LLamaAgent(model)
    else:
        pass

    return generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613")
    args = parser.parse_args()

    generator = load_generator(args.model)


if __name__ == '__main__':
    main()