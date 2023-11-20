# author: wanghanbin
# Time: 2023/11/20
# desc: The main function of the project

import argparse
from agents.gpt.gpt_wrapper import GPTAgent
from agents.llama.llama_wrapper import LLamaAgent
GPT_FAMILY=[
    ""
]

LLAMA_FAMILY=[
    ""
]



def load_generator(model_type):
    """
    :param model_type: The model to use for completion.
    :return: Return the generator.
    """
    if model_type in GPT_FAMILY:
        generator = GPTAgent(model_type)
    elif model_type in LLAMA_FAMILY:
        generator = LLamaAgent(model_type)
    else:
        pass

    return generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="gpt-3.5-turbo-0613")
    args = parser.parse_args()

    generator = load_generator(args.model_type)


if __name__ == '__main__':
    main()