# author: wanghanbin
# Time: 2023/11/20
# desc: The main function of the project

import argparse
from agents.gpt.gpt_wrapper import GPTAgent
from agents.llama.llama_wrapper import LLamaAgent
from agents.baichuan.baichuan_wrapper import BaichuanAgent
from agents.models import GPT_FAMILY, LLAMA_FAMILY, BAICHUAN_FAMILY

def load_generator(model):
    """
    :param model: The model to use for completion.
    :return: Return the generator.
    """
    if model in GPT_FAMILY:
        generator = GPTAgent(model)
    elif model in LLAMA_FAMILY:
        generator = LLamaAgent(model)
    elif model in BAICHUAN_FAMILY:
        generator = BaichuanAgent(model)
    else:
        pass

    return generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613",help="The model to use for completion.")
    args = parser.parse_args()

    generator = load_generator(args.model)

    # See GPT call example in agents/gpt/gpt_wrapper.py/if __name__ == '__main__':
    # See LLama call example in agents/llama/llama_wrapper.py/if __name__ == '__main__':
    # See Baichuan call example in agents/baichuan/baichuan_wrapper.py/

    # Your call code here


if __name__ == '__main__':
    main()