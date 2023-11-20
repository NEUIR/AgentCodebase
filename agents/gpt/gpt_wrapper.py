import backoff
import openai
import os


# openai.api_key = "PUT YOUR KEY HERE"
openai.api_key = "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"

class GPTAgent():
    def __init__(self, model):
        """
        :param model: The model to use for completion.
        """
        self.model = model


    @backoff.on_exception(
        backoff.fibo,
        # https://platform.openai.com/docs/guides/error-codes/python-library-error-types
        (
            openai.error.APIError,
            openai.error.Timeout,
            openai.error.RateLimitError,
            openai.error.ServiceUnavailableError,
            openai.error.APIConnectionError,
            KeyError,
        ),
    )
    def __call__(self, system_prompt, user_prompt, max_tokens, temperature, stop_words):
        """
        :param system_prompt: The system prompt to prompt the model.
        :param user_prompt: The user requirments.
        :param max_tokens: The maximum number of tokens to generate.
        :param temperature: The sampling temperature.
        :param stop: The stop sequence or a list of stop sequences.
        :return: Return the response and the usage of the model.
        """
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop_words,
        )

        return response.choices[0].message["content"].strip(), response["usage"]


if __name__ == '__main__':
    import os
    os.environ["http_proxy"] = "127.0.0.1:7890"
    os.environ["https_proxy"] = "127.0.0.1:7890"

    # set the model
    model = "gpt-3.5-turbo-0613"
    generator = GPTAgent(model)

    # set the parameters
    system_prompt ="你是一个百事通，请你帮我回答下面的问题"
    user_prompt = "你知道乔布斯吗？"
    max_tokens = 512
    temperature = 0
    stop_words = ["</Chinese>"]


    print("    System Prompt: ", system_prompt)
    print("    User Prompt: ", user_prompt)

    # call the model,invoke the __call__ function
    response, usage = generator(system_prompt,user_prompt, max_tokens, temperature, stop_words)

    print("    Response: ", response)
    print("    Usage: ", usage)
