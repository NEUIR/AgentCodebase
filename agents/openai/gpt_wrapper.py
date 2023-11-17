import backoff
import openai
import os

openai.api_key = "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"

class GPTAgent():
    def __init__(self, model, max_tokens, temperature, stop):
        """
        :param model: The model to use for completion.
        :param prompt: The prompt(s) to generate completions for.
        :param max_tokens: The maximum number of tokens to generate.
        :param temperature: The sampling temperature.
        :param stop:The stop sequence or a list of stop sequences.
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stop = stop


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
    def call_lm(self,prompt):
        response = openai.Completion.create(model=self.model,
                                            prompt=prompt,
                                            max_tokens=self.max_tokens,
                                            temperature=self.temperature,
                                            stop=self.stop)

        return response.choices[0].text.strip()
