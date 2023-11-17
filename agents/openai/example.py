import os
from gpt_wrapper import GPTAgent

os.environ["http_proxy"]="127.0.0.1:7890"
os.environ["https_proxy"]="127.0.0.1:7890"


engine = "gpt-3.5-turbo-instruct-0914"
max_tokens = 512
temperature = 0.2
stop = ["</Chinese>"]


prompt = "你知道乔布斯吗？"
print(prompt)
agent = GPTAgent(engine, max_tokens, temperature, stop)

response = agent.call_lm(prompt)


print(response)