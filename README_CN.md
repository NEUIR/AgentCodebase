# AgentCodebase
目前，该项目支持GPT，LLama，Baichuan系列模型的推理。

## Quick Start

### GPT 

完整调用示例可以参考[代码](https://github.com/NEUIR/AgentCodebase/blob/54d97bb62e9ed7f117970903e7fbd99265fa9623/agents/gpt/gpt_wrapper.py#L52)

1. 在文件[gpt_wrapper.py](https://github.com/NEUIR/AgentCodebase/blob/54d97bb62e9ed7f117970903e7fbd99265fa9623/agents/gpt/gpt_wrapper.py#L7)设置api_key

   ```python
   openai.api_key = "sk-xxxx"
   ```

2. 初始化generator

   导入python package.

   ```
   from agents.gpt.gpt_wrapper import GPTAgent
   ```

   定义使用的模型

   ```python
   model = "gpt-3.5-turbo-0613"
   ```

   目前支持的模型有：

   ```python
   GPT_FAMILY=[
       # Model Description:https://platform.openai.com/docs/models/gpt-3-5
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
   ```

   初始化Agent对象

   ```
   generator = GPTAgent(model)
   ```

3. 定义超参数

   ```python
   # set the parameters
   system_prompt ="你是一个百事通，请你帮我回答下面的问题"
   user_prompt = "你知道乔布斯吗？"
   max_tokens = 512
   temperature = 0
   stop_words = ["</Chinese>"]
   ```

4. 调用模型

   ```python
   response, usage = generator(system_prompt,user_prompt, max_tokens, temperature, stop_words)
   ```




### Llama







### Baichuan
