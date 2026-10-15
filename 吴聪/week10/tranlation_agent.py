import os

os.environ["OPENAI_API_KEY"] = "sk-c08c5149dd8946c38c069f4632913516"
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"

import asyncio
from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner

from agents import set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")  # 强制 SDK 使用 Chat Completions API，而非默认的 Responses API。
set_tracing_disabled(True)  # 彻底关闭 SDK 的链路追踪（Tracing） 功能

translation_agent = Agent(
    name="translator",
    model="deepseek-flash",
    instructions="您是专业的翻译专家。如果我输入的是中文翻译成英文, 如果是英文则翻译成中文。",
)

async def main():

    while True:
        query = input("请输入文本:")
        result = await Runner.run(translation_agent, query)
        print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
