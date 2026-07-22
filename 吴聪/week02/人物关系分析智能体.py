import os
from pathlib import Path

import json
from dotenv import load_dotenv
from openai import OpenAI

_ENV_PATH = Path(__file__).resolve().parent / "llm.deepseek.env"
load_dotenv(_ENV_PATH)

api_key = os.getenv("LLM_API_KEY", "")
base_url = os.getenv("LLM_BASE_URL", "")
model = os.getenv("LLM_MODEL", "")

client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

system_prompt = """
你是一个人物关系梳理大师。请将人物关系梳理为json格式。
输入示例：小明喜欢小姚，但是小姚喜欢小王, 小明和小王是好朋友。
json输出示例：
[
    {
        "source": "小明"
        "relation"："爱慕"
        "target": "小姚"
    },
    {
        "source": "小姚"
        "relation"："爱慕"
        "target": "小王"
    },
    {
        "source": "小明"
        "relation"："朋友"
        "target": "小王"
    }
]
"""

message = [{"role": "system", "content": system_prompt}]


if __name__ == "__main__":
    while True:
        question = input("输入需要分析的关系(q退出)：")
        if question == 'q':
            break

        message.append({"role": "user", "content": question})

        response = client.chat.completions.create(
            model=model,
            messages=message,
            stream=False,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
            response_format={
                'type': 'json_object'
            }
        )
        try:
            data = json.loads(response.choices[0].message.content, )
            print(json.dumps(data, ensure_ascii=False, indent=2))
        except json.JSONDecodeError:
            print("返回的不是有效JSON")

        message.append(response.choices[0].message)