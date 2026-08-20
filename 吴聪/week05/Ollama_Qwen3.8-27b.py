from openai import OpenAI


client = OpenAI(base_url="http://localhost:11434/v1", api_key="1234")


response = client.chat.completions.create(
    model="Qwen3.8-27b-u",
    messages=[
        {"role": "system", "content": "你是一个有用的助手"},
        {"role": "user", "content": "你好"}
    ],
    temperature=0.7,
    max_tokens=512
)


print(response.choices[0].message.content)