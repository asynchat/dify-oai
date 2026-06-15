import os
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("VALID_API_KEYS", "your_valid_api_key_here").split(",")[0]
BASE_URL = "http://localhost:8000/v1"
MODEL = os.getenv("DIFY_TEST_MODEL", "Test")


async def run_case(client: AsyncOpenAI, include_tool_extensions: bool):
    label = "扩展模式" if include_tool_extensions else "严格 OpenAI 模式"
    print(f"\n{'=' * 50}")
    print(f"[*] {label} (include_tool_extensions={include_tool_extensions})")
    print(f"{'=' * 50}")

    extra_body = {"include_tool_extensions": include_tool_extensions}

    print("\n[*] 非流式请求...")
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "请分析。"}],
            stream=False,
            extra_body=extra_body,
        )
        message = response.choices[0].message
        print("[+] content:", message.content)
        print("[+] tool_calls:", message.tool_calls)
        choice_dump = response.choices[0].model_dump()
        if choice_dump.get("tool_results"):
            print("[+] tool_results:", choice_dump["tool_results"])
    except Exception as e:
        print(f"[-] 非流式失败: {e}")

    print("\n[*] 流式请求...")
    try:
        stream = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "请分析。"}],
            stream=True,
            extra_body=extra_body,
        )
        print("[+] 流式 chunk:")
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if getattr(delta, "tool_calls", None):
                print("  tool_calls:", delta.tool_calls)
            if getattr(delta, "tool_results", None):
                print("  tool_results:", delta.tool_results)
            if delta.content:
                print(delta.content, end="", flush=True)
        print("\n[+] 流式结束")
    except Exception as e:
        print(f"[-] 流式失败: {e}")


async def test_chat_completion():
    print(f"[*] OpenAI 客户端: base_url={BASE_URL}, api_key={API_KEY[:8]}...")
    print("[*] Dify 原始事件日志在服务端终端，或文件: mcp-server/logs/dify_raw_events.jsonl")
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

    await run_case(client, include_tool_extensions=False)
    await run_case(client, include_tool_extensions=True)


if __name__ == "__main__":
    asyncio.run(test_chat_completion())
