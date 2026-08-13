# -*- coding: utf-8 -*-
"""脚本 07：命令行问答演示。

运行：python scripts/07_demo_cli.py
输入 "exit" 或 "退出" 结束。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecommerce_qa.pipeline import EcommerceQA  # noqa: E402


def main():
    qa = EcommerceQA()
    print("=" * 50)
    print("电商智能问答系统（命令行演示）")
    print("系统状态：")
    for name, ready in qa.status().items():
        print(f"  {name}: {'就绪' if ready else '未就绪'}")
    print("输入 exit 退出\n" + "=" * 50)

    session = {"slots": {}}
    while True:
        try:
            question = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question in ("exit", "退出"):
            break

        result = qa.answer(question, session)
        print(f"  意图: {result['intent']}（置信度 {result['intent_prob']:.2f}）")
        print(f"  槽位: {result['slots']}")
        print(f"  策略: {result['strategy']}  来源: {result['source']}")
        if result["knowledge"]:
            print("  检索到知识:")
            for k in result["knowledge"][:5]:
                print(f"    - {k}")
        print(f"  回答: {result['answer']}")
        print("-" * 50)


if __name__ == "__main__":
    main()
