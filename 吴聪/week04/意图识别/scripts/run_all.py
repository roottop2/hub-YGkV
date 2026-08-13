# -*- coding: utf-8 -*-
"""一键全流程：生成数据 -> 训练三个模型 -> 建图谱 -> 评测。

运行：python scripts/run_all.py
"""
import subprocess
import sys
from pathlib import Path


def main():
    scripts_dir = Path(__file__).resolve().parent
    steps = [
        "01_generate_data.py",
        "02_train_ner.py",
        "03_train_relation.py",
        "04_build_graph.py",
        "05_train_intent.py",
        "06_evaluate.py",
    ]
    for step in steps:
        print(f"\n{'=' * 50}\n===== 运行 {step} =====\n{'=' * 50}")
        subprocess.run([sys.executable, str(scripts_dir / step)], check=True)
    print("\n全流程完成！可运行 Streamlit 界面：streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
