# -*- coding: utf-8 -*-
"""Streamlit 网页聊天界面。

启动：streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

# 保证从任意目录启动都能导入项目包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from ecommerce_qa.pipeline import EcommerceQA  # noqa: E402

st.set_page_config(page_title="电商智能问答系统", page_icon="🛒", layout="wide")


@st.cache_resource(show_spinner="正在加载模型与知识图谱...")
def get_qa() -> EcommerceQA:
    """加载问答系统（Streamlit 缓存，只加载一次）。"""
    return EcommerceQA()


def render_status(qa: EcommerceQA) -> None:
    """侧边栏：展示各模块就绪状态。"""
    with st.sidebar:
        st.subheader("🛠️ 系统状态")
        for name, ready in qa.status().items():
            mark = "✅ 就绪" if ready else "❌ 未就绪"
            st.write(f"{name}: {mark}")
        st.divider()
        st.caption("提示：未就绪的模块请先运行\n`python scripts/run_all.py` 完成训练与建图。")


def main() -> None:
    st.title("🛒 电商智能问答系统")
    st.caption("基于知识图谱 + RAG 的智能客服：理解意图 → 检索知识 → 大模型生成")

    qa = get_qa()
    render_status(qa)

    # 初始化会话状态（多轮对话）
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.qa_session = {"slots": {}}

    # 渲染历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("detail"):
                with st.expander("🔍 调试信息（意图/槽位/检索知识）"):
                    st.json(msg["detail"])

    # 输入框
    if prompt := st.chat_input("请输入您的问题，例如：我的订单到哪了？"):
        # 用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 调用问答流水线
        result = qa.answer(prompt, st.session_state.qa_session)

        # 助手回答
        detail = {
            "意图": result["intent"],
            "意图置信度": round(result["intent_prob"], 3),
            "槽位": result["slots"],
            "策略": result["strategy"],
            "回答来源": result["source"],
            "检索到的知识": result["knowledge"],
        }
        with st.chat_message("assistant"):
            st.markdown(result["answer"])
            with st.expander("🔍 调试信息（意图/槽位/检索知识）"):
                st.json(detail)

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "detail": detail,
        })


if __name__ == "__main__":
    main()
