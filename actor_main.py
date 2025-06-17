from sirochatora.sirochatora import Sirochatora, Actor
from sirochatora.util.siroutil import ConfJsonLoader

import streamlit as st

import asyncio
from os import environ
from PIL import Image

from langchain_core.messages import AIMessage, HumanMessage, trim_messages

def memclear(buf_size:int = 30):
    st.session_state.chat_hist = []

def customize_ui():
    # this may not work as expected
    st.markdown("""
    <style>
    .stChatMessageAvatar img {
        width: 256px !important;
        height: 256px !important;
        border-radius: 50% !important;
    }
    </style>
""", unsafe_allow_html=True)

async def main():

    conf:ConfJsonLoader = ConfJsonLoader("sirochatora/conf.json")
    environ["LANGSMITH_API_KEY"] = conf._conf["LANGSMITH_API_KEY"]
    environ["LANGCHAIN_PROJECT"] = conf._conf["LANGCHAIN_PROJECT"]
    avator_image_dir = f"{environ["NEKORC_PATH"]}/siroapp/avator"
    session_id = "nekonekoneko"

    st.set_page_config(layout = "wide")
    customize_ui()

    my_avatar_img = Image.open(f"{avator_image_dir}/choryo.png")

    st.sidebar.header("設定")

    available_models = ["gemma3:4b", "ollama3.2"]
    available_actors = ["みーこ", "ぴぴん", "とらきち", "しろ", "みけよん"]
    model_chosen = st.sidebar.selectbox(label = "LLMモデル", index = 0, options = available_models)
    actor_chosen = st.sidebar.selectbox(label = "ねこちゃん", index = 0, options = available_actors)
    temperature_as_int = st.sidebar.slider(label = "Temperature", min_value = 0, max_value = 10, value = 3, step = 1)
    hist_window = st.sidebar.slider(label = "チャットの履歴サイズ", min_value = 1, max_value = 100, value = 30, step = 1)

    print(f"built SC with model {model_chosen}, temperature {temperature_as_int / 10}")

    sc:Sirochatora = Sirochatora(
        model_name = model_chosen,
        temperature = temperature_as_int / 10,
        is_chat_mode = True, 
        session_id = session_id
    )
    actor:Actor = Actor(
        name = "みーこ",
        full_name = "猫宮 みーこ",
        persona_id = "meeko",
        image = f"{avator_image_dir}/meeko_256x256.png",
    )
    avator_img = Image.open(actor._image)
    avator_map = {
        "human": my_avatar_img,
        "ai": avator_img
    }


    sc.add_system_message(actor.persona_system_message)
    st.title(f"{actor._name}とのお話 [SESSION:{session_id}]")

    if "chat_hist" not in st.session_state:
        st.session_state.chat_hist = []
    st.session_state.chat_hist = sc._msg_history.get_messages()
    st.session_state.chat_hist = trim_messages(
        st.session_state.chat_hist, 
        token_counter = len,
        max_tokens = hist_window,
        strategy = 'last',
        start_on = "human"
    )
    print(f"trimmed messages with max_token = {hist_window} - current size: {len(st.session_state.chat_hist)}")

    for msg in st.session_state.chat_hist:
        with st.chat_message(msg.type, avatar = avator_map[msg.type]):
            st.write(msg.content)

    if ipt := st.chat_input():
        with st.chat_message("human"):
            st.write(ipt)
        st.session_state.chat_hist.append(HumanMessage(ipt))

        ai_resp = await sc.query_async(ipt)
        st.session_state.chat_hist.append(ai_resp)

        if isinstance(ai_resp.content, str):
            with st.chat_message("ai", avatar = avator_img):
                st.write(ai_resp.content)



if __name__ == "__main__":
    asyncio.run(main())