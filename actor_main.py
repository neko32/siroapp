from sirochatora.sirochatora import Sirochatora, Actor
from sirochatora.util.siroutil import ConfJsonLoader

import streamlit as st

import asyncio
from os import environ

from langchain_core.messages import AIMessage, HumanMessage


async def main():

    conf:ConfJsonLoader = ConfJsonLoader("sirochatora/conf.json")
    environ["LANGSMITH_API_KEY"] = conf._conf["LANGSMITH_API_KEY"]
    environ["LANGCHAIN_PROJECT"] = conf._conf["LANGCHAIN_PROJECT"]
    session_id = "nekonekoneko"

    sc:Sirochatora = Sirochatora(
        is_chat_mode = True, 
        session_id = session_id
    )
    actor:Actor = Actor(
        name = "みーこ",
        full_name = "猫宮 みーこ",
        persona_id = "meeko",
        image = "/tmp/meeko.jpg"
    )
    sc.add_system_message(actor.persona_system_message)
    st.title(f"{actor._name}とのお話 [SESSION:{session_id}]")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages = sc._msg_history.get_messages()

    for msg in st.session_state.messages:
        with st.chat_message(msg.type):
            st.write(msg.content)

    if ipt := st.chat_input():
        with st.chat_message("human"):
            st.write(ipt)
        st.session_state.messages.append(HumanMessage(ipt))

        #while True:
        ai_resp = await sc.query_async(ipt)
        st.session_state.messages.append(ai_resp)

        if isinstance(ai_resp.content, str):
            with st.chat_message("ai"):
                st.write(ai_resp.content)



if __name__ == "__main__":
    asyncio.run(main())