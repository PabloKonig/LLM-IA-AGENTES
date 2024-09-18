from threading import Thread
import re
import solara
import solara.lab
from typing import List
from typing_extensions import TypedDict
from langchain.schema import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
import os
import yaml

config = yaml.safe_load(open("config.yml"))
os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = config["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = str(config["LANGCHAIN_TRACING_V2"]).lower()
os.environ["LANGCHAIN_ENDPOINT"] = config["LANGCHAIN_ENDPOINT"]
os.environ["LANGCHAIN_PROJECT"] = config["LANGCHAIN_PROJECT"]
os.environ["LANGCHAIN_HUB_API_KEY"] = config["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_HUB_API_URL"] = config["LANGCHAIN_HUB_API_URL"]
os.environ["TAVILY_API_KEY"] = config["TAVILY_API_KEY"]

class MessageDict(TypedDict):
    role: str
    content: str

# Crea el agente LangGraph
memory = MemorySaver()
model = ChatOpenAI(model="gpt-4o-2024-08-06")
search = TavilySearchResults(max_results=2)
tools = [search]
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# Configuración para LangGraph
config_langgraph = {"configurable": {"thread_id": "abc123"}}

messages: solara.Reactive[List[MessageDict]] = solara.reactive([])

def add_chunk_to_ai_message(chunk: str):
    messages.value = [
        *messages.value[:-1],
        {
            "role": "assistant",
            "content": messages.value[-1]["content"] + chunk,
        },
    ]

@solara.component
def Page():
    solara.lab.theme.themes.light.primary = "#0000ff"
    solara.lab.theme.themes.light.secondary = "#0000ff"
    solara.lab.theme.themes.dark.primary = "#0000ff"
    solara.lab.theme.themes.dark.secondary = "#0000ff"
    title = "Chatbot con OpenAI"
    with solara.Head():
        solara.Title(f"{title}")

    with solara.Column(align="center"):
        user_message_count = len([m for m in messages.value if m["role"] == "user"])

        def send(message):
            messages.value = [*messages.value, {"role": "user", "content": message}]

        def response(message):
            messages.value = [*messages.value, {"role": "assistant", "content": ""}]
            
            def add_chunk_to_ai_message(chunk: str):
                messages.value = [
                    *messages.value[:-1],
                    {
                        "role": "assistant",
                        "content": messages.value[-1]["content"] + chunk,
                    },
            ]

            # Ejecuta el agente y procesa el streaming de la respuesta
            stream_response = agent_executor.stream({"messages": [HumanMessage(content=message)]}, config_langgraph)
            for chunk in stream_response:
                add_chunk_to_ai_message(chunk.message.value[-1].content)  # Añade el chunk al mensaje del asistente

        def result():
            if messages.value != []:
                response(messages.value[-1]["content"])

        result = solara.lab.use_task(result, dependencies=[user_message_count])

        with solara.lab.ChatBox(style={"position": "fixed", "overflow-y": "scroll", "scrollbar-width": "none", "-ms-overflow-style": "none", "top": "0", "bottom": "10rem", "width": "70%"}):
            for item in messages.value:
                with solara.lab.ChatMessage(
                    user=item["role"] == "user",
                    name="User" if item["role"] == "user" else "Chatbot",
                    avatar_background_color="#33cccc" if item["role"] == "assistant" else "#ff991f",
                    border_radius="20px",
                    style="background-color:darkgrey!important;" if solara.lab.theme.dark_effective else "background-color:lightgrey!important;"
                ):
                    item["content"] = re.sub(r'<\|im_end\|>', '', item["content"])
                    solara.Markdown(item["content"])

        solara.lab.ChatInput(send_callback=send, style={"position": "fixed", "bottom": "3rem", "width": "70%"})

Page()