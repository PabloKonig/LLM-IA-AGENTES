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

# Import relevant functionality
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# Create the agent
model = ChatOpenAI(model="gpt-4o-2024-08-06")
search = TavilySearchResults(max_results=5)
tools = [search]
agent_executor = create_react_agent(model, tools)

# Use the agent
def run_agent(query):
    """Ejecuta el agente y maneja posibles errores."""
    try:
        input_data = {"messages": [HumanMessage(content=query)]}  
        response = agent_executor.invoke(input_data)  # Ejecuta el agente
        respuesta_final = response["messages"][-1].content
        return  respuesta_final if respuesta_final else "No se encontró respuesta."
    except Exception as e:
        print(f"Error al ejecutar el agente: {e}")
        return "Lo siento, hubo un error al procesar tu solicitud."

response = run_agent("Datos de 17/09/2024 de la economía Argentina")
print("Chatbot:", response)



