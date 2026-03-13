from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict, Any

from finance_agent.config import AppConfig
from finance_agent.DataLoader import DataRepository
from finance_agent.Agents import Agents

class AgentState(TypedDict):
    prompt: str #Prompt asked by the user
    prompt_type: str # Can be "Lookup" or "Reasoning"
    classifier_result: dict[str, Any]
    entities: list[str]
    columns: list[str]
    answer_text: str

#This is the orchestrator
class FinanceAgentGraph :

    def __init__(self, config: AppConfig, data: DataRepository) :
        self.__config = config
        self.__data = data
        self.__llm = ChatGoogleGenerativeAI(
            model = self.__config.model,
            google_api_key = self.__config.google_api_key,
            temperature = 0.0 # Keep this deterministic
        )
        self.__agents = Agents(llm=self.__llm, data=self.__data)
        self.__graph = self.__compile_graph()

    def __compile_graph(self) :
        graph_builder = StateGraph(AgentState)

        #First agent will classify the type of prompt between "Lookup" or "Reasoning"
        graph_builder.add_node("classify", self.__classify_prompt)
        graph_builder.add_node("entity_identifier", self.__entity_identifier)
        graph_builder.add_node("data_retriever", self.__data_retriever)

        graph_builder.add_edge(START, "classify")
        graph_builder.add_edge("classify", "entity_identifier")
        graph_builder.add_edge("entity_identifier", "data_retriever")
        graph_builder.add_edge("data_retriever", END)
        return graph_builder.compile()


    def __classify_prompt(self, state: AgentState) :
        classifier_result = self.__agents.classify_prompt(state["prompt"])
        return {
            "prompt_type": classifier_result["prompt_type"],
            "classifier_result": classifier_result
        }
    
    def __entity_identifier(self, state: AgentState) :
        entity_identifier_result = self.__agents.entity_identifier(state["prompt"])
        return {
            "entities": entity_identifier_result["entities"]
        }
    
    def __data_retriever(self, state: AgentState) :
        data_retriever_result = self.__agents.retriever(state["prompt"], state["prompt_type"], state["entities"])
        return {
            "columns": data_retriever_result["columns"]
        }
        
            
    def run(self, prompt) :
        final_state = self.__graph.invoke({"prompt": prompt})
        return {
            "prompt": prompt,
            "prompt_type": final_state["prompt_type"],
            "classifier_result": final_state["classifier_result"]
        }