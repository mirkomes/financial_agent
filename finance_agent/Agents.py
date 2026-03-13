from langchain_google_genai import ChatGoogleGenerativeAI
from finance_agent.DataLoader import DataRepository
from typing_extensions import Any
import json, re

class Agents :
    def __init__(self, llm: ChatGoogleGenerativeAI, data: DataRepository) :
        self.__llm = llm
        self.__data = data

    def __normalize_json_response(self, response) :
        match = re.search(r"(?s).*?```json\s*(\{.*?\})\s*```.*", response)
        return match.group(1)

    #Classify the prompt between "Loopkup" and "Reasoning"
    def classify_prompt(self, user_prompt):
        prompt = f"""Classify the financial question into one of two classes:
        1) "lookup": direct data retrieval from the datasets
        2) "reasoning": a question that requires calculations, comparisons, or analysis

        Return as output JSON only with:
        {{
            "prompt_type": "lookup|reasoning",
            "rationale": "one short sentence"
        }}

        Question: {user_prompt}
        """
        
        raw = self.__llm.invoke(prompt)
        parsed = json.loads(self.__normalize_json_response(raw.content))

        #Check the validity of the prompt type
        prompt_type = parsed.get("prompt_type")
        if prompt_type.lower() != "lookup" and prompt_type.lower() != "reasoning" :
            raise ValueError("Unrecongnized prompt type: {}".format(prompt_type))
        
        return {
            "prompt_type": prompt_type,
            "rationale": parsed.get("rationale"),
        }
    
    #Retrieve the rows and columns from dataset to be considered for further analyses or lookups
    def retriever(self, user_prompt, prompt_type) :
        prompt = f"""You are the retriever agent of a local financial analysis system.
        Use the columns information below to decide which one of them must be considered for data retrieval based on the user request.
        The user could have requested some data by specifying the exact column name or by specifying a column name that is not perfectly equal to the available ones.
        In case an exact column name cannot be found, identify the column that fits most the data the user has requested.

        The available files and columns are the following:
        
        
        The user request is described by the following lookup prompt:
        {user_prompt}"""