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
    
    #This is used to identify for which entities the user is requiring the data or the analysis
    def entity_identifier(self, user_prompt) :
        prompt = f"""The user question provided below specifies a financial question that could include data retrieval or analysis about one or more financial entities (e.g.: a stock).
        The financial entity could be specified by name or by a ticker represented by the BSE Code (i.e.: the Bombay Stock Exchange code) or by the NSE Code (i.e.: National Stock Exchange code)
        The available data the user could ask question about is within the context of the Indian stock market.

        Return as output JSON only with the list entities as follows:
        {{
            "entities": []
        }}

        where "entities" is the list of financial entities identified in the text of the user question.
        
        User question: {user_prompt}
        """

        raw = self.__llm.invoke(prompt)
        parsed = json.loads(self.__normalize_json_response(raw.content))

        if len(parsed["entities"]) > 2 :
            raise NotImplementedError("Cannot manage more than 2 entities in a request")
        
        return {
            "entities": parsed["entities"]
        }

    
    #Retrieve the rows and columns from dataset to be considered for further analyses or lookups
    def retriever(self, user_prompt, prompt_type, financial_entities) :

        #Check first the rows to be considered for the analysis requested by the user
        #Maximum 2 rows will be retrieved
        #Rows identification is performed programmatically. The datasets are used as a KB in this case.

        entities_identified = True
        for financial_entity in financial_entities :
            for data_frame_key, data_frame in self.__data.data_frames.itmes() :
                
                #Try to find the entity by Name first (with wildcards)

                #Then try to find the entity by BSE code (exact key)

                #Then try to find the entity by NSE code (exact key)



        prompt = f"""You are the retriever agent of a local financial analysis system.
        Use the columns information below to decide which one of them must be considered for data retrieval based on the user request.
        The user could have requested some data by specifying the exact column name or by specifying a column name that is not perfectly equal to the available ones.
        In case an exact column name cannot be found, identify the column that fits most the data the user has requested.

        The available files and columns are the following:
        
        
        The user request is described by the following lookup prompt:
        {user_prompt}"""