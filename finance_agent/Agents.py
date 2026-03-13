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
    
    def __invoke_llm(self, prompt) :
        raw = self.__llm.invoke(prompt)
        return json.loads(self.__normalize_json_response(raw.content))
        

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
        
        parsed = self.__invoke_llm(prompt)

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

        parsed = self.__invoke_llm(prompt)

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

        retrieved_row_indexes = {}
        entities_identified = True

        for financial_entity in financial_entities :
            current_entity = financial_entity.strip()
            current_entity_found = False

            for data_frame_key, data_frame in self.__data.data_frames.items() :
                matched_rows = data_frame.iloc[0:0]

                if data_frame_key not in retrieved_row_indexes :
                    retrieved_row_indexes[data_frame_key] = []

                #Try to find the entity by Name first using a SQL LIKE style match.
                name_tokens = [re.escape(token) for token in current_entity.split() if token]
                if name_tokens :
                    like_pattern = ".*".join(name_tokens)
                    matched_rows = data_frame[
                        data_frame["Name"].astype(str).str.contains(like_pattern, case=False, na=False, regex=True)
                    ]

                #Then try to find the entity by BSE code using an exact match.
                if matched_rows.empty :
                    bse_code = re.sub(r"\.0$", "", current_entity) # Remove the ".0" suffix, which is present for some codes
                    matched_rows = data_frame[
                        data_frame["BSE Code"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip() == bse_code
                    ]

                #Then try to find the entity by NSE code using an exact match.
                if matched_rows.empty :
                    nse_code = current_entity.upper()
                    matched_rows = data_frame[
                        data_frame["NSE Code"].astype(str).str.strip().str.upper() == nse_code
                    ]

                if len(matched_rows) > 1 :
                    raise ValueError(f"Ambiguous entity match for '{current_entity}' in dataset '{data_frame_key}'")

                if not matched_rows.empty :
                    row_indexes = matched_rows.index.tolist()

                    retrieved_row_indexes[data_frame_key].extend(row_indexes)
                    current_entity_found = True

            if not current_entity_found :
                entities_identified = False

        if not entities_identified :
            raise ValueError("Could not identify all requested financial entities")
        
        #If we arrive here it means that the entities have been identified.
        #It is now time to retrieve the columns to be used for the lookup or the analysis
        unique_columns_text = "; ".join(self.__data.unique_quantitative_columns)

        prompt = f"""You are the data retriever agent of a local financial analysis system.
        Use the columns information below to decide which one of them must be considered based on the user request. Each column contains quantitative data associated with financial entities (e.g. stocks).
        The user could have requested some data or some analyses based on some data.
        In case the request is of type "lookup", then we must just retrieve data from the available columns.
        In case the request is of type "reasoning, then the retrieved columns will be used later for doing the proper comparisons or calculations as requested by the user.
        Identify the column or the columns that fits most the data needed to accomplish the user request.

        The available columns are the following and they are all representing quantitative data associated with financial entities like e.g. stocks (columns separated by ";"):
        {unique_columns_text}

        Return as output JSON only with the list of columns to be considered as follows:
        {{
            "columns": []
        }}

        where "columns" is the list of columns containing the quantitative data to be considered for answering the user request.

        The request type is "{prompt_type}"
        
        The user request is described by the following prompt:
        {user_prompt}"""

        parsed = self.__invoke_llm(prompt)

        return {
            "columns": parsed["columns"]
        }


