from langchain_google_genai import ChatGoogleGenerativeAI
from finance_agent.DataLoader import BASE_CONTEXT_COLUMNS, DATASET_FILES, DataRepository
from typing_extensions import Any
import json, re

class Agents :
    def __init__(self, llm: ChatGoogleGenerativeAI, data: DataRepository) :
        self.__llm = llm
        self.__data = data

    def __normalize_json_response(self, response) :
        match = re.search(r"(?s).*?```json\s*(\{.*?\})\s*```.*", response)
        return match.group(1)
    
    def __invoke_llm_json_response(self, prompt) :
        raw = self.__llm.invoke(prompt)
        return json.loads(self.__normalize_json_response(raw.content))
    
    def __invoke_llm_text_response(self, prompt) :
        raw = self.__llm.invoke(prompt)
        return raw.content
        

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
        
        parsed = self.__invoke_llm_json_response(prompt)

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

        parsed = self.__invoke_llm_json_response(prompt)

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

        parsed = self.__invoke_llm_json_response(prompt)

        return {
            "columns": parsed["columns"],
            "rows": retrieved_row_indexes
        }
    
    #Given the identified rows and columns, return the analysis requested by the user
    #This analyzer is executed only in case of prompt having type "reasoning"
    def analyzer(self, user_prompt, rows, columns) :

        #Load the data needed for the llm
        llm_context, columns_by_file = self.__load_data_for_llm(rows, columns)

        prompt = f"""You are a financial expert and you must give an answer to the user request provided below based on the available data.
        The available data that has been selected for providing the user an answer is the following:
        
        {llm_context}
        
        The user request is the following:
        {user_prompt}
        
        Provide the user a coincise response that satisfies the request and provide also a motivation. Do the proper calculations if the provided data
        does not directly expose the values requested by the user."""

        llm_response = self.__invoke_llm_text_response(prompt)

        return {
            "columns_by_file": columns_by_file,
            "final_response": llm_response
        }
    
    #Given the identified rows and columns, return the data requested by the user
    #This responder is executed only in case of prompt having type "lookup"
    def lookup_responder(self, user_prompt, rows, columns) :
        
        #Load the data needed for the llm
        llm_context, columns_by_file = self.__load_data_for_llm(rows, columns)

        prompt = f"""You are a financial expert and you must give an answer to the user request provided below based on the available data.
        The available data that has been selected for providing the user an answer is the following:
        
        {llm_context}
        
        The user request is the following:
        {user_prompt}
        
        Provide the user a coincise response that satisfies the request and provide also a motivation.
        The available data is expected to answer the user request without doing any complex calculation."""

        llm_response = self.__invoke_llm_text_response(prompt)

        return {
            "columns_by_file": columns_by_file,
            "final_response": llm_response
        }
    
    #Used to cite the data used for giving the user the final answer
    def cite_data(self, rows, columns_by_file) :
        citations = {}

        for file_id, selected_columns in columns_by_file.items() :
            #Consider only files where at least one row has been considered
            selected_rows = rows.get(file_id, [])
            if not selected_rows :
                continue

            #For each considered file, include the cited column of that file
            available_columns = self.__data.data_frames[file_id].columns
            citation_columns = []
            for column in BASE_CONTEXT_COLUMNS + selected_columns :
                if column in available_columns and column not in citation_columns :
                    citation_columns.append(column)

            citations[DATASET_FILES[file_id]] = {
                "rows": selected_rows,
                "columns": citation_columns
            }

        return {
            "citations": citations
        }


    def __load_data_for_llm(self, rows, columns) :
        #Load all the needed data given the specified rows and columns
        #rows is a dictionary specifying the indexes of the rows to be considered for each available file
        #columns is a list of columns to be considered
        file_priority = ["balance_sheet", "price", "ratios_1", "ratios_2"]
        active_files = [file_id for file_id in file_priority if rows.get(file_id)]
        columns_by_file = {}
        unique_columns = list(dict.fromkeys(columns))

        # Resolve each requested column on the active files only, following the file priority.
        for column in unique_columns :
            selected_file = None

            for file_id in active_files :
                if column in self.__data.data_frames[file_id].columns :
                    selected_file = file_id
                    break

            if selected_file is None :
                raise ValueError(f"Column '{column}' is not available in the selected rows")

            if selected_file not in columns_by_file :
                columns_by_file[selected_file] = []

            columns_by_file[selected_file].append(column)

        loaded_data = {}

        # Load only the requested rows and the columns assigned to each file.
        for file_id in active_files :
            if file_id not in columns_by_file :
                continue

            selected_rows = rows[file_id]
            selected_columns = columns_by_file[file_id]
            context_columns = []

            for column in BASE_CONTEXT_COLUMNS + selected_columns :
                if column in self.__data.data_frames[file_id].columns and column not in context_columns :
                    context_columns.append(column)

            loaded_data[file_id] = self.__data.data_frames[file_id].iloc[selected_rows][context_columns].copy()

        context_sections = []

        # Convert each filtered dataframe into compact JSON text for the LLM context.
        for file_id in active_files :
            if file_id not in loaded_data :
                continue

            file_rows = loaded_data[file_id].to_dict(orient="records")
            file_context_text = json.dumps(file_rows, ensure_ascii=True)
            context_sections.append(f"{file_id}: {file_context_text}")

        llm_context = "\n".join(context_sections)

        return llm_context, columns_by_file


