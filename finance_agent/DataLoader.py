import pandas as pd
import os


DATASET_FILES = {
    "balance_sheet": "Balance_Sheet_final.csv",
    "price": "price_final.csv",
    "ratios_1": "ratios_1_final.csv",
    "ratios_2": "ratios_2_final.csv",
}

FILE_DESCRIPTIONS = {
    "balance_sheet": "Balance sheet information for companies, including assets, liabilities, and equity.",
    "price": "Pricing data, including price and return-related fields.",
    "ratios_1": "Key financial ratios for a subset of companies.",
    "ratios_2": "Additional financial ratios for the same subset of companies.",
}

BASE_CONTEXT_COLUMNS = ["Name", "BSE Code", "NSE Code", "Industry"]

class DataRepository :

    def __init__(self, data_dir):
        self.__data_dir = data_dir
        self.data_frames = {} # Loaded data
        self.data_frames_descriptions = {}
        self.__load_data()

    def __load_data(self) :

        #Build a set of industries (could be useful later on)
        industries = set()

        for file_id, file_name in DATASET_FILES.items() :
            file_path = os.path.join(self.__data_dir, file_name)

            #Load the current data frame
            current_data_frame = pd.read_csv(file_path)

            #Remove the join_key columns as it is not used
            if "join_key" in current_data_frame.columns :
                current_data_frame.drop(columns=["join_key"])

            #Normalize the loaded data
            current_data_frame["Name"] = current_data_frame["Name"].astype(str).str.strip()
            current_data_frame["NSE Code"] = current_data_frame["NSE Code"].apply(self.__normalize_nse_code)
            current_data_frame["BSE Code"] = current_data_frame["BSE Code"].apply(self.__normalize_bse_code)
            current_data_frame["Industry"] = current_data_frame["Industry"].astype(str).str.strip()

            #Add the industries in the overall set
            for industry in current_data_frame["Industry"].dropna() :
                industries.add(industry)

            #Save the loaded data
            self.data_frames[file_id] = current_data_frame
            self.data_frames_descriptions[file_id] = FILE_DESCRIPTIONS[file_id]


    def __normalize_nse_code(self, value) :
        # Treat missing NSE codes and "__NA__" as empty values.
        if value is None or pd.isna(value):
            return ""
        text = str(value).strip()
        return "" if text == "__NA__" else text


    def __normalize_bse_code(self, value) :
        # Treat missing BSE codes and "-1.0" as empty values.
        if value is None or pd.isna(value):
            return ""
        text = str(value).strip()
        return "" if text == "-1.0" else text