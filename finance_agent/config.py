import os
from pathlib import Path
import json

class AppConfig:

    def __init__(self):
        self.__config_path = os.path.join(Path(__file__).resolve().parents[1], "config.json")
        self.google_api_key = None
        self.model = None
        self.data_dir = None
        self.__load_config()

    
    def __load_config(self):
        with open(self.__config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        self.google_api_key = config_data.get("google_api_key")
        self.model = config_data.get("model")
        self.data_dir = config_data.get("data_dir")

