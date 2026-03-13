
from argparse import ArgumentParser
from pathlib import Path
import json
import sys
import os

# For local debug ONLY
IS_DEBUG = True
if __package__ is None or __package__ == "":
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from finance_agent.config import AppConfig
from finance_agent.DataLoader import DataRepository
from finance_agent.FinanceAgent import FinanceAgentGraph

def main(argv: list[str] | None = None) :

    # Initialize the prompts to be elaborated
    prompts = []
    if not IS_DEBUG :
        #Read the arguments
        argument_parser = ArgumentParser()
        args = argument_parser.parse_args(argv)

        if args.query is not None :
            prompts.append(args.query)
        elif args.batch_file_path is not None :
            
            # Load the file containing the batch of prompts
            batch_file_path = Path(args.batch_file_path).resolve()
            with open(batch_file_path, 'r', encoding="utf-8") as f:
                batch_prompts = json.load(f)

            #Append all the prompts to be elaborated
            for prompt in batch_prompts :
                prompts.append(prompt)
    else :
        #DEBUG MODE
        prompts = ["What is the price of AARON and 63 Moons Tech.?"]


    #Load the configuration
    config_obj = AppConfig()

    #Load the available data
    available_data = DataRepository(data_dir=os.path.join(package_root, "data"))

    #Initialize the agent
    finance_agent = FinanceAgentGraph(config=config_obj, data=available_data)

    #Execute the agent for each prompt
    for prompt in prompts :
        current_prompt_result = finance_agent.run(prompt=prompt)



if __name__ == "__main__" :
    raise SystemExit(main())