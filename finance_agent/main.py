
from argparse import ArgumentParser
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness
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

#Prompts evaluation with Ragas
def evaluate_prompt_results(prompt_results, config) :

    evaluation_rows = []
    for prompt_result in prompt_results :
        evaluation_rows.append({
            "user_input": prompt_result["prompt"],
            "response": prompt_result["answer_text"],
            "retrieved_contexts": prompt_result["retrieved_contexts"]
        })

    evaluation_dataset = EvaluationDataset.from_list(evaluation_rows)
    evaluator_llm = ChatGoogleGenerativeAI(
        model = config.model,
        google_api_key = config.google_api_key,
        temperature = 0.0
    )
    evaluation_result = evaluate(
        dataset = evaluation_dataset,
        metrics = [Faithfulness()],
        llm = LangchainLLMWrapper(evaluator_llm)
    )

    evaluation_frame = evaluation_result.to_pandas()
    for index, prompt_result in enumerate(prompt_results) :
        prompt_result["faithfulness_score"] = evaluation_frame.iloc[index]["faithfulness"]

    return prompt_results

def print_prompt_results(prompt_results) :
    separator = "=" * 55

    for prompt_result in prompt_results :
        print(separator)
        print("User prompt:")
        print(prompt_result["prompt"])
        print()
        print("Answer:")
        print(prompt_result["answer_text"])
        print()
        print("Faithfulness score:")
        print(prompt_result.get("faithfulness_score"))
        print()
        print("Citations:")

        for file_name, citation_data in prompt_result["citations"].items() :
            print(f"- {file_name}")
            print(f"  rows: {citation_data['rows']}")
            print(f"  columns: {citation_data['columns']}")

        print(separator)
        print()

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
        prompts = ["Compare the debt levels of 5PAISA vs Ador Fontech",
                   "What is the price of AARON and 63 Moons Tech.?",
                   "Compare the return on equity of AARON vs 5PAISA",
                   "Did 63MOONS increase its working capital compared to the preceding year?",
                   "What is the debt to equity of AARON?"]


    #Load the configuration
    config_obj = AppConfig()

    #Load the available data
    available_data = DataRepository(data_dir=os.path.join(package_root, "data"))

    #Initialize the agent
    finance_agent = FinanceAgentGraph(config=config_obj, data=available_data)

    #Execute the agent for each prompt
    prompt_results = []
    for prompt in prompts :
        current_prompt_result = finance_agent.run(prompt=prompt)
        prompt_results.append(current_prompt_result)

    #Evaluation
    evaluate_prompt_results(prompt_results, config_obj)

    #Print results
    print_prompt_results(prompt_results)



if __name__ == "__main__" :
    raise SystemExit(main())
