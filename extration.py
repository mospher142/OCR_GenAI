from prompt import get_prompt, Info
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
import os
import json
from pathlib import Path

class TextToJsonProcessor:
    def __init__(self, model_name: str = "gpt-4o-mini") -> None:
        self.model = ChatOpenAI(model=model_name)
        self.parser = PydanticOutputParser(pydantic_object=Info)
        self.prompt = get_prompt(self.parser) 
        self.chain = self.prompt | self.model | self.parser

    def process_file(self, file_path: str) -> str:
        """
        Processes a text file and outputs a JSON file with the extracted information.

        Args:
            file_path (str): Path to the text file.

        Returns:
            str: Path to the generated JSON file.
        """
        # Validate the input file
        if not os.path.isfile(file_path) or not file_path.endswith(".txt"):
            raise ValueError("The provided file must be a valid .txt file.")

        # Read the content of the text file
        with open(file_path, "r", encoding="utf-8") as f:
            document = f.read()

        # Determine the folder where the JSON should be saved
        pdf_folder = Path(file_path).parent.name  
        results_dir = os.path.join("results", pdf_folder)
        
        # Ensure the results directory exists
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Define the JSON file path inside the folder
        json_file_name = f"{Path(file_path).stem}.json"
        json_file_path = os.path.join(results_dir, json_file_name)

        # Process the document and generate JSON content
        print(f"Processing document from: {file_path}")
        result = self.chain.invoke({"query": document})

        # Write the JSON file into the results folder
        with open(json_file_path, "w", encoding="utf-8") as json_file:
            json.dump(result.dict(), json_file, indent=4)

        print(f"Processed data saved to: {json_file_path}")
        return json_file_path


