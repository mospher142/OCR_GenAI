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
        self.prompt = get_prompt(self.parser)  # Get prompt from the utility function
        self.chain = self.prompt | self.model | self.parser

    def process_file(self, file_path: str) -> str:
        """
        Processes a text file and outputs a JSON file with the extracted information.

        Args:
            file_path (str): Path to the text file.

        Returns:
            str: Path to the generated JSON file.
        """
        if not os.path.isfile(file_path) or not file_path.endswith(".txt"):
            raise ValueError("The provided file must be a valid .txt file.")

        with open(file_path, "r", encoding="utf-8") as f:
            document = f.read()

        folder_name = Path(file_path).parent.name
        json_file_path = os.path.join(os.path.dirname(file_path), f"{folder_name}.json")

        print(f"Processing document from: {file_path}")
        result = self.chain.invoke({"query": document})

        with open(json_file_path, "w", encoding="utf-8") as json_file:
            json.dump(result.dict(), json_file, indent=4)

        print(f"Processed data saved to: {json_file_path}")
        return json_file_path
