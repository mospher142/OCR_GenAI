import os
import json
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from typing import List
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Document(BaseModel):
    company_name: str = Field(description="Company Name")
    company_identifier: str = Field(description="Company Identifier Number")
    document_purpose: str = Field(description="Document Purpose")
    key_terms: str = Field(description="Key terms related with the document")


class Info(BaseModel):
    infomration: List[Document]


class TextToJsonProcessor:
    def __init__(self, model_name: str = "gpt-4o-mini") -> None:
        self.model = ChatOpenAI(model=model_name)
        self.parser = PydanticOutputParser(pydantic_object=Info)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "You are an AI assistant specialized in extracting key business details from documents in multiple languages. "
                 "Your goal is to extract relevant information accurately, translate it into English if necessary, and format the output "
                 "according to the specified JSON schema. Follow the instructions carefully to ensure precision and completeness.\n\n"
                 "### Instructions:\n"
                 "1. **Company Name**: Extract the full name of the company exactly as it appears in the document.\n"
                 "2. **Company Identifier**: Extract the unique identifier for the company, such as a registration or business number. "
                 "   - If the identifier contains non-numeric characters, convert them to numbers based on the closest resemblance.\n"
                 "   - Ensure the identifier is in the format: `####-###-###`.\n"
                 "   - If no identifier is present, explicitly mention 'Not Found'.\n"
                 "3. **Document Purpose**: Identify the purpose or intent of the document, such as 'Appointment of Directors' or 'Annual Report'. "
                 "   - Retrieve this information from fields such as 'Objet de l'acte' or equivalent terms in other languages.\n"
                 "   - Translate this purpose into English if necessary.\n"
                 "4. **Key Terms about the Document Purpose**: Extract detailed, relevant information related to the document's purpose. "
                 "   - For example, if the document concerns the appointment of directors, include terms such as 'Position Title', 'Effective Date', and any other important information. "
                 "   - Translate these terms into English as needed.\n\n"
                 "### Formatting Instructions:\n"
                 "{format_instructions}\n\n"
                 "Ensure your output is complete and follows the JSON schema exactly as specified."
                 ),
                ("human", "{query}"),

            ]
        ).partial(format_instructions=self.parser.get_format_instructions())
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


