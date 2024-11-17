from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class Document(BaseModel):
    company_name: str = Field(description="Company Name")
    company_identifier: str = Field(description="Company Identifier Number")
    document_purpose: str = Field(description="Document Purpose")
    key_terms: dict = Field(description="Key terms related with the document")


class Info(BaseModel):
    infomration: list[Document]


def get_prompt(parser: PydanticOutputParser) -> ChatPromptTemplate:
    """
    Returns the prompt template for structured data extraction.

    Args:
        parser (PydanticOutputParser): The parser for formatting output.

    Returns:
        ChatPromptTemplate: Configured prompt template.
    """
    return ChatPromptTemplate.from_messages(
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
             "   - Instead of concatenating all details into a single line, return them as a structured object"
             "### Formatting Instructions:\n"
             "{format_instructions}\n\n"
             "Ensure your output is complete and follows the JSON schema exactly as specified."
             ),
            ("human", "{query}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())
