import os
import json
from pathlib import Path
from preprocessing import Preprocessing
from extration import TextToJsonProcessor


class Pipeline:
    def __init__(self, pdf_file: str, language: str = 'en', model_name: str = "gpt-4o-mini") -> None:
        """
        Initialize the Pipeline class with a PDF file and necessary configurations.
        """
        self.pdf_file = pdf_file
        self.language = language
        self.preprocessing = Preprocessing(pdf_file, language)
        self.text_processor = TextToJsonProcessor(model_name)

    def process(self) -> str:
        
        # Step 1: Preprocess the PDF
        print("Starting PDF preprocessing...")
        pdf_images_dir = self.preprocessing.pdf_to_images() 
        self.preprocessing.preprocess_images() 

        # Step 2: Extract text from preprocessed images
        print("Extracting text from preprocessed images...")
        extracted_texts_file = os.path.join(pdf_images_dir, "extracted_texts.txt")
        extracted_texts = self.preprocessing.process_and_extract_text_from_images()

        # Write extracted text to a file for processing
        with open(extracted_texts_file, "w", encoding="utf-8") as f:
            for image_name, texts in extracted_texts.items():
                f.write(f"### {image_name} ###\n")
                f.write("\n".join(texts) + "\n\n")

        # Step 3: Use the text processor to generate JSON
        print("Processing extracted text to generate JSON...")
        json_file_path = self.text_processor.process_file(extracted_texts_file)

        print("Pipeline processing complete.")
        return json_file_path


if __name__ == "__main__":
    pdf_path = "extract/24000001.pdf"  
    pipeline = Pipeline(pdf_file=pdf_path)
    result_json = pipeline.process()
    print(f"Generated JSON file: {result_json}")
