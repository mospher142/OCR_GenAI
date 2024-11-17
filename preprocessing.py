import os
import cv2
import fitz
import easyocr
import numpy as np
from typing import Dict, List


class Preprocessing:
    def __init__(self, pdf_file: str, language: str = 'en') -> None:
        if not os.path.isfile(pdf_file):
            raise FileNotFoundError(f"The file '{pdf_file}' does not exist.")
        if not pdf_file.endswith('.pdf'):
            raise ValueError(f"The file '{pdf_file}' is not a valid PDF.")
        
        self.pdf_file: str = pdf_file
        self.language: str = language
        self.pdf_name: str = os.path.splitext(os.path.basename(pdf_file))[0]
        self.output_dir: str = f"{self.pdf_name}"
        
        try:
            # This helps the EasyOCR reader to extract better the data based on the language
            self.reader = easyocr.Reader([language]) 
        except Exception as e:
            raise RuntimeError(f"Failed to initialize EasyOCR reader: {str(e)}")
    
    def pdf_to_images(self, zoom: int = 6) -> str:
        """
        Converts the PDF to images, saving them in a directory named after the PDF.
        Returns the directory path containing the images.
        """
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            existing_files = [f for f in os.listdir(self.output_dir) if f.endswith(".png")]
            if existing_files:
                print(f"Images already exist in '{self.output_dir}'.")
                return self.output_dir
            
            doc = fitz.open(self.pdf_file)
            page_count = doc.page_count
            print(f"Converting PDF '{self.pdf_file}' with {page_count} pages into images...")

            mat = fitz.Matrix(zoom, zoom)
            for i in range(page_count):
                output_path = os.path.join(self.output_dir, f"image_{i+1}.png")
                try:
                    page = doc.load_page(i)
                    pix = page.get_pixmap(matrix=mat)
                    pix.save(output_path)
                    print(f"Saved {output_path}")
                except Exception as e:
                    print(f"Failed to process page {i+1}: {str(e)}")
            
            doc.close()
            print(f"Converted {page_count} pages to images in '{self.output_dir}'.")
            return self.output_dir
        
        except Exception as e:
            raise RuntimeError(f"Error during PDF-to-image conversion: {str(e)}")

    def preprocess_images(self) -> None:
        """
        Preprocesses the images extracted from the PDF by cropping and applying filters.
        Saves the processed images in the same directory with a 'processed_' prefix.
        """
        try:
            for image_file in os.listdir(self.output_dir):
                if image_file.endswith(".png") and image_file.startswith("image_"):
                    image_path = os.path.join(self.output_dir, image_file)
                    
                    try:
                        image = cv2.imread(image_path)
                        if image is None:
                            raise ValueError(f"Failed to load image: {image_path}")
                        
                        if "image_1" in image_file:
                            image = image[900:, :]
                        else:
                            image = image[200:, :]
                        
                        image = image[:-300, :]
                        image = image[:, 600:-200]
                        
                        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        denoised_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

                        kernel = np.ones((1, 2), np.uint8)
                        erosion = cv2.erode(denoised_img, kernel, iterations=1)

                        _, im_bw = cv2.threshold(erosion, 250, 220, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                        processed_image_path = os.path.join(self.output_dir, f"processed_{image_file}")
                        cv2.imwrite(processed_image_path, im_bw)
                        print(f"Processed and saved {processed_image_path}")
                    except Exception as e:
                        print(f"Error processing image '{image_file}': {str(e)}")
            print(f"All images in '{self.output_dir}' have been processed.")

        except Exception as e:
            raise RuntimeError(f"Error during image preprocessing: {str(e)}")

    def process_and_extract_text_from_images(self) -> Dict[str, List[str]]:
        """
        Detects and extracts text from processed images using EasyOCR.
        Saves a consolidated text file with extracted texts and returns a dictionary.
        """
        try:
            output_folder = os.path.join(self.output_dir, "processed_with_boxes")
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            extracted_texts: Dict[str, List[str]] = {}

            for image_file in os.listdir(self.output_dir):
                if image_file.startswith("processed_") and image_file.endswith(".png"):
                    image_path = os.path.join(self.output_dir, image_file)
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"Skipping invalid image: {image_path}")
                        continue

                    try:
                        merged_boxes = self.process_image_with_easyocr(image_path, output_folder)
                        document: List[str] = []
                        for (x, y, w, h) in merged_boxes:
                            roi = img[y:y + h, x:x + w]
                            result = self.reader.readtext(roi, detail=1)
                            box_text = " ".join([text for (_, text, confidence) in result if confidence > 0.2])
                            document.append(box_text)

                        extracted_texts[image_file] = document
                    except Exception as e:
                        print(f"Error extracting text from '{image_file}': {str(e)}")
            
            output_text_file = os.path.join(self.output_dir, "extracted_texts.txt")
            with open(output_text_file, "w", encoding="utf-8") as text_file:
                for image_name, texts in extracted_texts.items():
                    text_file.write(f"### {image_name} ###\n")
                    text_file.write("\n".join(texts) + "\n\n")

            print(f"Extracted texts saved to '{output_text_file}'")
            return extracted_texts
        
        except Exception as e:
            raise RuntimeError(f"Error during text extraction: {str(e)}")

    def process_image_with_easyocr(self, image_path: str, output_folder: str) -> List[tuple]:
        """
        Detects and merges bounding boxes around text in an image.
        Saves an annotated image with bounding boxes and returns the list of bounding boxes.
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Error: Could not load image from path {image_path}")
            
            # Perform OCR
            try:
                results = self.reader.readtext(img)
            except Exception as e:
                raise RuntimeError(f"OCR failed for image {image_path}: {str(e)}")

            horizontal_boxes: List[tuple] = []
            current_line_boxes: List[tuple] = []

            # Process OCR results
            for (bbox, text, confidence) in results:
                try:
                    if confidence > 0.2:
                        (top_left, top_right, bottom_right, bottom_left) = bbox
                        top_left = tuple(map(int, top_left))
                        bottom_right = tuple(map(int, bottom_right))

                        width = bottom_right[0] - top_left[0]
                        height = bottom_right[1] - top_left[1]

                        if width > height:
                            horizontal_boxes.append((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))
                except Exception as e:
                    print(f"Error processing bounding box {bbox}: {str(e)}")
            
            # Sort bounding boxes by their y-coordinates
            try:
                horizontal_boxes.sort(key=lambda box: box[1])
            except Exception as e:
                raise RuntimeError(f"Failed to sort bounding boxes: {str(e)}")

            # Merge bounding boxes
            try:
                merged_boxes: List[tuple] = []
                for box in horizontal_boxes:
                    if not current_line_boxes:
                        current_line_boxes.append(box)
                        continue

                    _, y1, _, y2 = box
                    _, prev_y1, _, prev_y2 = current_line_boxes[-1]

                    if abs(y1 - prev_y1) <= 10 or abs(y2 - prev_y2) <= 10:
                        current_line_boxes.append(box)
                    else:
                        min_x1 = min(b[0] for b in current_line_boxes)
                        min_y1 = min(b[1] for b in current_line_boxes)
                        max_x2 = max(b[2] for b in current_line_boxes)
                        max_y2 = max(b[3] for b in current_line_boxes)
                        merged_boxes.append((min_x1, min_y1, max_x2 - min_x1, max_y2 - min_y1))
                        current_line_boxes = [box]

                if current_line_boxes:
                    min_x1 = min(b[0] for b in current_line_boxes)
                    min_y1 = min(b[1] for b in current_line_boxes)
                    max_x2 = max(b[2] for b in current_line_boxes)
                    max_y2 = max(b[3] for b in current_line_boxes)
                    merged_boxes.append((min_x1, min_y1, max_x2 - min_x1, max_y2 - min_y1))
                    
            except Exception as e:
                raise RuntimeError(f"Error during merging bounding boxes: {str(e)}")

            # Draw bounding boxes on the image
            try:
                for (x, y, w, h) in merged_boxes:
                    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            except Exception as e:
                raise RuntimeError(f"Error drawing bounding boxes on the image: {str(e)}")

            # Save the annotated image
            try:
                output_image_path = os.path.join(output_folder, f"bounding_boxes_{os.path.basename(image_path)}")
                cv2.imwrite(output_image_path, img)
                print(f"Image with merged bounding boxes saved as {output_image_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to save the image with bounding boxes: {str(e)}")

            return merged_boxes

        except ValueError as ve:
            print(f"Value Error: {str(ve)}")
        except RuntimeError as re:
            print(f"Runtime Error: {str(re)}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")



if __name__ == "__main__":
    preprocessor = Preprocessing(pdf_file="extract/24000007.pdf", language="fr")
    pdf_images_path = preprocessor.pdf_to_images()  
    preprocessor.preprocess_images()  
    extracted_texts = preprocessor.process_and_extract_text_from_images()  
