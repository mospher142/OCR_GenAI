import os
import cv2
import fitz
import easyocr
import numpy as np
from typing import Dict, List


class Preprocessing:
    def __init__(self, pdf_file: str, language: str = 'en') -> None:
        self.pdf_file: str = pdf_file
        self.language: str = language
        self.pdf_name: str = os.path.splitext(os.path.basename(pdf_file))[0]
        self.output_dir: str = f"{self.pdf_name}"
        self.reader = easyocr.Reader([language])  # Initialize EasyOCR reader with specified language
    
    def pdf_to_images(self, zoom: int = 6) -> str:
        """
        Converts the PDF to images, saving them in a directory named after the PDF.
        Returns the directory path containing the images.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        existing_files = [f for f in os.listdir(self.output_dir) if f.endswith(".png")]
        if existing_files:
            print(f"Images already exist in '{self.output_dir}'.")
            return self.output_dir
        
        doc = fitz.open(self.pdf_file)
        page_count = doc.page_count  # Store the page count before closing the document
        print(f"Converting PDF '{self.pdf_file}' with {page_count} pages into images...")

        mat = fitz.Matrix(zoom, zoom)
        for i in range(page_count):
            output_path = os.path.join(self.output_dir, f"image_{i+1}.png")
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat)
            pix.save(output_path)
            print(f"Saved {output_path}")

        doc.close()  # Close the document after processing all pages
        print(f"Converted {page_count} pages to images in '{self.output_dir}'.")
        return self.output_dir

    def preprocess_images(self) -> None:
        """
        Preprocesses the images extracted from the PDF by cropping and applying filters.
        Saves the processed images in the same directory with a 'processed_' prefix.
        """
        for image_file in os.listdir(self.output_dir):
            if image_file.endswith(".png") and image_file.startswith("image_"):
                image_path = os.path.join(self.output_dir, image_file)
                image = cv2.imread(image_path)

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

        print(f"All images in '{self.output_dir}' have been processed.")

    def process_and_extract_text_from_images(self) -> Dict[str, List[str]]:
        """
        Detects and extracts text from processed images using EasyOCR.
        Saves a consolidated text file with extracted texts and returns a dictionary.
        """
        output_folder = os.path.join(self.output_dir, "processed_with_boxes")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        extracted_texts: Dict[str, List[str]] = {}

        for image_file in os.listdir(self.output_dir):
            if image_file.startswith("processed_") and image_file.endswith(".png"):
                image_path = os.path.join(self.output_dir, image_file)
                img = cv2.imread(image_path)

                merged_boxes = self.process_image_with_easyocr(image_path, output_folder)

                document: List[str] = []
                for (x, y, w, h) in merged_boxes:
                    roi = img[y:y + h, x:x + w]
                    result = self.reader.readtext(roi, detail=1)
                    box_text = " ".join([text for (_, text, confidence) in result if confidence > 0.2])
                    document.append(box_text)

                extracted_texts[image_file] = document

        output_text_file = os.path.join(self.output_dir, "extracted_texts.txt")
        with open(output_text_file, "w", encoding="utf-8") as text_file:
            for image_name, texts in extracted_texts.items():
                text_file.write(f"### {image_name} ###\n")
                text_file.write("\n".join(texts) + "\n\n")

        print(f"Extracted texts saved to '{output_text_file}'")
        return extracted_texts

    def process_image_with_easyocr(self, image_path: str, output_folder: str) -> List[tuple]:
        """
        Detects and merges bounding boxes around text in an image.
        Saves an annotated image with bounding boxes and returns the list of bounding boxes.
        """
        img = cv2.imread(image_path)
        results = self.reader.readtext(img)

        horizontal_boxes: List[tuple] = []
        current_line_boxes: List[tuple] = []

        for (bbox, text, confidence) in results:
            if confidence > 0.2:
                (top_left, top_right, bottom_right, bottom_left) = bbox
                top_left = tuple(map(int, top_left))
                bottom_right = tuple(map(int, bottom_right))

                width = bottom_right[0] - top_left[0]
                height = bottom_right[1] - top_left[1]

                if width > height:
                    horizontal_boxes.append((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))

        horizontal_boxes.sort(key=lambda box: box[1])

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

        for (x, y, w, h) in merged_boxes:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        output_image_path = os.path.join(output_folder, f"bounding_boxes_{os.path.basename(image_path)}")
        cv2.imwrite(output_image_path, img)
        print(f"Image with merged bounding boxes saved as {output_image_path}")
        return merged_boxes


if __name__ == "__main__":
    preprocessor = Preprocessing(pdf_file="extract/24000010.pdf", language="fr")
    pdf_images_path = preprocessor.pdf_to_images()  
    preprocessor.preprocess_images()  
    extracted_texts = preprocessor.process_and_extract_text_from_images()  
