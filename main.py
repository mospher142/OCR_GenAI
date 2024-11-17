import os
from pathlib import Path
from pipeline import Pipeline


def list_pdf_files(folder: str):
    """
    Lists all PDF files in a given folder.

    Args:
        folder (str): Path to the folder.

    Returns:
        list: List of PDF file paths.
    """
    return [str(file) for file in Path(folder).glob("*.pdf")]


def main():
    
    # User input: Folder or single file
    choice = input("Do you want to process a folder or a specific file? (Enter 'folder' or 'file'): ").strip().lower()
    
    if choice == 'folder':
        folder = input("Enter the path to the folder: ").strip()
        if not os.path.isdir(folder):
            print(f"Error: '{folder}' is not a valid folder.")
            return

        pdf_files = list_pdf_files(folder)
        if not pdf_files:
            print("No PDF files found in the specified folder.")
            return

        print("\nThe following PDF files were found:")
        for idx, file in enumerate(pdf_files, start=1):
            print(f"{idx}. {file}")

        print("\nOptions:")
        print("0. Process all files")
        print("1. Select specific files")
        option = input("Enter your choice (0 or 1): ").strip()

        if option == '0':  # Process all files
            files_to_process = pdf_files
        elif option == '1':  # Select specific files
            selections = input("Enter the numbers of the files you want to process (comma-separated): ").strip()
            try:
                indices = [int(i) - 1 for i in selections.split(",")]
                files_to_process = [pdf_files[i] for i in indices]
            except (ValueError, IndexError):
                print("Invalid selection.")
                return
        else:
            print("Invalid option.")
            return

    elif choice == 'file':
        file_path = input("Enter the path to the PDF file: ").strip()
        if not file_path.endswith(".pdf") or not os.path.isfile(file_path):
            print(f"Error: '{file_path}' is not a valid PDF file.")
            return
        files_to_process = [file_path]

    else:
        print("Invalid choice. Please enter 'folder' or 'file'.")
        return

    # Process selected files
    for file_path in files_to_process:
        print(f"\nProcessing file: {file_path}")
        try:
            pipeline = Pipeline(pdf_file=file_path)
            result_json = pipeline.process()
            print(f"Generated JSON file: {result_json}")
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
