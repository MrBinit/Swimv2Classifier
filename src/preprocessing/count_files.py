import os

def count_files_in_folder(folder_path):
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        print("The specified folder path does not exist.")
        return 0

    # Count files in the folder
    file_count = sum([1 for item in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, item))])
    return file_count

# Example usage
folder_path = "/home/binit/classifier/data/split_data/train/AI"  # Replace with your folder path
print(f"Number of files in the folder: {count_files_in_folder(folder_path)}")



# real: 6250
# fake: 6800