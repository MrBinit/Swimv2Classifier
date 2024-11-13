import os

def delete_mp4_files(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith('.mp4'):
            os.remove(file_path)
            print(f"Deleted: {file_path}")

# Define the path to the folder
folder_path = '/home/binit/classifier/data/raw/real'
delete_mp4_files(folder_path)
