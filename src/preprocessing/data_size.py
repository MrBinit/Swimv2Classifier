import os

def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
    return total_size

def format_size(size_in_bytes):
    # Convert bytes to KB, MB, GB, etc.
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024

# Define the path to the folder
folder_path = '/home/binit/classifier/data/raw'
folder_size = get_folder_size(folder_path)
print(f"Size of '{folder_path}': {format_size(folder_size)}")
