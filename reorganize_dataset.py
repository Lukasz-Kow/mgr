import os
import shutil
import math

# Configuration
DATASET_DIR = "Alzheimer_MRI_4_classes_dataset"
MAX_FILES_PER_FOLDER = 1000

def reorganize_folder(class_path):
    # Get all files in the directory (excluding subdirectories if any, though dataset implies flat structure per class)
    files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
    files.sort() # Sort to ensure deterministic order
    
    total_files = len(files)
    if total_files <= MAX_FILES_PER_FOLDER:
        print(f"Skipping {class_path}: {total_files} files (<= {MAX_FILES_PER_FOLDER})")
        return

    print(f"Processing {class_path}: {total_files} files found.")
    
    num_parts = math.ceil(total_files / MAX_FILES_PER_FOLDER)
    
    for i in range(num_parts):
        # Human readable part names with class prefix
        suffix = "th"
        if (i + 1) % 10 == 1 and (i + 1) % 100 != 11:
            suffix = "st"
        elif (i + 1) % 10 == 2 and (i + 1) % 100 != 12:
            suffix = "nd"
        elif (i + 1) % 10 == 3 and (i + 1) % 100 != 13:
            suffix = "rd"
            
        class_name = os.path.basename(class_path)
        part_folder_name = f"{class_name}_{i+1}{suffix}_part"
        part_folder_path = os.path.join(class_path, part_folder_name)
        
        os.makedirs(part_folder_path, exist_ok=True)
        
        # Determine slice of files for this part
        start_idx = i * MAX_FILES_PER_FOLDER
        end_idx = min((i + 1) * MAX_FILES_PER_FOLDER, total_files)
        files_to_move = files[start_idx:end_idx]
        
        print(f"Moving {len(files_to_move)} files to {part_folder_name}...")
        
        for filename in files_to_move:
            src = os.path.join(class_path, filename)
            dst = os.path.join(part_folder_path, filename)
            shutil.move(src, dst)
            
    print(f"Finished processing {class_path}")

def main():
    base_path = os.path.abspath(DATASET_DIR)
    
    if not os.path.exists(base_path):
        print(f"Error: Directory {base_path} does not exist.")
        return

    # Iterate over each class folder
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            reorganize_folder(item_path)
            
if __name__ == "__main__":
    main()
