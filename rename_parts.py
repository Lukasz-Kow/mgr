import os

DATASET_DIR = "Alzheimer_MRI_4_classes_dataset"

def rename_folders():
    base_path = os.path.abspath(DATASET_DIR)
    
    if not os.path.exists(base_path):
        print(f"Error: Directory {base_path} does not exist.")
        return

    for class_name in os.listdir(base_path):
        class_path = os.path.join(base_path, class_name)
        if os.path.isdir(class_path):
            for item in os.listdir(class_path):
                # Check for "X part" folders
                if " part" in item and os.path.isdir(os.path.join(class_path, item)):
                    # New name: Clean spaces and prepend class name
                    # "1st part" -> "Class_1st_part"
                    clean_item = item.replace(" ", "_")
                    new_name = f"{class_name}_{clean_item}"
                    
                    old_path = os.path.join(class_path, item)
                    new_path = os.path.join(class_path, new_name)
                    
                    print(f"Renaming: {old_path} -> {new_path}")
                    os.rename(old_path, new_path)

if __name__ == "__main__":
    rename_folders()
