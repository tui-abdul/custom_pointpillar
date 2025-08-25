import json
import os

# Map SUSTech object types to KITTI
TYPE_MAP = {
    "car": "car",
    "motorcycle": "motorcycle"  # Or "cyclist" if preferred
}

def sustech_to_kitti_all_angles(json_data):
    lines = []
    for obj in json_data['objs']:
        obj_type = TYPE_MAP.get(obj['obj_type'], 'Misc')
        psr = obj['psr']
        pos = psr['position']
        rot = psr['rotation']
        scale = psr['scale']
        
        # KITTI-like format with all three rotation values
        line = f"{obj_type} 0 0 0 0 0 0 0 {scale['z']} {scale['y']} {scale['x']} {pos['x']} {pos['y']} {pos['z']} {rot['x']} {rot['y']} {rot['z']}"
        lines.append(line)
    return lines

# Folder containing JSON files
json_folder = "/mnt/sda/Abdul_Haq/dataset_first_21/20250814_3d&2d/sunny/sequence_1/lidar_point_cloud_0"  # adjust if needed

# Folder to save output txt files
output_folder = "../../dataset_local/intersection_data_21/sunny/sequence_1/label_2"
os.makedirs(output_folder, exist_ok=True)  # create folder if it doesn't exist

# List all JSON files in folder
json_files = sorted(f for f in os.listdir(json_folder) if f.endswith(".json"))

for json_file in json_files:
    # Compute output filename: "000.json" -> "1.txt"
    base_number = int(json_file.split(".")[0])
    txt_file = f"{base_number + 1}.txt"
    
    # Load JSON
    with open(os.path.join(json_folder, json_file), "r") as f:
        data = json.load(f)
    
    # Convert to KITTI-like format
    kitti_lines = sustech_to_kitti_all_angles(data)
    
    # Write to text file in the output folder
    with open(os.path.join(output_folder, txt_file), "w") as f:
        for line in kitti_lines:
            f.write(line + "\n")
    
    print(f"Converted {json_file} -> {os.path.join(output_folder, txt_file)}")

print("All files converted and saved to", output_folder)
