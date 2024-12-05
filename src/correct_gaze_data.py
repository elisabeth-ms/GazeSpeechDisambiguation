import json
import os
import copy

# Function to load gaze data from a JSON file
def load_gaze_data(gaze_data_file):
    try:
        if not os.path.exists(gaze_data_file):
            print(f"File not found: {gaze_data_file}")
            return None
        with open(gaze_data_file, 'r') as f:
            gaze_data = json.load(f)
        return gaze_data
    except Exception as e:
        print(f"Error loading file {gaze_data_file}: {e}")
        return None



start_time_change = 1733321219.1598685 + 4.0

end_time_change = 1733321219.1598685 + 5.8



gaze_data_file = "/hri/localdisk/emende/testing_gaze/Elisabeth.json"

gaze_data = load_gaze_data(gaze_data_file)

user_raw_gaze_data = gaze_data["gaze_data"]

    
for gaze in user_raw_gaze_data:
    time = gaze["time"]
    if time >= start_time_change and time <= end_time_change:
        objects = gaze["objects"]
        print("Found gaze data within the time range: ", objects[0]["name"])
        # Correct the object name
        index_replacement = 0
        for i, obj in enumerate(objects):
            print(obj)
            if obj["name"] == "bottle_of_orange_juice":
                index_replacement = i
                obj["angle_diff"] += 5.0
                


        

gaze_data_file_corrected = "/hri/localdisk/emende/testing_gaze/Elisabeth_corrected.json"
# Save the corrected data to a new JSON file
with open(gaze_data_file_corrected, 'w') as file:
    json.dump(gaze_data, file, indent=4)

print(f"Corrected data saved to {gaze_data_file_corrected}")