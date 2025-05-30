"""
This script analyzes real-time gaze data to identify the object in the scene that the user is focusing on.
It uses thresholds for gaze velocity, angle differences, and gaze duration to determine focus and excludes irrelevant objects.
The script speaks the name of a random object in the scene and checks if the user focuses on the object for a sufficient duration.
If the user correctly focuses on the announced object, it provides positive feedback.
"""

import sys
import time
import numpy as np
import platform
import sys
import time
import random
import json
import os
if platform.system() == "Linux":
    sys.path.append("lib")
elif platform.system() == "Windows":
    sys.path.append("bin")

from pyAffaction import *

def speak(SIM, text):
    """
    Sends a command to the simulation to speak the provided text.

    Parameters:
        SIM (LlmSim): Simulation object to execute the speak command.
        text (str): The text to be spoken by the simulation.

    Returns:
        None
    """
    
    SIM.execute(f"speak {text}")

class GazeDataManager:
    """
    Manages gaze data retrieval from the simulation.

    Methods:
        get_raw_gaze_data: Retrieves raw gaze data for all users.
    """
    
    def __init__(self, SIM):
        """
        Initializes the GazeDataManager with a simulation instance.

        Parameters:
            SIM (LlmSim): Simulation object to interact with.
        """
        
        self.SIM = SIM

    def get_raw_gaze_data(self):
        """
        Retrieves raw gaze data for all users from the GazeComponent.

        Returns:
            list: Raw gaze data for all users.
        """
        
        return self.SIM.get_gaze_data()

def save_to_json(data, filename="gaze_data.json"):
    """
    Saves the provided data to a JSON file.

    Parameters:
        data (dict): Data to be saved.
        filename (str): Name of the file to save the data.

    Returns:
        None
    """
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def create_test_folder(main_dir_path, test_number):
    """
    Creates a test folder for storing data.

    Parameters:
        main_dir_path (str): Path to the main directory where test folders are created.
        test_number (int): Test folder number.

    Returns:
        str: Path to the created test folder.
    """
    test_folder = os.path.join(main_dir_path, f'test{test_number}')
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
        return test_folder
    else:
        return None

SIM = None
random_object_timings = {}
filtered_gaze_entries = []

def main():
    """
    Main function to analyze real-time gaze data and provide feedback based on the user's focus.

    Parameters:
        None

    Returns:
        None
    """
    # Set up the simulation 
    setLogLevel(-1)
    global SIM
    SIM = LlmSim()
    SIM.noTextGui = True
    SIM.unittest = False
    SIM.speedUp = 1
    SIM.noLimits = False
    SIM.verbose = False
    # SIM.usersGazeComponentEnabled = True
    SIM.addLandmarkZmq(camera_name="camera_0", withArucoTracking=True, withSkeletonTracking=True)
    SIM.xmlFileName = "g_example_breakfast_scenario.xml" # Change the file name to the desired scenario
    # SIM.playTransformations = True
    # SIM.recordTransformations = True
    SIM.addTTS("native")
    SIM.init(True)
    SIM.run()
    
    gaze_manager = GazeDataManager(SIM=SIM)
    sleep_time = 0.01
    gaze_start_time = None
    current_object = None
    
    
    # Thresholds and parameters for gaze analysis
    gaze_velocity_threshold = 15
    angle_diff_threshold = 8 
    angle_diff_xz_threshold = 8
    gaze_duration_threshold = 0.1
    already_said = False
    
    excluded_objects = ['table','hand_left_robot', 'hand_right_robot', 'Hand_left_Elisabeth', 'Hand_right_Elisabeth']
    
    user_name = "Elisabeth"
    
    
    # Query the objects from the scene
    result = SIM.get_objects()
    if not result:
        return "No objects were observed."
    
    objects = [obj for obj in result["objects"] if obj not in excluded_objects]
    
    
    random_object = None
    next_object = True
    aux = None
    start_object_time = None
    end_object_time = None
    test_number = 0
    test_folder_path = None
    main_dir_path = "/hri/elisabeth/data/testGaze" # Change the path to the desired directory
    last_gazed_time = None
    while test_folder_path is None:
        test_number += 1
        test_folder_path = create_test_folder(main_dir_path, test_number)

    try:
        while True:
            time.sleep(sleep_time)
            if(next_object):
                if last_gazed_time is None:
                    last_gazed_time = getWallclockTime()
                aux = random.choice(objects)
                while aux == random_object:
                    aux = random.choice(objects)
                random_object = aux
                print("Random object: ", random_object)
                start_object_time = getWallclockTime()
                speak(SIM, random_object)
                time.sleep(0.1)
                next_object = False
            
            
            all_users_gaze_data = gaze_manager.get_raw_gaze_data()
            
            for user_gaze_data in all_users_gaze_data:
                if user_gaze_data["agent_name"] != user_name:
                    print("Skipping gaze data for user: ", user_gaze_data["agent_name"])
                gaze_data = user_gaze_data["gaze_data"]
                for entry in gaze_data:
                    if entry['time'] > last_gazed_time:
                        # Collect relevant data

                        filtered_entry = {
                            "time": entry['time'],
                            "gaze_velocity": entry['gaze_velocity'],
                            "objects": entry['objects'],
                        }
                        filtered_gaze_entries.append(filtered_entry)
                        # If gaze veloity is high, exclude the object (unstable gaze)
                        if entry['gaze_velocity'] > gaze_velocity_threshold:
                            current_object = None
                            continue
                        object_name = None  # Default
                        obj = None
                        for obj in entry['objects']:
                            if obj['angle_diff'] < angle_diff_threshold and obj['angle_diffXZ'] < angle_diff_xz_threshold:
                                object_name = obj['name']
                                break
                        # Initialize the first object and gaze start time
                        if current_object is None and gaze_start_time is None:
                            current_object = object_name
                            gaze_start_time = entry['time']
                            continue  # Skip to the next iteration since we just initialized
                        
                        if current_object is not None:
                            current_time = entry['time']
                            gaze_duration = current_time - gaze_start_time
                            if gaze_duration > gaze_duration_threshold and current_object == random_object:
                                speak(SIM, "CORRECT")
                                end_object_time = getWallclockTime()
                                if random_object not in random_object_timings:
                                    random_object_timings[random_object] = []
                                
                                random_object_timings[random_object].append({"start_time": start_object_time, "end_time": end_object_time})
                                save_to_json(random_object_timings, "random_object_timings.json")
                                time.sleep(0.1)
                                print("Object: ", current_object, " Duration: ", gaze_duration, "angle diff: ", obj['angle_diff'], "angle diff xz: ", obj['angle_diffXZ'])
                                next_object = True
                                continue
                        
                        if object_name != current_object:
                            current_time = entry['time']
                            current_object = object_name
                            gaze_start_time = current_time
                if gaze_data:
                    last_gazed_time = gaze_data[-1]['time']
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")

        random_object_timings_file = os.path.join(test_folder_path, 'random_object_timings.json')
        gaze_data_file = os.path.join(test_folder_path, 'gaze_data.json')
        save_to_json(random_object_timings, random_object_timings_file)
        save_to_json(filtered_gaze_entries, gaze_data_file)
        sys.exit()

if __name__ == "__main__":
    main()
