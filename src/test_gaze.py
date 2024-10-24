# This script analyzes real-time gaze data to determine which object in the scene the user is focusing on.
# It uses thresholds for gaze velocity, angle differences, and gaze duration to filter out unstable or unfocused gazes.
# When the user's gaze remains on an object for a specified duration, the object is announced via speech output.
# Objects like robot hands are excluded from the analysis to avoid irrelevant detections.
# The script continuously updates and speaks the name of the focused object as the user interacts with the scene.


import sys
import time
import numpy as np
import platform
import sys
import time

if platform.system() == "Linux":
    sys.path.append("lib")
elif platform.system() == "Windows":
    sys.path.append("bin")

from pyAffaction import *

def speak(SIM, text):
    SIM.execute(f"speak {text}")

class GazeDataManager:
    def __init__(self, SIM):
        self.SIM = SIM

    def get_raw_gaze_data(self):
        return self.SIM.get_gaze_data()


SIM = None

def main():
    # Set up the simulation 
    setLogLevel(-1)
    global SIM
    SIM = LlmSim()
    SIM.noTextGui = True
    SIM.unittest = False
    SIM.speedUp = 1
    SIM.noLimits = False
    SIM.verbose = False
    SIM.maxGazeAngleDiff = 120.0
    SIM.saveGazeData = True
    SIM.xmlFileName = "g_example_cola_orange_juice_two_glasses_bowl_ice.xml"
    SIM.playTransformations = True
    SIM.recordTransformations = True
    SIM.init(True)
    SIM.addTTS("native")
    camera_name = "camera_0" 
    SIM.addLandmarkZmq()
    SIM.run()
    
    gaze_manager = GazeDataManager(SIM=SIM)
    sleep_time = 0.2
    gaze_start_time = None
    current_object = None
    
    gaze_velocity_threshold = 20
    angle_diff_threshold = 10 
    angle_diff_xz_threshold = 10
    gaze_duration_threshold = 0.5
    already_said = False
    
    excluded_objects = ['hand_left_robot', 'hand_right_robot']
    # Keep the main thread alive
    try:
        while True:
            time.sleep(sleep_time)
            gaze_data = gaze_manager.get_raw_gaze_data()
            print(gaze)
            for entry in gaze_data:
                if entry['time'] >= gaze_data[-1]['time'] - sleep_time:
                    # If gaze veloity is high, exclude the object (unstable gaze)
                    if entry['gaze_velocity'] > gaze_velocity_threshold:
                        current_object = None
                        continue
                    object_name = None  # Default
                    for obj in entry['objects']:
                        if obj['name'] in excluded_objects:
                            continue
                        if obj['angleDiff'] < angle_diff_threshold and obj['angleDiffXZ'] < angle_diff_xz_threshold:
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
                        if gaze_duration > gaze_duration_threshold and not already_said:
                            speak(SIM, current_object)
                            print("Object: ", current_object, " Duration: ", gaze_duration)
                            already_said = True
                            continue
                    
                    if object_name != current_object:
                        current_time = entry['time']
                        current_object = object_name
                        gaze_start_time = current_time
                        already_said = False

            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit()

if __name__ == "__main__":
    main()
