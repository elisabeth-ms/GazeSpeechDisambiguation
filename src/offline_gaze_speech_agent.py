import sys
import os
import platform
import sys
import time
import py_LLM_handler
from matplotlib import pyplot as plt
sys.path.append(os.path.abspath("/hri/localdisk/emende/AttentiveSupport/src"))
from function_analyzer import FunctionAnalyzer


import json


if platform.system() == "Linux":
    sys.path.append("lib")
elif platform.system() == "Windows":
    sys.path.append("bin")

from pyAffaction import *
sys.path.append(os.path.abspath("/hri/localdisk/emende/AttentiveSupport/src/Smile/src/AffAction/python"))
import pyGaze

class MissingEnvironmentVariable(Exception):
    pass


if "OPENAI_API_KEY" not in os.environ:
    raise MissingEnvironmentVariable(
        "Please set an environment variable with your OPENAI_API_KEY. "
        "See https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety"
    )


SIM = None
print_emojis = True  
speech_directory_path = None
gpt_responses_path = None
filtered_gaze_data_directory_path = None
sceneTransformationDataPlayerEnabled = None
sceneTransformationDataRecorderEnabled = None


input_mode = "gaze_history_speech" # Options: "speech_only", "gaze_only", "gaze_history_speech", "synchronized_gaze_speech"
config_file = "gpt_gaze_speech_scene_config"

main_dir_path = '/home/elisabeth/data/interaction_recordings/'

# Initialize dialogue and interaction counters
dialogue_number = 1 # Adjust this based on which dialogue you want to load
interaction_number = 1  # Adjust this based on which interaction to start with


main_dir = 'interaction_recordings/'
main_dir_path = os.path.join('/home/elisabeth/data', main_dir)
excluded_objects = ['Hand_left_Elisabeth', 'Hand_right_Elisabeth']




# Function to load gaze data from a JSON file
def load_gaze_data(interaction_folder_path, user_name):
    gaze_data_file = os.path.join(interaction_folder_path, 'raw_gaze_data', f'{user_name}.json')
    with open(gaze_data_file, 'r') as f:
        gaze_data = json.load(f)
    return gaze_data

# Function to load speech data from a JSON file
def load_speech_data(interaction_folder_path, user_name):
    speech_data_file = os.path.join(interaction_folder_path, 'speech_data', f'{user_name}.json')
    with open(speech_data_file, 'r') as f:
        speech_data = json.load(f)
    return speech_data


time_taken = 0

# Loop through dialogues and interactions, load pre-recorded data
def run_offline_interactions(llm_handler, main_dir_path, dialogue_number, interaction_number, user_name="Elisabeth"):
    dialogue_folder_path = os.path.join(main_dir_path, f'dialogue{dialogue_number}')
    transformation_file = None
    # Loop through each interaction folder in the dialogue folder
    while True:
        interaction_folder_path = os.path.join(dialogue_folder_path, f'dialogue{dialogue_number}_{interaction_number}')
        
        
        # Check if the interaction folder exists
        if not os.path.exists(interaction_folder_path):
            print(f"No more interactions in dialogue {dialogue_number}.")
            break
        
        # Load pre-recorded gaze and speech data
        user_raw_gaze_data = load_gaze_data(interaction_folder_path, user_name)
        
        
        speech_data = load_speech_data(interaction_folder_path, user_name)
        
        if SIM.sceneTransformationDataPlayerEnabled:
            print("Scene transformation data player is enabled.")
            transformation_file = os.path.join(interaction_folder_path, 'transformations.json')
        
        # Extract relevant information from speech data
        speech_input = speech_data["transcript"]
        start_time = speech_data["listening_start_time"]

        gaze_history, objects_timestamps = pyGaze.compute_list_closest_objects_gaze_history(user_raw_gaze_data["gaze_data"], start_time, 15.0,8.0, 8.0, excluded_objects, 5.0, 0.5, 0.08)

        start_time = speech_data['listening_start_time']
        word_data = [(word_info['word'], word_info['start_time'], word_info['end_time']) for word_info in speech_data['words']]
        global time_taken
        
        print(objects_timestamps)
        new_objects_timestamps = []
        for segment in objects_timestamps:
            if segment[1]>3.0 and segment[2]<8.2:
              new_objects_timestamps.append(segment)
            print(segment[0])
            print(segment[1])
            print(segment[2])
        
        print("-----------------------------------------------")

        print(new_objects_timestamps)
        print("-----------------------------------------------")

        new_word_data = []
        for i, word in enumerate(word_data):
            if i>1:
              new_word_data.append((word[0], word[1]+1, word[2]+1))
        print(new_word_data)
        pyGaze.plot_multi_gaze_and_speech(new_objects_timestamps, new_word_data)
        # pyGaze.plot_angle_diff_over_time(user_raw_gaze_data["gaze_data"], start_time, start_time+10.0, '3D')
        
        plt.show()
        if input_mode == "speech_only":
            print(f"{llm_handler._user_speech_emojis if print_emojis else ''}{speech_input}")
            llm_handler.play_with_functions_synchronized(speech_input, person_name=user_name)
        
        elif input_mode == "gaze_only":
            print(f"{llm_handler._user_gaze_emojis if print_emojis else ''}{gaze_history}")
            llm_handler.play_with_functions_synchronized(gaze_history, person_name=user_name)
        elif input_mode == "gaze_history_speech":
            print(f"{llm_handler._user_speech_emojis if print_emojis else ''}{speech_input}")
            print(f"{llm_handler._user_gaze_emojis if print_emojis else ''}{gaze_history}")
            before_time = time.time()
            llm_handler.play_with_functions_gaze_history_speech(speech_input=speech_input, gaze_history=gaze_history, person_name=user_name)
            print("Time taken current call: ", time.time() - before_time)
            time_taken += time.time() - before_time
            print("Time taken so far: ", time_taken)
        elif input_mode == "synchronized_gaze_speech":
            input_data = pyGaze.merge_gaze_word_intervals(objects_timestamps, word_data)
            print(f"{llm_handler._user_emojis if print_emojis else ''}{input_data}")
            print(f"{llm_handler._user_speech_emojis if print_emojis else ''}{speech_input}")
            print(f"{llm_handler._user_gaze_emojis if print_emojis else ''}{objects_timestamps}")
            before_time = time.time()
            llm_handler.play_with_functions_synchronized(input=input_data, person_name=user_name)
            print("Time taken current call: ", time.time() - before_time)
            time_taken += time.time() - before_time
            print("Time taken so far: ", time_taken)
        else:
            print("Invalid input mode. Please select one of the following: 'speech_only', 'gaze_only', 'gaze_history_speech', 'synchronized_gaze_speech'")
        if transformation_file:
            print("Loading transformation data...")
            SIM.load_transformation_data_from_file(transformation_file)
            SIM.start_playback_transformation_data()
        # Move to the next interaction
        interaction_number += 1
        input("Press Enter to continue...")



def main():

    print_emojis = True
    person_name = "Elisabeth"
    llm_handler = py_LLM_handler.LLMHandler(config_module=config_file)
    global SIM
    SIM = llm_handler.get_simulation()
    SIM.run()
    
    run_offline_interactions(llm_handler, main_dir_path, dialogue_number, interaction_number, user_name="Elisabeth")

            
if __name__ == "__main__":
    main()
