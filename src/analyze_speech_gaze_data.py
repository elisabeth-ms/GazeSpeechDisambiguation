import os
import json
import matplotlib.pyplot as plt
import random
import platform
import sys
import getch

if platform.system() == "Linux":
    sys.path.append("lib")
elif platform.system() == "Windows":
    sys.path.append("bin")

from pyAffaction import *
import pyGaze
setLogLevel(-1)

sim = LlmSim()
sim.noTextGui = True
sim.unittest = False
sim.speedUp = 1
sim.noLimits = False
sim.verbose = False
sim.playTransformations = True
sim.xmlFileName = "g_example_cola_orange_juice_two_glasses_bowl_ice.xml"
sim.init(True)
sim.run()

# Cross-platform getch function to capture key presses
def getch():
    """Get a single character from standard input without echo."""
    import sys
    if sys.platform.startswith('win'):
        import msvcrt
        return msvcrt.getch().decode()
    else:
        import tty
        import termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

# Function to find the folder of a specific test
def find_test_folder(base_directory, test_number):
    test_folder_name = f'test{test_number}'
    test_folder_path = os.path.join(base_directory, test_folder_name)

    if os.path.exists(test_folder_path):
        return test_folder_path
    else:
        print(f"Test folder '{test_folder_name}' not found.")
        return None

# Function to read JSON data from a file
def read_json_file(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def get_adjusted_threshold(base_threshold, distance):
    """
    Calculate the adjusted threshold based on the distance to the object.
    The minimum distance should be 0.3 meters. For distances below this, the base threshold is used.
    
    :param base_threshold: The base angle threshold for close objects (at 0.3 meters).
    :param distance: The distance from the user to the object.
    :return: The adjusted angle threshold.
    """
    if distance < 0.6:
        return base_threshold  # Use the base threshold for distances below 0.3 meters
    adjustment_factor = 0.6 / distance  # Inverse relationship for distances greater than 0.3 meters
    return base_threshold * adjustment_factor


def filter_gaze_history_closset_object(gaze_history, excluded_objects):
    """
    Filters the gaze history by excluding the robot's hands ('hand_left_robot' and 'hand_right_robot').

    :param gaze_history: A list of tuples representing the gaze history [(object_name, duration_in_seconds), ...]
    :return: A filtered gaze history list excluding the robot's hands.
    """
    filtered_history = [
        (obj, duration) for obj, duration in gaze_history 
        if obj not in excluded_objects
    ]
    return filtered_history

def json_to_words_data(speech_data):
    words_data = []
    for word in speech_data['words']:
        words_data.append((word['word'], float(word['start_time']), float(word['end_time'])))
    print(words_data)
    return words_data


# Function to process gaze and speech data for a specific test
def process_test_data(base_directory, test_number):
    # Find the test folder
    test_folder = find_test_folder(base_directory, test_number)
    if not test_folder:
        return

    # Define directories for gaze and speech data
    gaze_folder = os.path.join(test_folder, 'gaze')
    speech_folder = os.path.join(test_folder, 'speech')
    transformations_folder = os.path.join(test_folder, 'transformations')

    if not os.path.exists(gaze_folder) or not os.path.exists(speech_folder):
        print("Gaze or speech folder not found in the test folder.")
        return

    # Get all JSON files in the gaze and speech folders
    gaze_files = sorted(os.listdir(gaze_folder))
    speech_files = sorted(os.listdir(speech_folder))
    transformations_files = sorted(os.listdir(transformations_folder))

    # Filter JSON files
    gaze_files = [f for f in gaze_files if f.endswith('.json')]
    speech_files = [f for f in speech_files if f.endswith('.json')]
    transformations_files = [f for f in transformations_files if f.endswith('.json')]

    # Process corresponding gaze and speech files
    for gaze_file, speech_file, transformations_file in zip(gaze_files, speech_files, transformations_files):
        if gaze_file != speech_file or gaze_file != transformations_file:
            print(f"Mismatch between gaze, speech, transformations file names: {gaze_file}, {speech_file} and {transformations_file}")
            continue
        
        print(f"Processing: {gaze_file}")

        # Read gaze and speech data
        gaze_data = read_json_file(os.path.join(gaze_folder, gaze_file))
        speech_data = read_json_file(os.path.join(speech_folder, speech_file))
        sim.load_transformation_data_from_file(os.path.join(transformations_folder, transformations_file))

        


        start_listening_time = speech_data['listening_start_time']
        end_listening_time = speech_data['listening_end_time']
        excluded_objects = ['hand_left_robot', 'hand_right_robot']


        gaze_history, objects_timestamps = pyGaze.compute_gaze_history_closest_object(gaze_data, start_listening_time, 30.0,12.0, 12.0, excluded_objects, 5.0, 0.5)
        print("Gaze History:", gaze_history)
        # print("Filtered Gaze History:", filtered_gaze_history)

        gaze_history = pyGaze.compute_multi_object_gaze_history(gaze_data, start_listening_time, 60.0, 20.0)
        filtered_gaze_history = pyGaze.filter_multi_object_gaze_history(gaze_history, excluded_objects)
        print("Multi-Object Gaze History:", gaze_history)
        print("Filtered Multi-Object Gaze History:", filtered_gaze_history)
        
        print("speech_data: ", speech_data['transcript'])
        # Extract relevant information and plot the data
        words_data = json_to_words_data(speech_data)
        pyGaze.plot_gaze_and_speech(objects_timestamps, words_data)
        pyGaze.plot_angle_diff_over_time(gaze_data, start_listening_time, end_listening_time, '3D')
        pyGaze.plot_angle_diff_over_time(gaze_data, start_listening_time, end_listening_time, 'XY')
        pyGaze.plot_angle_diff_over_time(gaze_data, start_listening_time, end_listening_time, 'XZ')
        pyGaze.plot_gaze_velocity_over_time(gaze_data, start_listening_time, end_listening_time)

        sim.start_playback_transformation_data()
        plt.show()

        replay = True
        while replay:
            sim.start_playback_transformation_data()
            # if 'c' key is pressed, we continue to the next file
            key = getch()
            if key == 'c':
                replay = False


# Function to assign a random color
def get_random_color():
    return (random.random(), random.random(), random.random())


# Main function to run the analysis
def main():
    base_directory = "../../inputData"  # Example: "../../inputData"
    test_number = input("Enter the test number to analyze: ")

    try:
        test_number = int(test_number)
        process_test_data(base_directory, test_number)
    except ValueError:
        print("Invalid test number. Please enter a valid integer.")

if __name__ == "__main__":
    main()
