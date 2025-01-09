import os
import json



def create_interaction_recordings(main_dir_path):
    """
    Creates the main directory for interaction recordings if it doesn't exist.

    Parameters:
        main_dir_path (str): Path to the main directory.
    Returns:
        str: Absolute path to the main directory.
    """
    if not os.path.exists(main_dir_path):
        os.makedirs(main_dir_path)
    return os.path.abspath(main_dir_path)


def create_dialogue_folder(main_dir_path, dialogue_number):
    """
    Creates a folder for a specific dialogue.

    Parameters:
        main_dir_path (str): Path to the main directory.
        dialogue_number (int): Number identifying the dialogue.

    Returns:
        str or None: Path to the dialogue folder if created, otherwise None.
    """
    dialogue_folder = os.path.join(main_dir_path, f'dialogue{dialogue_number}')
    if not os.path.exists(dialogue_folder):
        os.makedirs(dialogue_folder)
        return dialogue_folder
    else:
        return None


def create_interaction_folder(dialogue_dir_path,dialogue_number, interaction_number):
    """
    Creates a folder for a specific interaction within a dialogue.

    Parameters:
        dialogue_dir_path (str): Path to the dialogue folder.
        dialogue_number (int): Number identifying the dialogue.
        interaction_number (int): Number identifying the interaction.

    Returns:
        str or None: Path to the interaction folder if created, otherwise None.
    """
    print("dialogue_dir_path: ", dialogue_dir_path)
    print("dialogue_number: ", dialogue_number)
    print("interaction_number: ", interaction_number)
    interaction_dir_path = os.path.join(dialogue_dir_path, f'dialogue{dialogue_number}_{interaction_number}')
    print("interaction_dir_path: ", interaction_dir_path)
    if not os.path.exists(interaction_dir_path):
        os.makedirs(interaction_dir_path)
        return interaction_dir_path
    else:
        return None


def save_raw_gaze_data(interaction_folder_path, all_users_raw_gaze_data):
    """
    Saves raw gaze data for all users into JSON files.

    Parameters:
        interaction_folder_path (str): Path to the interaction folder.
        all_users_raw_gaze_data (list): List of raw gaze data dictionaries for each user.

    Returns:
        None
    """
    raw_gaze_data_folder = os.path.join(interaction_folder_path, 'raw_gaze_data')

    os.makedirs(raw_gaze_data_folder, exist_ok=True)

    for user_raw_gaze_data in all_users_raw_gaze_data:
        user_name = user_raw_gaze_data["agent_name"]
        raw_gaze_data_file = os.path.join(raw_gaze_data_folder, f'{user_name}.json')
        with open(raw_gaze_data_file, 'w') as f:
            json.dump(user_raw_gaze_data, f, indent=4)
    
            
def save_speech_data(interaction_folder_path,user_name, press_s_time, press_f_time, transcript, word_data):
    
    """
    Saves speech data for a user into a JSON file.

    Parameters:
        interaction_folder_path (str): Path to the interaction folder.
        user_name (str): Name of the user.
        press_s_time (float): Listening start time.
        press_f_time (float): Listening end time.
        transcript (str): Full speech transcript.
        word_data (list): List of words with start and end times.

    Returns:
        None
    """
    speech_data_folder = os.path.join(interaction_folder_path, 'speech_data')
    os.makedirs(speech_data_folder, exist_ok=True)

    # Create a structured dictionary with all the necessary data
    data = {
        "listening_start_time": press_s_time,
        "listening_end_time": press_f_time,
        "transcript": transcript,
        "words": [
            {
                "word": word,
                "start_time": start_time,
                "end_time": end_time
            } for word, start_time, end_time in word_data
        ]
    }
    
    speech_data_file = os.path.join(speech_data_folder, f'{user_name}.json')

    # Save the data to a JSON file
    with open(speech_data_file, 'w') as f:
        json.dump(data, f, indent=4)
        
        
def save_transformations_data_to_json(directory, file_name, json_transformations):
    """
    Saves transformation data into a JSON file.

    Parameters:
        directory (str): Directory where the file will be saved.
        file_name (str): Name of the JSON file.
        json_transformations (dict): Transformation data to save.

    Returns:
        None
    """
    
    with open(directory+"/"+file_name, 'w') as f:
        json.dump(json_transformations, f, indent=4)

def save_config_to_json(file_path, gaze_velocity_threshold, angle_diff_threshold, angle_diff_xz_threshold, 
    excluded_objects, off_target_velocity_threshold, off_target_duration_threshold, minimum_fixation_duration):
    """
    Saves configuration parameters into a JSON file.

    Parameters:
        file_path (str): Path to the JSON file.
        gaze_velocity_threshold (float): Threshold for gaze velocity.
        angle_diff_threshold (float): Threshold for angle difference in 3D.
        angle_diff_xz_threshold (float): Threshold for angle difference in XZ plane.
        excluded_objects (list): List of objects to exclude.
        off_target_velocity_threshold (float): Threshold for off-target gaze velocity.
        off_target_duration_threshold (float): Minimum duration for off-target gaze.
        minimum_fixation_duration (float): Minimum duration to consider a gaze fixation.

    Returns:
        None
    """
    # Create a dictionary with the provided parameters
    config = {
        "gaze_velocity_threshold": gaze_velocity_threshold,
        "angle_diff_threshold": angle_diff_threshold,
        "angle_diff_xz_threshold": angle_diff_xz_threshold,
        "excluded_objects": excluded_objects,
        "off_target_velocity_threshold": off_target_velocity_threshold,
        "off_target_duration_threshold": off_target_duration_threshold,
        "minimum_fixation_duration": minimum_fixation_duration
    }
    
    # Save the configuration to a JSON file
    try:
        with open(file_path, 'w') as json_file:
            json.dump(config, json_file, indent=4)
        print(f"Configuration saved to {file_path}")
    except Exception as e:
        print(f"Error saving configuration: {e}")



def chat_message_to_dict(message):
    """
    Converts a ChatCompletionMessage object into a JSON-serializable dictionary.

    Parameters:
        message (dict): Message object to convert.

    Returns:
        dict: JSON-serializable dictionary representation of the message.
    """
    if isinstance(message, dict):
        return message
    return {
        "role": message.role,
        "content": message.content,
        "function_call": message.function_call,
        "tool_calls": [
            {
                "id": call.id,
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments
                },
                "type": call.type
            } for call in message.tool_calls
        ] if message.tool_calls else None,
        "refusal": message.refusal
    }

def save_interaction_data_to_json(directory,file_name, responses):
    """
    Saves interaction data (speech input, gaze input, and GPT function responses) into a JSON file.

    Parameters:
        directory (str): Directory where the file will be saved.
        file_name (str): Name of the JSON file.
        responses (list): List of GPT responses to save.

    Returns:
        None
    """
    # Convert responses into JSON-serializable format
    serializable_responses = [chat_message_to_dict(response) for response in responses]

    # Create the structured data dictionary
    data = {
        "interaction_responses": serializable_responses
    }


    # Save the data to a JSON file
    with open(directory+"/"+file_name, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Interaction data saved to {file_name}")
