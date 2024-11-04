import os
import json




# Create the main folder for interaction recordings and return its path
def create_interaction_recordings(main_dir_path):
    if not os.path.exists(main_dir_path):
        os.makedirs(main_dir_path)
    return os.path.abspath(main_dir_path)


# Create a dialogue folder 
def create_dialogue_folder(main_dir_path, dialogue_number):
    dialogue_folder = os.path.join(main_dir_path, f'dialogue{dialogue_number}')
    if not os.path.exists(dialogue_folder):
        os.makedirs(dialogue_folder)
        return dialogue_folder
    else:
        return None

def create_interaction_folder(dialogue_dir_path,dialogue_number, interaction_number):
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

# Create raw_gaze_data and speech_data folders with files per user
def save_raw_gaze_data(interaction_folder_path, all_users_raw_gaze_data):
    raw_gaze_data_folder = os.path.join(interaction_folder_path, 'raw_gaze_data')

    os.makedirs(raw_gaze_data_folder, exist_ok=True)

    for user_raw_gaze_data in all_users_raw_gaze_data:
        user_name = user_raw_gaze_data["agent_name"]
        raw_gaze_data_file = os.path.join(raw_gaze_data_folder, f'{user_name}.json')
        with open(raw_gaze_data_file, 'w') as f:
            json.dump(user_raw_gaze_data, f, indent=4)
            
def save_speech_data(interaction_folder_path,user_name, press_s_time, press_f_time, transcript, word_data):
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
    # Save the raw gaze data to a JSON file
    with open(directory+"/"+file_name, 'w') as f:
        json.dump(json_transformations, f, indent=4)




def chat_message_to_dict(message):
    """
    Converts a ChatCompletionMessage object into a JSON-serializable dictionary.
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
    Save interaction data (speech input, gaze input, and GPT function responses) to a JSON file.

    :param file_name: Name of the file where the data will be saved.
    :param responses: List of responses, as structured in your description.
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
