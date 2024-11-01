import sys
import argparse
import time
import sounddevice as sd
import numpy as np
import queue
import threading
from google.cloud import speech
import os
import platform
import sys
import time
import getch
import matplotlib.pyplot as plt
import random

import openai
import importlib
import logging
import inspect
from typing import List, Tuple, Optional, Dict, Any


sys.path.append(os.path.abspath("/hri/localdisk/emende/AttentiveSupport/src"))
from function_analyzer import FunctionAnalyzer

from typing import (
    Literal,
    Union,
    Optional,
)
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


gaze_start_time = -1.0
main_dir_path = '/hri/storage/user/emenende/interaction_recordings'

# Initialize dialogue and interaction counters
dialogue_number = 46  # Adjust this based on which dialogue you want to load
interaction_number = 1  # Adjust this based on which interaction to start with

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



class LLMHandler:
    def __init__(self, config_module: str = "gpt_gaze_speech_config"):
        # Dynamic config loading
        config = importlib.import_module(config_module)
        tool_module = importlib.import_module(config.tool_module)
        tools = {
            n: f for n, f in inspect.getmembers(tool_module) if inspect.isfunction(f)
        }
        global SIM
        SIM = tool_module.SIMULATION
        
        global recordTransformationsEnabled
        recordTransformationsEnabled = tool_module.recordTransformationsEnabled

        # LLM settings
        if not os.path.isfile(os.getenv("OPENAI_API_KEY")):
            openai.api_key_path = None
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = openai.OpenAI()
        self.model = config.model_name
        self.temperature = config.temperature

        # Character
        self.name = "Johnnie"
        self.character: str = config.system_prompt.format(name=self.name)
        self.function_resolver = tools
        self.function_analyzer = FunctionAnalyzer()

        self.tools_descriptions = [
            self.function_analyzer.analyze_function(function_)
            for function_ in tools.values()
        ]
        self.amnesic: bool = False

        self.messages = [
            {"role": "system", "content": self.character},
        ]

        self._user_speech_emojis = "🧑‍🎙️  SPEECH INPUT: "
        self._user_gaze_emojis = "🧑👀 GAZE INPUT: "

        print("🤖 SYSTEM PROMPT: ",  self.character)



    def _query_llm(self, 
                   messages,
                   tool_choice: Union[Literal["none", "auto"]] = "auto",
                   retries: int =3,):
        response, i = None, 0

        while True:
            i +=1
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=messages,
                    tools = self.tools_descriptions,
                    tool_choice=tool_choice,
                )
                logging.info(response)
            except openai.OpenAIError as e:
                logging.error(f"❌ OpenAI error, retrying ({e})")
            if response:
                break
            if i >= retries:
                raise Exception(f"❌ {retries} OpenAI errors, aborting.")
                # Get the number of tokens used
        token_usage = response.usage.total_tokens
        print(f"Total tokens used in this call: {token_usage}")
        return response    
    

    def play_with_functions(self, input, person_name) -> None:
        user_message = {
            "role": "user",
            "content":  f"{person_name}:  Input: {input}"
        }
        self.messages.append(user_message)
        response = self._query_llm(self.messages)
        self.messages.append(response.choices[0].message)

          # run with function calls as long as necessary
        while response.choices[0].message.tool_calls:
            tool_calls = response.choices[0].message.tool_calls
            input_ = [tc for tc in tool_calls if tc.function.name == "input"]
            # gaze_ = [tc for tc in tool_calls if tc.function.name == "gaze"]
            # speech_ = [tc for tc in tool_calls if tc.function.name == "speak"]
            actions_ = [
                tc for tc in tool_calls if tc not in input_
            ]
            for tcs in [input_, actions_]:
                if not tcs:
                    continue
                for tc in tcs:
                    function_call = tc.function
                    # invoke function
                    func = function_call.name
                    fn_args = json.loads(function_call.arguments)
                    print(
                        "🤖🔧 GPT response is function call: "
                        + func
                        + "("
                        + str(fn_args)
                        + ")"
                    )
                    if SIM.hasBeenStopped:
                        fn_res = "This task has not been completed due to user interruption."
                    else:
                        fcn = self.function_resolver[func]
                        fn_res = fcn(**fn_args)
                    # track function result
                    print("🔧 Function result is: " + fn_res)
                    self.messages.append(
                        {
                            "role": "tool",
                            "name": func,
                            "content": fn_res,
                            "tool_call_id": tc.id,
                        }
                    )
            if SIM.hasBeenStopped:
                break
            else:
                response = self._query_llm(self.messages)
                self.messages.append(response.choices[0].message)

        if SIM.hasBeenStopped:
            SIM.hasBeenStopped = False
            print("🤖💭 I WAS STOPPED: Getting back to my default pose.")
            self.reset_after_interrupt()
            self.messages.append(
                {
                    "role": "system",
                    "content": f"You were stopped by the user and are now back in your default pose.",
                },
            )
        else:
            print("🤖💭 FINAL RESPONSE: " + response.choices[0].message.content)

        if self.amnesic:
            self.reset()



    def append_user_input(self, speech_input, gaze_history):
        # Append the user speech input and gaze history to the messages
        user_message = {
            "role": "user",
            "content": f"Speech input: {speech_input}. Gaze history (in seconds): {gaze_history}"
        }
        self.messages.append(user_message)

    def get_response(self):
        # Get the GPT response
        response = self._query_llm(self.messages)
        message = response.choices[0].message
        self.messages.append(response.choices[0].message)
        print(self.messages)
        return self.messages



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




SIM = None
speech_directory_path = None
gpt_responses_path = None
filtered_gaze_data_directory_path = None
recordTransformationsEnabled = None

main_dir = 'interaction_recordings'
main_dir_path = os.path.join('/hri/storage/user/emenende', main_dir)
excluded_objects = ['hand_left_robot', 'hand_right_robot', 'camera']

def create_combined_input(speech_input, objects_timestamps):
    start_time = speech_input['listening_start_time']
    word_data = [(word_info['word'], word_info['start_time'], word_info['end_time']) for word_info in speech_input['words']]
    relative_objects_timestamps =[]
    # Convert absolute gaze times to relative times
    for objects, start, end in objects_timestamps:
        if start > start_time or end > start_time:
            start -= start_time
            end -= start_time
            if start < 0:
                start = 0
            relative_objects_timestamps.append((objects, start, end))
    
    print("Word data: ", word_data)
    print("Gaze timestamps: ", relative_objects_timestamps)
    # Initialize the result structure



def merge_gaze_word_intervals(
        gaze_data: List[Tuple[List[str], float, float]],
        word_data: List[Tuple[str, float, float]]
    ) -> Dict[str, Any]:

    merged_rows = []
    speech_idx = 0
    gaze_idx = 0
    current_gaze_objects = []
    
    while gaze_idx < len(gaze_data) or speech_idx < len(word_data):
        if speech_idx < len(word_data):
            word, word_start, word_end = word_data[speech_idx]
        else:
            word, word_start, word_end = None, float('inf'), float('inf')
        
        if gaze_idx < len(gaze_data):
            gaze_objects, gaze_start, gaze_end = gaze_data[gaze_idx]
        else:
            gaze_objects, gaze_start, gaze_end = [], float('inf'), float('inf')

        if word_start < gaze_start:  # Speech starts first
            time_str = f"{word_start:.3f}-{word_end:.3f}"
            if current_gaze_objects:
                merged_rows.append([time_str, word, current_gaze_objects])
            else:
                merged_rows.append([time_str, word, []])
            speech_idx += 1
            current_gaze_objects = []  # Reset for next word
        else:  # Gaze starts first or overlaps with word
            time_str = f"{gaze_start:.3f}-{gaze_end:.3f}"
            if gaze_start >= word_start and gaze_end <= word_end:  # Gaze falls within word interval
                if current_gaze_objects and current_gaze_objects[-1][0] == gaze_objects:
                    # Extend the last time interval for the same objects
                    current_gaze_objects[-1] = (gaze_objects, f"{current_gaze_objects[-1][1].split('-')[0]}-{gaze_end:.3f}")
                else:
                    current_gaze_objects.append((gaze_objects, time_str))
            elif gaze_end > word_end:  # Gaze continues beyond word
                merged_rows.append([f"{word_start:.3f}-{word_end:.3f}", word, current_gaze_objects])
                speech_idx += 1
                current_gaze_objects = [(gaze_objects, f"{word_end:.3f}-{gaze_end:.3f}")]
            else:
                merged_rows.append([time_str, None, [gaze_objects]])  # No word during this gaze interval
                current_gaze_objects = []
            gaze_idx += 1

    return {
        "headers": ["Time", "Word", "Gazed Objects"],
        "rows": merged_rows
    }

    
    

# Loop through dialogues and interactions, load pre-recorded data
def run_offline_interactions(llm_handler, main_dir_path, dialogue_number, interaction_number, user_name="Elisabeth"):
    dialogue_folder_path = os.path.join(main_dir_path, f'dialogue{dialogue_number}')
    
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
        
        # Extract relevant information from speech data
        speech_input = speech_data["transcript"]
        start_time = speech_data["listening_start_time"]

        gaze_history, objects_timestamps = pyGaze.compute_list_closest_objects_gaze_history(user_raw_gaze_data["gaze_data"], gaze_start_time, 15.0,10.0, 10.0, excluded_objects, 5.0, 0.5, 0.01)

        
        start_time = speech_data['listening_start_time']
        word_data = [(word_info['word'], word_info['start_time'], word_info['end_time']) for word_info in speech_data['words']]
        relative_objects_timestamps =[]
        # Convert absolute gaze times to relative times
        for objects, start, end in objects_timestamps:
            if start > start_time or end > start_time:
                start -= start_time
                end -= start_time
                if start < 0:
                    start = 0
                relative_objects_timestamps.append((objects, start, end))
    
        print("Word data: ", word_data)
        print("Gaze timestamps: ", relative_objects_timestamps)
        
        input = merge_gaze_word_intervals(relative_objects_timestamps, word_data)
        print(json.dumps(input, indent=4))
                
        # # Extract gaze history from gaze data
        # # Here you might process `gaze_data` to get it in the same format as required by `play_with_functions`
        # gaze_history = [
        #     (entry["gazed_objects"], entry["duration"]) for entry in gaze_data
        # ]  # Modify this line according to your gaze JSON structure

        # # Print or log the loaded data for debugging
        # print(f"Interaction {interaction_number}:")
        # print(f"Speech Input: {speech_input}")
        # print(f"Gaze History: {gaze_history}")
        
        # # Send pre-recorded data to LLM for response
        llm_handler.play_with_functions(input=input, person_name=user_name)

        # Move to the next interaction
        interaction_number += 1
        


def main():

    print_emojis = True
    person_name = "Elisabeth"
    llm_handler = LLMHandler(config_module="gpt_time_synchronized_gaze_speech_config")
    global SIM
    SIM.run()
    
    run_offline_interactions(llm_handler, main_dir_path, dialogue_number, interaction_number, user_name="Elisabeth")

            

if __name__ == "__main__":
    main()
