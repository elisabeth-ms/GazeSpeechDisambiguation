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


sys.path.append("../src/AttentiveSupport/src/")
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
sys.path.append("../src/AttentiveSupport/src/")
import pyGaze

class MissingEnvironmentVariable(Exception):
    pass


if "OPENAI_API_KEY" not in os.environ:
    raise MissingEnvironmentVariable(
        "Please set an environment variable with your OPENAI_API_KEY. "
        "See https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety"
    )


gaze_start_time = -1.0

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

    

class LLMHandler:
    def __init__(self, config_module: str = "gpt_gaze_speech_scene_config"):
        # Dynamic config loading
        config = importlib.import_module(config_module)
        tool_module = importlib.import_module(config.tool_module)
        tools = {
            n: f for n, f in inspect.getmembers(tool_module) if inspect.isfunction(f)
        }
        global SIM
        SIM = tool_module.SIMULATION
        
        # global recordTransformationsEnabled
        # recordTransformationsEnabled = tool_module.recordTransformationsEnabled

        # LLM settings
        if not os.path.isfile(os.getenv("OPENAI_API_KEY")):
            openai.api_key_path = None
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = openai.OpenAI()
        self.model = config.model_name
        self.temperature = config.temperature

        # Character
        self.name = "Tiago"
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
        # print(f"Total tokens used in this call: {token_usage}")
        return response    
    

    def play_with_functions(self, speech_input, gaze_history, person_name) -> None:
        user_message = {
            "role": "user",
            "content":  f"{person_name}:  Speech input: {speech_input}. Gaze history (in seconds): {gaze_history}"
        }
        self.messages.append(user_message)
        response = self._query_llm(self.messages)
        self.messages.append(response.choices[0].message)

          # run with function calls as long as necessary
        while response.choices[0].message.tool_calls:
            tool_calls = response.choices[0].message.tool_calls
            gaze_ = [tc for tc in tool_calls if tc.function.name == "gaze"]
            speech_ = [tc for tc in tool_calls if tc.function.name == "speak"]
            actions_ = [
                tc for tc in tool_calls if tc not in gaze_ and tc not in speech_
            ]
            for tcs in [gaze_, speech_, actions_]:
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
        # print(self.messages)
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

def save_interaction_data_to_json(directory,file_name, speech_input, gaze_input, responses):
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





SIM = None
speech_directory_path = None
gpt_responses_path = None
filtered_gaze_data_directory_path = None
recordTransformationsEnabled = None

main_dir = 'terminal_interaction_recordings'
main_dir_path = os.path.join('/hri/storage/user/emenende', main_dir)

dialogue_number = 0
dialogue_folder_path = None

interaction_number = 1
interaction_folder_path = None

main_dir_path = create_interaction_recordings(main_dir_path)
while dialogue_folder_path is None:
    dialogue_number += 1
    dialogue_folder_path = create_dialogue_folder(main_dir_path, dialogue_number)
print(f"Dialogue folder path: {dialogue_folder_path}")

def main():

    print_emojis = True
    person_name = "Elisabeth"
    llm_handler = LLMHandler(config_module="gpt_gaze_speech_scene_tiago_config")
    global SIM

    SIM.run()
    print("Welcome to 'gaze_speech_terminal_agent'!")
    print("Type 'quit' to exit.")
    # Keep the main thread alive
    
    while True:
        try:
            speech_input = input("Enter SPEECH INPUT: ")
            if speech_input.lower() == 'quit':
                break
            
            gaze_input = input("Enter GAZE INPUT: ")
            if gaze_input.lower() == 'quit':
                break
            
            
            if speech_input and gaze_input:
                # llm_handler.append_user_input(speech_input=transcript, gaze_history=gaze_history)
                print(f"{llm_handler._user_speech_emojis if print_emojis else ''}{speech_input}")
                print(f"{llm_handler._user_gaze_emojis if print_emojis else ''}{gaze_input}")
                llm_handler.play_with_functions(speech_input=speech_input, gaze_history=gaze_input, person_name=person_name)
                response = llm_handler.get_response()
                # Save interaction data (speech input, gaze input, and GPT responses)
                # print("response: ", response)

            

        except KeyboardInterrupt:
            print("\nInterrupted by user")
            sys.exit()


if __name__ == "__main__":
    main()
