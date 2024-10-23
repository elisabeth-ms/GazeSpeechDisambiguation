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

class MicrophoneStream:
    def __init__(self, rate=16000, chunk_size=1600, hints_filename=None, transcription_queue=None):
        self.rate = rate
        self.chunk_size = chunk_size
        self.q = queue.Queue()
        self.hints_filename = hints_filename
        self.stream = None
        self.running = False
        self.processing_thread = None
        self.transcription_queue = transcription_queue # Shared queue for transcription

        # Load hints if any
        if hints_filename:
            if not os.path.isfile(hints_filename):
                raise IOError(f"Hints file '{hints_filename}' not found.")
            print(f"Loading hints file '{hints_filename}'.")
            with open(hints_filename) as hints_file:
                phrases = [x.strip() for x in hints_file.read().split("\n") if x.strip()]
            self.context = speech.SpeechContext(phrases=phrases)
        else:
            self.context = None

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        self.q.put(indata.copy())

    def start_streaming(self):
        self.running = True
        self.stream = sd.InputStream(
            samplerate=self.rate,
            channels=1,
            dtype='int16',
            callback=self.audio_callback,
            blocksize=self.chunk_size,
        )
        self.stream.start()
        print("Audio stream started.")

    def stop_streaming(self):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        print("Audio stream stopped.")

    def process_audio(self):
        client = speech.SpeechClient()
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.rate,
            language_code="en-US",
            max_alternatives=1,
            enable_word_time_offsets=True
        )
        if self.context:
            config.speech_contexts = [self.context]
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=False
        )

        # Generator function to yield audio content
        def generator():
            while self.running or not self.q.empty():  # Wait for queue to empty
                try:
                    data = self.q.get(timeout=1)
                    if data is None:
                        continue
                    audio_content = data.tobytes()
                    yield speech.StreamingRecognizeRequest(audio_content=audio_content)
                except queue.Empty:
                    continue

        requests = generator()
        try:
            responses = client.streaming_recognize(streaming_config, requests)
            self.listen_print_loop(responses)
        except Exception as e:
            print(f"Error during streaming recognition: {e}")

    def listen_print_loop(self, responses):
        for response in responses:
            if not response.results:
                continue

            result = response.results[0]
            if not result.alternatives:
                continue

            alternative = result.alternatives[0]
            transcript = alternative.transcript



            words_info = alternative.words

            word_data = [(word_info.word, word_info.start_time.total_seconds(), word_info.end_time.total_seconds())
                                for word_info in words_info]
            
            if self.transcription_queue:
                self.transcription_queue.put((transcript, word_data))                
            # print("\nTranscription:")
            # print(transcript)
            # print("\nWord-level timestamps:")
            # for word_info in words_info:
            #     word = word_info.word
            #     start_time = word_info.start_time.total_seconds()
            #     end_time = word_info.end_time.total_seconds()
            #     print(f"Word: '{word}', start_time: {start_time:.2f}s, end_time: {end_time:.2f}s")
            # print("\n" + "-"*40)








def plot_closest_objects(first_object_data):
    filtered_data = [(time, obj) for time, obj, dist, vel in first_object_data if obj is not None]

    unique_objects = list(set(obj for _, obj in filtered_data))

    object_mapping = {obj: i for i, obj in enumerate(unique_objects)}

    plt.figure(figsize=(10, 6))

    for obj in unique_objects:
        obj_times = [time for time, obj_name in filtered_data if obj_name == obj]
        obj_indices = [object_mapping[obj]] * len(obj_times)  

        plt.scatter(obj_times, obj_indices, label=obj, s=20)

    # Customize plot
    plt.yticks(list(object_mapping.values()), list(object_mapping.keys()))  
    plt.xlabel("Time")
    plt.ylabel("Objects")
    plt.title("Closest Object to Gaze Over Time")
    plt.legend()
    plt.grid(True)



def plot_word_timestamps(word_data):
    fig, ax = plt.subplots(figsize=(10, 2))  # Set a shorter height to make it a single horizontal bar
    
    # Plot each word on the same line (y=0)
    for i, (word, start_time, end_time) in enumerate(word_data):
        # Draw a horizontal line (same y=0 for all)
        ax.hlines(y=0, xmin=start_time, xmax=end_time, color=(random.random(), random.random(), random.random()), linewidth=8)
        
        # Add the word as text, placed at the middle of the line
        mid_time = (start_time + end_time) / 2
        ax.text(mid_time, 0, word, ha='center', va='center', fontsize=12, color='black')

    # Customize the plot
    ax.set_yticks([])  # Remove y-ticks
    ax.set_yticklabels([])  # Remove y-axis labels
    ax.set_ylim(-1, 1)  # Set y-limits to keep everything on one line
    ax.set_xlabel('Time (seconds)')
    ax.set_title('Word-level Timestamps')
    ax.grid(False)  # Turn off grid lines


    
def create_execution_folder(directory, test_number):
    """
    Create a new folder named 'test{number}' in the specified directory. The number will be
    incremented based on existing folders.
    
    :param directory: The base directory where the folder should be created.
    :return: The path to the newly created folder.
    """
    # Find the next available test number
    test_number = 1
    while os.path.exists(os.path.join(directory, f'test{test_number}')):
        test_number += 1

    # Create the new folder
    new_folder = os.path.join(directory, f'test{test_number}')
    os.makedirs(new_folder)
    
    print(f"Created folder: {new_folder}")
    return new_folder

class GazeDataManager:
    def __init__(self, threshold_angle=10.0, threshold_gaze_vel=0.0025, objects_not_wanted=None):
        self.threshold_angle = threshold_angle
        self.threshold_gaze_vel = threshold_gaze_vel
        self.objects_not_wanted = objects_not_wanted

    
    # Setter for threshold_angle
    def set_threshold_angle(self, threshold_angle):
        if threshold_angle > 0:
            self.threshold_angle = threshold_angle
        else:
            raise ValueError("Threshold angle must be positive")

    # Setter for threshold_gaze_vel
    def set_threshold_gaze_vel(self, threshold_gaze_vel):
        if threshold_gaze_vel > 0:
            self.threshold_gaze_vel = threshold_gaze_vel
        else:
            raise ValueError("Threshold gaze velocity must be positive")

    # Setter for objects_not_wanted
    def set_objects_not_wanted(self, objects_not_wanted):
        if isinstance(objects_not_wanted, list):
            self.objects_not_wanted = objects_not_wanted
        else:
            raise ValueError("Objects not wanted must be a list")

    def get_raw_gaze_data(self):
        global SIM
        return SIM.get_gaze_data()

    def get_filtered_gaze_data(self, gaze_data, start_time):
        first_object_data = []
        for data_point in gaze_data:
            person_name = data_point.get("agent_name")
            current_time = data_point.get("time")
            current_gaze_vel = data_point.get("gaze_velocity")

            if current_time >= start_time:
            
                objects = data_point.get("objects", [])        
                if objects:

                    # Initialize with the first object
                    selected_object_name = objects[0].get("name")
                    selected_object_angle_diff = objects[0].get("angleDiff")
                    selected_object_distance = objects[0].get("distance")
                    if selected_object_angle_diff> 20:
                        continue
                    # Iterate over the rest of the objects
                    for i in range(1, len(objects)):
                        current_object_name = objects[i].get("name")
                        current_object_angle_diff = objects[i].get("angleDiff")
                        current_object_distance = objects[i].get("distance")

                        # Check if the angle difference is below the threshold
                        if abs(selected_object_angle_diff - current_object_angle_diff) < self.threshold_angle and selected_object_name in self.objects_not_wanted:
                            # If the current object is closer, update the selected object
                            if current_object_distance < selected_object_distance:
                                selected_object_name = current_object_name
                                selected_object_angle_diff = current_object_angle_diff
                                selected_object_distance = current_object_distance
                            
                    if selected_object_name == 'camera':
                        selected_object_name = 'Johnnie'

                    if current_gaze_vel <= self.threshold_gaze_vel:
                        first_object_data.append((current_time-start_time, selected_object_name, selected_object_angle_diff, current_gaze_vel))
        return person_name, first_object_data
    



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
        print(self.messages)
        return self.messages


def save_speech_data_to_json(directory,file_name, press_s_time, press_f_time, transcript, word_data):
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

    # Save the data to a JSON file
    with open(directory+"/"+file_name, 'w') as f:
        json.dump(data, f, indent=4)


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

def save_gaze_data_to_json(directory, file_name, raw_gaze_data, start_time):
    # Save the raw gaze data to a JSON file
    raw_data_from_start_time = [entry for entry in raw_gaze_data if entry["time"] >= (start_time-0.2)]
    with open(directory+"/"+file_name, 'w') as f:
        json.dump(raw_data_from_start_time, f, indent=4)

def key_listener(llm_handler,stream, gaze_manager, transcription_queue, plot_speech_queue, plot_gaze_queue, print_emojis =False):
    print("Press 's' to start streaming, 'f' to stop streaming, 'e' to exit.")
    while True:
        key = getch()
        if key == 's':
            if not stream.running:
                stream.running = True
                print("\nStarting streaming...")
                gaze_start_time = getWallclockTime()
                stream.start_streaming()
                stream.processing_thread = threading.Thread(target=stream.process_audio)
                stream.processing_thread.start()
            else:
                print("\nAlready streaming.")
        elif key == 'f':
            if stream.running:
                transcript, gaze_history = None, None
                print("\nStopping streaming and waiting for final transcription...")
                stream.running = False
                stream.processing_thread.join()  # Wait for transcription to finish
                stream.stop_streaming()
                gaze_data = gaze_manager.get_raw_gaze_data()
                # person_name, filtered_gaze_data = gaze_manager.get_filtered_gaze_data(gaze_data, gaze_start_time)
                excluded_objects = ['hand_left_robot', 'hand_right_robot']
                person_name = 'Daniel'
                gaze_history, objects_timestamps = pyGaze.compute_gaze_history_closest_object(gaze_data, gaze_start_time, 10.0,10.0, 10.0, excluded_objects, 5.0, 0.5)

                # print("Gaze history: ", gaze_history)
                # for entry in gazed_objects_timestamps:
                #     print(f"Object: '{entry[0]}', start_time: {entry[1]:.2f}s, end_time: {entry[2]:.2f}s")

                while not transcription_queue.empty():
                    transcript, word_data = transcription_queue.get()

                    print(f"Received Transcription: {transcript}")
                    print("Word-level timestamps:")
                    for word, start_time, end_time in word_data:
                        print(f"Word: '{word}', start_time: {start_time:.2f}s, end_time: {end_time:.2f}s")
                    
                    speech_file_name = f'speech_data_{int(getWallclockTime())}.json'
                    gaze_data_file_name = f'gaze_data_{int(getWallclockTime())}.json'
                    global speech_directory_path
                    global gaze_directory_path
                    save_speech_data_to_json(speech_directory_path,speech_file_name, gaze_start_time, getWallclockTime(), transcript, word_data)
                    plot_speech_queue.put(word_data)
                    save_gaze_data_to_json(gaze_directory_path, gaze_data_file_name, gaze_data, gaze_start_time)

                plot_gaze_queue.put(objects_timestamps)
                if transcript and gaze_history:
                    # llm_handler.append_user_input(speech_input=transcript, gaze_history=gaze_history)
                    print(f"{llm_handler._user_speech_emojis if print_emojis else ''}{transcript}")
                    print(f"{llm_handler._user_gaze_emojis if print_emojis else ''}{gaze_history}")
                    llm_handler.play_with_functions(speech_input=transcript, gaze_history=gaze_history, person_name=person_name)
                    response = llm_handler.get_response()
                    # Save interaction data (speech input, gaze input, and GPT responses)
                    interaction_file_name = f'interaction_data_{int(getWallclockTime())}.json'
                    global gpt_responses_path
                    save_interaction_data_to_json(gpt_responses_path,interaction_file_name, transcript, gaze_history, response)
                    print("response: ", response)

            else:
                print("\nNot streaming.")
        elif key == 'e':
            if stream.running:
                print("\nStopping streaming before exiting...")
                stream.running = False
                stream.processing_thread.join()
                stream.stop_streaming()
            print("\nExiting...")
            os._exit(0)  # Force exit
        else:
            print("\nInvalid key. Press 's' to start, 'f' to stop, 'e' to exit.")




SIM = None
speech_directory_path = None
gpt_responses_path = None
filtered_gaze_data_directory_path = None
number_test =16
def main():
    # Create a command-line parser.
    parser = argparse.ArgumentParser(description="Google Cloud Speech-to-Text streaming with word-level timestamps.")

    parser.add_argument(
        "--sample-rate",
        default=16000,
        type=int,
        help="Sample rate in Hz of the audio data. Valid values are: 8000-48000. 16000 is optimal. [Default: 16000]",
    )
    parser.add_argument(
        "--hints",
        default=None,
        help="Name of the file containing hints for the ASR as one phrase per line [default: None]",
    )
    args = parser.parse_args()

    threshold_gaze_vel = 0.0025
    threshold_angle = 10.0 # deg
    objects_not_wanted = ['Johnnie', 'hand_left_robot', 'hand_right_robot', 'Daniel']
    global number_test
    global speech_directory_path
    global gpt_responses_path
    global filtered_gaze_data_directory_path
    global gaze_directory_path
    speech_directory_path = create_execution_folder("../../speechData", number_test)
    gpt_responses_path = create_execution_folder("../../gptResponses", number_test)
    gaze_directory_path = create_execution_folder("../../gazeData", number_test)
    filtered_gaze_data_directory_path = create_execution_folder("../../filteredGazeData", number_test)

    llm_handler = LLMHandler(config_module="gpt_gaze_speech_config")
    transcription_queue = queue.Queue()  # Shared queue to hold transcriptions
    plot_speech_queue = queue.Queue()  # Queue to hold data for plotting
    plot_gaze_queue = queue.Queue()
    plot_diagrams = True
    stream = MicrophoneStream(rate=args.sample_rate, hints_filename=args.hints, transcription_queue=transcription_queue)
    
    gaze_manager = GazeDataManager(threshold_angle=threshold_angle, threshold_gaze_vel=threshold_gaze_vel, objects_not_wanted=objects_not_wanted)

    # Start key listener thread
    key_thread = threading.Thread(target=key_listener, args=(llm_handler,stream, gaze_manager, transcription_queue, plot_speech_queue, plot_gaze_queue, True))
    key_thread.daemon = True
    key_thread.start()
    global SIM

    SIM.save_gaze_data_to_file("../../gazeData/","test"+str(number_test)+".csv")
    SIM.run()

    # Keep the main thread alive
    try:
        while True:
            time.sleep(0.1)
            word_data = None
            first_object_gaze_data = []
            if not key_thread.is_alive():
                break
            if plot_diagrams and not plot_speech_queue.empty() and not plot_gaze_queue.empty():
                while not plot_speech_queue.empty():
                    word_data = plot_speech_queue.get()

                if not plot_gaze_queue.empty():
                    first_object_gaze_data = plot_gaze_queue.get()

                if word_data and first_object_gaze_data:
                    pyGaze.plot_gaze_and_speech(first_object_gaze_data, word_data)
                    plt.show()                   
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        if stream.running:
            stream.running = False
            stream.processing_thread.join()
            stream.stop_streaming()
        sys.exit()


if __name__ == "__main__":
    main()
