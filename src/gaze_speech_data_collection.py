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
import json

if platform.system() == "Linux":
    sys.path.append("lib")
elif platform.system() == "Windows":
    sys.path.append("bin")

from pyAffaction import *
import pyGaze

gaze_start_time = -1.0
speaking_user = "Elisabeth"
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




class GazeDataManager:
    def __init__(self, SIM):
        self.SIM = SIM

    def get_raw_gaze_data(self):
        return self.SIM.get_gaze_data()


def key_listener(stream, gaze_manager, transcription_queue, plot_speech_queue, plot_gaze_queue):
    print("Press 's' to start streaming, 'f' to stop streaming, 'e' to exit.")
    while True:
        global SIM
        key = getch()
        if key == 's':
            if not stream.running:
                # SIM.load_transformation_data_from_file("/hri/localdisk/emende/inputData/test70/transformations/1729584048_data.json")
                # SIM.start_playback_transformation_data()
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
                print("\nStopping streaming and waiting for final transcription...")
                stream.running = False
                stream.processing_thread.join()  # Wait for transcription to finish
                stream.stop_streaming()
                all_users_raw_gaze_data = gaze_manager.get_raw_gaze_data()
                excluded_objects = ['hand_left_robot', 'hand_right_robot']
                for user_raw_gaze_data in all_users_raw_gaze_data:
                    gaze_history, objects_timestamps = pyGaze.compute_gaze_history_closest_object(user_raw_gaze_data["gaze_data"], gaze_start_time, 30.0,12.0, 12.0, excluded_objects, 5.0, 0.5)
                    print("Gaze history: ", gaze_history)
                    for entry in objects_timestamps:
                        print(f"Object: '{entry[0]}', start_time: {entry[1]:.2f}s, end_time: {entry[2]:.2f}s")
                
                foldername = f'{int(gaze_start_time)}_data'

                while not transcription_queue.empty():
                    transcript, word_data = transcription_queue.get()

                    print(f"Received Transcription: {transcript}")
                    print("Word-level timestamps:")
                    for word, start_time, end_time in word_data:
                        print(f"Word: '{word}', start_time: {start_time:.2f}s, end_time: {end_time:.2f}s")
                    # Save the speech data to a JSON file
                    global input_data_directory_path
                    os.makedirs(input_data_directory_path+"/speech/"+foldername, exist_ok=True)
                    save_speech_data_to_json(input_data_directory_path+"/speech/"+foldername, speaking_user+".json", gaze_start_time, getWallclockTime(), transcript, word_data)
                    plot_speech_queue.put(word_data)
                plot_gaze_queue.put(objects_timestamps)
                for user_raw_gaze_data in all_users_raw_gaze_data:
                    raw_gaze_data = user_raw_gaze_data["gaze_data"]
                    os.makedirs(input_data_directory_path+"/gaze/"+foldername, exist_ok=True)
                    save_gaze_data_to_json(input_data_directory_path+"/gaze/"+foldername, user_raw_gaze_data["agent_name"]+".json", raw_gaze_data, gaze_start_time)
                json_transformations = SIM.get_recorded_transformations(gaze_start_time, getWallclockTime())
                save_transformations_data_to_json(input_data_directory_path+"/transformations", foldername, json_transformations)
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

def save_gaze_data_to_json(directory, file_name, raw_gaze_data, start_time):
    # Save the raw gaze data to a JSON file
    raw_data_from_start_time = [entry for entry in raw_gaze_data if entry["time"] >= (start_time-0.2)]
    with open(directory+"/"+file_name, 'w') as f:
        json.dump(raw_data_from_start_time, f, indent=4)

def save_transformations_data_to_json(directory, file_name, json_transformations):
    # Save the raw gaze data to a JSON file
    with open(directory+"/"+file_name, 'w') as f:
        json.dump(json_transformations, f, indent=4)

input_data_directory_path = None
test_number = 1
SIM = None

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
    
    threshold_gaze_vel = 50.0 # deg/s
    threshold_angle = 10.0 # deg
    objects_not_wanted = ['Johnnie', 'hand_left_robot', 'hand_right_robot']
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




    global input_data_directory_path
    input_data_directory_path = "../../inputData"

    if not os.path.exists(input_data_directory_path):
        os.makedirs(input_data_directory_path)
    
    # Create a new folder for the current test
    input_data_directory_path = create_execution_folder(input_data_directory_path, test_number)

    # Create folder for speech data
    os.makedirs(os.path.join(input_data_directory_path, "speech"))
    # Create folder for gaze data
    os.makedirs(os.path.join(input_data_directory_path, "gaze"))
    # Create folder for transformations data
    os.makedirs(os.path.join(input_data_directory_path, "transformations"))
    
    objects_in_scene = SIM.get_objects()['objects']
    print("Objects in scene: ", objects_in_scene)
    agents_in_scene = SIM.get_agents()['agents']
    print("Agents in scene: ", agents_in_scene)

    test_data = {
    "xmlFileName": SIM.xmlFileName,
    "objects": objects_in_scene,
    "agents": agents_in_scene
    }


    # Save the test data to a JSON file
    with open(input_data_directory_path+'/description.json', 'w') as f:
        json.dump(test_data, f, indent=4)


    plot_diagrams = True
    transcription_queue = queue.Queue()  # Shared queue to hold transcriptions
    plot_speech_queue = queue.Queue()  # Queue to hold data for plotting
    plot_gaze_queue = queue.Queue()
    stream = MicrophoneStream(rate=args.sample_rate, hints_filename=args.hints, transcription_queue=transcription_queue)
    
    gaze_manager = GazeDataManager(SIM=SIM)

    # Start key listener thread
    key_thread = threading.Thread(target=key_listener, args=(stream, gaze_manager, transcription_queue, plot_speech_queue, plot_gaze_queue))
    key_thread.daemon = True
    key_thread.start()

    # Keep the main thread alive
    try:
        while True:
            time.sleep(0.1)
            word_data = None
            first_object_gaze_data = []
            if not key_thread.is_alive():
                break
            if plot_diagrams and not plot_speech_queue.empty() and not plot_gaze_queue.empty():
                if not plot_speech_queue.empty():
                    word_data = plot_speech_queue.get()

                if not plot_gaze_queue.empty():
                    first_object_gaze_data = plot_gaze_queue.get()

                if word_data and first_object_gaze_data:
                    print("Plotting gaze and speech data...")
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
