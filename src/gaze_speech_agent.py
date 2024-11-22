import sys
import argparse
import time
import sounddevice as sd
import queue
import threading
from google.cloud import speech
import os
import platform
import sys
import time
import getch
import matplotlib.pyplot as plt

import py_LLM_handler
import data_handler
import pickle

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

    # Setter for objects_not_wanted
    def set_objects_not_wanted(self, objects_not_wanted):
        if isinstance(objects_not_wanted, list):
            self.objects_not_wanted = objects_not_wanted
        else:
            raise ValueError("Objects not wanted must be a list")

    def get_all_users_raw_gaze_data(self):
        return self.SIM.get_gaze_data()


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
                all_users_raw_gaze_data = gaze_manager.get_all_users_raw_gaze_data()
                

                person_name = 'Elisabeth'
                for user_raw_gaze_data in all_users_raw_gaze_data:
                    if user_raw_gaze_data["agent_name"] != person_name:
                        continue
                    gaze_history, objects_timestamps = pyGaze.compute_list_closest_objects_gaze_history(user_raw_gaze_data["gaze_data"], gaze_start_time, gaze_velocity_threshold, angle_diff_threshold, angle_diff_xz_threshold, excluded_objects, off_target_velocity_threshold, off_target_duration_threshold, minimum_fixation_duration)

                

                all_transcripts = []
                all_word_data = []
                full_transcript = None
                while not transcription_queue.empty():
                    transcript, word_data = transcription_queue.get()
                    all_transcripts.append(transcript)
                    all_word_data.extend(word_data)
                    print(transcript)
                if all_transcripts:
                    full_transcript = " ".join(all_transcripts)
                
                    # Lets create an interaction folder
                    global interaction_folder_path
                    global interaction_number
                    global dialogue_number
                    global dialogue_folder_path
                    interaction_folder_path = data_handler.create_interaction_folder(dialogue_folder_path,dialogue_number, interaction_number)
                    interaction_number += 1
                    data_handler.save_raw_gaze_data(interaction_folder_path, all_users_raw_gaze_data)
                    data_handler.save_speech_data(interaction_folder_path, person_name, gaze_start_time, getWallclockTime(), all_transcripts, all_word_data)
                   
                    global recordTransformationsEnabled
                    recordTransformationsEnabled = SIM.recordTransformations
                    if recordTransformationsEnabled:
                        json_transformations = SIM.get_recorded_transformations(gaze_start_time, getWallclockTime())
                        data_handler.save_transformations_data_to_json(interaction_folder_path, 'transformations.json', json_transformations)
                    plot_speech_queue.put(all_word_data)

                plot_gaze_queue.put(objects_timestamps)
                if input_mode == "gaze_only":
                    print(f"{llm_handler._user_gaze_emojis if print_emojis else ''}{gaze_history}")
                    llm_handler.play_with_functions_synchronized(gaze_history, person_name=person_name)
                else:
                    if full_transcript and gaze_history:
                        
                        if input_mode == "speech_only":
                            print(f"{llm_handler._user_speech_emojis if print_emojis else ''}{full_transcript}")
                            llm_handler.play_with_functions_synchronized(full_transcript, person_name=person_name)
                            
                        elif input_mode == "gaze_history_speech":
                            print(f"{llm_handler._user_speech_emojis if print_emojis else ''}{full_transcript}")
                            print(f"{llm_handler._user_gaze_emojis if print_emojis else ''}{gaze_history}")
                            llm_handler.play_with_functions_gaze_history_speech(speech_input=full_transcript, gaze_history=gaze_history, person_name=person_name)

                        elif input_mode == "synchronized_gaze_speech": 
                            input = pyGaze.merge_gaze_word_intervals(objects_timestamps, word_data)
                            print(f"{llm_handler._user_emojis if print_emojis else ''}{input}")
                            print(f"{llm_handler._user_speech_emojis if print_emojis else ''}{full_transcript}")
                            print(f"{llm_handler._user_gaze_emojis if print_emojis else ''}{objects_timestamps}")
                            llm_handler.play_with_functions_synchronized(input=input, person_name=person_name)

                        else:
                            print("Invalid input mode. Please set input_mode to 'speech_only', 'gaze_only', 'gaze_history_speech', or 'synchronized_gaze_speech'.")
 
                response = llm_handler.get_response()
    
                # Save interaction data (speech input, gaze input, and GPT responses)
                data_handler.save_interaction_data_to_json(interaction_folder_path,"interaction_data.json", response)


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



input_mode = "gaze_history_speech" # Options: "speech_only", "gaze_only", "gaze_history_speech", "synchronized_gaze_speech"
config_file = "gpt_gaze_speech_config"

SIM = None
speech_directory_path = None
gpt_responses_path = None
filtered_gaze_data_directory_path = None
recordTransformationsEnabled = None

main_dir = 'interaction_recordings'
main_dir_path = os.path.join('/hri/localdisk/emende', main_dir)

# main_dir_path = os.path.join('/hri/storage/user/emenende', main_dir)

dialogue_number = 0
dialogue_folder_path = None

interaction_number = 1
interaction_folder_path = None

gaze_start_time = -1.0

gaze_velocity_threshold = 15.0
angle_diff_threshold = 8.0
angle_diff_xz_threshold = 8.0
excluded_objects = ['hand_left_robot', 'hand_right_robot']
off_target_velocity_threshold = 5.0
off_target_duration_threshold = 0.5
minimum_fixation_duration = 0.08

main_dir_path = data_handler.create_interaction_recordings(main_dir_path)
while dialogue_folder_path is None:
    dialogue_number += 1
    dialogue_folder_path = data_handler.create_dialogue_folder(main_dir_path, dialogue_number)
print(f"Dialogue folder path: {dialogue_folder_path}")

config_data_file = os.path.join(dialogue_folder_path, 'config.json')

data_handler.save_config_to_json(config_data_file, gaze_velocity_threshold, angle_diff_threshold, angle_diff_xz_threshold, excluded_objects, off_target_velocity_threshold, off_target_duration_threshold, minimum_fixation_duration)

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



    llm_handler = py_LLM_handler.LLMHandler(config_module=config_file)
    transcription_queue = queue.Queue()  # Shared queue to hold transcriptions
    plot_speech_queue = queue.Queue()  # Queue to hold data for plotting
    plot_gaze_queue = queue.Queue()
    plot_diagrams = True
    stream = MicrophoneStream(rate=args.sample_rate, hints_filename=args.hints, transcription_queue=transcription_queue)
    global SIM
    SIM = llm_handler.get_simulation()
    gaze_manager = GazeDataManager(SIM)

    # Start key listener thread
    key_thread = threading.Thread(target=key_listener, args=(llm_handler,stream, gaze_manager, transcription_queue, plot_speech_queue, plot_gaze_queue, True))
    key_thread.daemon = True
    key_thread.start()

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
                    pyGaze.plot_multi_gaze_and_speech(first_object_gaze_data, word_data)
                    global interaction_folder_path
                    plt.savefig(interaction_folder_path+"/plot.png")
                    with open(interaction_folder_path+"/figure.fig", "wb") as f:
                        pickle.dump(plt.gcf(), f)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        if stream.running:
            stream.running = False
            stream.processing_thread.join()
            stream.stop_streaming()
        sys.exit()


if __name__ == "__main__":
    main()
