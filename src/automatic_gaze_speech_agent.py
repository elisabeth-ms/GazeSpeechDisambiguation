import os
import sys
import sounddevice as sd
import queue
from google.cloud import speech
import time
import argparse
import threading
import platform

import py_LLM_handler
import data_handler
import pickle
import matplotlib.pyplot as plt

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


class MicrophoneStream:
    def __init__(self, rate=16000, chunk_size=1600, hints_filename=None, innactivity_timeout=2, gaze_manager=None, speech_gaze_queue=None):
        self.rate = rate
        self.chunk_size = chunk_size
        self.q = queue.Queue()
        self.hints_filename = hints_filename
        self.stream = None
        self.running = False
        self.processing_thread = None
        self.transcription_queue = queue.Queue() # Shared queue for transcription
        self.speech_gaze_queue = speech_gaze_queue
        self.inactivity_timeout = innactivity_timeout
        self.last_activity_time =  getWallclockTime()
        self.speech_detected = False  # Track whether any speech was detected
        self.first_recognition = True
        self.start_time = None

        self.gaze_manager = gaze_manager
        
        
        
        
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
        
        time.sleep(1)

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        self.q.put(indata.copy())

    def start_streaming(self):
        if self.stream is not None:
            print("Stream already running, restarting...")
            self.stop_streaming()  # Ensure clean restart
        self.running = True
        self.stream = sd.InputStream(
            samplerate=self.rate,
            channels=1,
            dtype='int16',
            callback=self.audio_callback,
            blocksize=self.chunk_size,
        )
        self.stream.start()
        self.start_time = getWallclockTime()  # Or another method to get absolute wall-clock time
        if not self.processing_thread or not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(target=self.process_audio, daemon=True)
            self.processing_thread.start()
        print("Audio stream started.")

    def stop_streaming(self):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join() # Wait for processing thread to finish
        self.q.queue.clear()  # Clear the queue
        print("Audio stream stopped.")

    def process_audio(self):
        print("Processing audio...")
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
        print("Running")
        requests = generator()
        try:
            responses = client.streaming_recognize(streaming_config, requests)
            print("Streaming recognition responses: ", responses)
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
            
            self.last_activity_time =  getWallclockTime()
            self.speech_detected = True
            
            self.transcription_queue.put((transcript, word_data))


    def monitor_inactivity(self):
        """Stop the stream if no activity within the timeout period."""
        gaze_history = None
        global interaction_folder_path
        global interaction_number
        global dialogue_number
        global dialogue_folder_path
        
        gaze_time_before_speaking = 2.0
        while self.running:
            if self.speech_detected and getWallclockTime() - self.last_activity_time > self.inactivity_timeout:
                print("Inactivity timeout reached. Stopping audio stream.")
                self.stop_streaming()

                all_users_raw_gaze_data = self.gaze_manager.get_all_users_raw_gaze_data()
                excluded_objects = ['hand_left_robot', 'hand_right_robot', 'camera']
                
              
                all_transcripts = []
                all_word_data = []
                full_transcript = None
                while not self.transcription_queue.empty():
                    transcript, word_data = self.transcription_queue.get()
                    all_transcripts.append(transcript)
                    all_word_data.extend(word_data)
                    # print(f"Transcript: {transcript}")
                if all_transcripts:
                    full_transcript = " ".join(all_transcripts)
                
                if all_word_data:
                    person_name = 'Elisabeth'
                    for user_raw_gaze_data in all_users_raw_gaze_data:
                        if user_raw_gaze_data["agent_name"] != person_name:
                            continue
                        print("Processing gaze data for user: ", user_raw_gaze_data["agent_name"])
                        gaze_history, objects_timestamps = pyGaze.compute_list_closest_objects_gaze_history(user_raw_gaze_data["gaze_data"], self.start_time+all_word_data[0][1]-gaze_time_before_speaking, 15.0,10.0, 10.0, excluded_objects, 5.0, 0.3, 0.04)
                    
                    print("Objects timestamps: ", objects_timestamps)
                    new_objects_timestamps = []
                    for object_timestamp in objects_timestamps:
                        new_objects_timestamps.append((object_timestamp[0], object_timestamp[1]+all_word_data[0][1]-gaze_time_before_speaking, object_timestamp[2]+all_word_data[0][1]-gaze_time_before_speaking))
                    
                        
                    
                    print("Lets check if the transcription queue has data: ")

                    print(f"Full transcript: {full_transcript}")
                    print("Gaze history: ", gaze_history)
                    interaction_folder_path = data_handler.create_interaction_folder(dialogue_folder_path,dialogue_number, interaction_number)
                    interaction_number += 1
                    data_handler.save_raw_gaze_data(interaction_folder_path, all_users_raw_gaze_data)
                    data_handler.save_speech_data(interaction_folder_path, person_name, self.start_time, getWallclockTime(), all_transcripts, all_word_data)
                    
                    
                    # print("All word data: ", all_word_data)
                    if all_word_data:
                        pyGaze.plot_multi_gaze_and_speech(new_objects_timestamps, all_word_data)
                        plt.savefig(interaction_folder_path+"/plot.png")
                        with open(interaction_folder_path+"/figure.fig", "wb") as f:
                            pickle.dump(plt.gcf(), f)
                        self.speech_gaze_queue.put((full_transcript, all_word_data, gaze_history, new_objects_timestamps))

                self.speech_detected = False  # Reset speech detection flag
                self.running = False
                self.first_recognition = True
                self.start_streaming()


            time.sleep(0.1) 



def llm_processing_loop(llm_handler, speech_gaze_queue, person_name, input_mode="gaze_history_speech", print_emojis=True):
    """Thread loop to process LLM interactions."""
    while True:
        try:

            if not speech_gaze_queue.empty():
                full_transcript, all_words_data, gaze_history, objects_timestamps = speech_gaze_queue.get()
                print(f"Full transcript: {full_transcript}")
                if input_mode == "speech_only":
                    print(f"{llm_handler._user_speech_emojis if print_emojis else ''}{full_transcript}")
                    llm_handler.play_with_functions_synchronized(full_transcript, person_name)
                elif input_mode == "gaze_history_speech":
                    print(f"{llm_handler._user_speech_emojis if print_emojis else ''}{full_transcript}")
                    print(f"{llm_handler._user_gaze_emojis if print_emojis else ''}{gaze_history}")
                    llm_handler.play_with_functions_gaze_history_speech(
                        speech_input=full_transcript,
                        gaze_history=gaze_history,
                        person_name=person_name
                    )
                elif input_mode == "synchronized_gaze_speech":
                    input = pyGaze.merge_gaze_word_intervals(objects_timestamps, all_words_data)
                    print(f"{llm_handler._user_emojis if print_emojis else ''}{input}")
                    llm_handler.play_with_functions_synchronized(input=input, person_name=person_name)

                response = llm_handler.get_response()
    
                # Save interaction data (speech input, gaze input, and GPT responses)
                data_handler.save_interaction_data_to_json(interaction_folder_path,"interaction_data.json", response)
                
               

        except Exception as e:
            print(f"Error in LLM processing loop: {e}")

        time.sleep(0.5)  # Prevent busy loop


## In the automatic_gaze_speech_agent.py, we can't use gaze_only mode because we need the speech to be able to segment the gaze data

input_mode = "gaze_history_speech" # Options: "speech_only", "gaze_history_speech", "synchronized_gaze_speech"
config_file = "gpt_gaze_speech_config"

SIM = None
speech_directory_path = None
gpt_responses_path = None
filtered_gaze_data_directory_path = None
recordTransformationsEnabled = None

main_dir = 'automatic_interaction_recordings'
main_dir_path = os.path.join('/hri/storage/user/emenende', main_dir)

dialogue_number = 0
dialogue_folder_path = None

interaction_number = 1
interaction_folder_path = None

gaze_start_time = -1.0

main_dir_path = data_handler.create_interaction_recordings(main_dir_path)
while dialogue_folder_path is None:
    dialogue_number += 1
    dialogue_folder_path = data_handler.create_dialogue_folder(main_dir_path, dialogue_number)
print(f"Dialogue folder path: {dialogue_folder_path}")


                
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


    time.sleep(1)
    llm_handler = py_LLM_handler.LLMHandler(config_module=config_file)

    global SIM
    SIM = llm_handler.get_simulation()
    gaze_manager = GazeDataManager(SIM)
    speech_gaze_queue = queue.Queue()
    stream = MicrophoneStream(rate=args.sample_rate, hints_filename=args.hints, innactivity_timeout=2, gaze_manager=gaze_manager, speech_gaze_queue=speech_gaze_queue)

    print_emojis = True
    person_name = 'Elisabeth'

    try:
        print("Starting stream...")
        stream.start_streaming()
        
        threading.Thread(target=stream.monitor_inactivity, daemon=True).start()

        
        # Start audio processing thread
        stream.processing_thread = threading.Thread(target=stream.process_audio)
        stream.processing_thread.start()

        # Start LLM processing in a separate thread
        llm_thread = threading.Thread(
            target=llm_processing_loop,
            args=(llm_handler, speech_gaze_queue, person_name, input_mode, print_emojis),
            daemon=True  # LLM thread stops when main program exits
        )
        llm_thread.start()
        SIM.run()

        stream.processing_thread.join()
        llm_thread.join()
        # Keep simulation running
        
        while True:
            time.sleep(0.5)
            

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print("Stopping stream...")
        stream.stop_streaming()

if __name__ == "__main__":
    main()