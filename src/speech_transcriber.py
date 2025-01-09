import os
import sys
import sounddevice as sd
import queue
from google.cloud import speech
import time
import argparse
import threading
class MicrophoneStream:
    def __init__(self, rate=16000, chunk_size=1600, hints_filename=None, transcription_queue=None, innactivity_timeout=2):
        self.rate = rate
        self.chunk_size = chunk_size
        self.q = queue.Queue()
        self.hints_filename = hints_filename
        self.stream = None
        self.running = False
        self.processing_thread = None
        self.transcription_queue = transcription_queue # Shared queue for transcription
        
        self.inactivity_timeout = innactivity_timeout
        self.last_activity_time = time.time()
        self.speech_detected = False  # Track whether any speech was detected

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
        threading.Thread(target=self.monitor_inactivity, daemon=True).start()

        print("Audio stream started.")

    def stop_streaming(self):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
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
        while self.running:

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
            
            self.last_activity_time = time.time()
            self.speech_detected = True
            
            if self.transcription_queue:
                self.transcription_queue.put((transcript, word_data))


    def monitor_inactivity(self):
        """Stop the stream if no activity within the timeout period."""
        while self.running:
            if self.speech_detected and time.time() - self.last_activity_time > self.inactivity_timeout:
                print("Inactivity timeout reached. Stopping audio stream.")
                self.stop_streaming()
                
                
                all_transcripts = []
                all_word_data = []
                full_transcript = None
                while not self.transcription_queue.empty():
                    transcript, word_data = self.transcription_queue.get()
                    all_transcripts.append(transcript)
                    all_word_data.extend(word_data)
                if all_transcripts:
                    full_transcript = " ".join(all_transcripts)
                    print(f"Full transcript: {full_transcript}")
                self.speech_detected = False  # Reset speech detection flag
                self.running = False
                self.start_streaming()
            time.sleep(0.2) 

                
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

    transcription_queue = queue.Queue()  # Shared queue to hold transcriptions
    stream = MicrophoneStream(rate=args.sample_rate, hints_filename=args.hints, transcription_queue=transcription_queue)
    
    
    try:
        print("Starting stream...")
        stream.start_streaming()
        stream.processing_thread = threading.Thread(target=stream.process_audio)
        
        stream.processing_thread.start()

        # Allow processing until user interruption
        stream.processing_thread.join()

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print("Stopping stream...")
        stream.stop_streaming()

if __name__ == "__main__":
    main()