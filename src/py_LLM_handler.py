import importlib
import inspect
import openai
import sys
import os
import logging
import json
from typing import (
    Literal,
    Union,
    Optional,
)

sys.path.append(os.path.abspath("/hri/localdisk/emende/AttentiveSupport/src"))
from function_analyzer import FunctionAnalyzer

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
        
        self.SIM = SIM 

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
        self._user_emojis = "🧑 INPUT: "
        print("🤖 SYSTEM PROMPT: ",  self.character)

    def get_simulation(self):
        return self.SIM


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
    

    def play_with_functions_synchronized(self, input, person_name) -> None:
        user_message = {
            "role": "user",
            "content":  f"{person_name}: Input: {input}"
        }
        self.messages.append(user_message)
        response = self._query_llm(self.messages)
        self.messages.append(response.choices[0].message)

          # run with function calls as long as necessary
        while response.choices[0].message.tool_calls:
            tool_calls = response.choices[0].message.tool_calls
            input_ = [tc for tc in tool_calls if tc.function.name == "input"]
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
                    if self.SIM.hasBeenStopped:
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

    def play_with_functions_gaze_history_speech(self, speech_input, gaze_history, person_name) -> None:
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
