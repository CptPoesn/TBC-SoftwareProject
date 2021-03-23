
import argparse
import asyncio
import logging

import our_dialogue_manager as dm
from nltk import word_tokenize

from rasa.core.channels.channel import CollectingOutputChannel

from rasa.shared.core.events import UserUttered
from rasa.core import constants
from rasa.core.channels import UserMessage
from rasa.shared.utils.io import json_to_string
from rasa.nlu.model import Interpreter
from rasa.model import get_model, get_model_subdirectories
from rasa.shared.constants import DEFAULT_MODELS_PATH, DEFAULT_CREDENTIALS_PATH, DEFAULT_ENDPOINTS_PATH
from rasa.cli.utils import get_validated_path
from rasa.core.agent import Agent


class CoreModelInterface():
    def __init__(self, model_path, utterance_predictor="huggingtweets/ppredictors"):
        """
        Generate models that are the same for every execution
        """
        model = get_validated_path(model_path, "model", DEFAULT_MODELS_PATH)
        model_path = get_model(model)
        self.agent = Agent.load(model_path)
        self.processor = self.agent.create_processor()
        self.generator, self.tokenizer = dm.get_utterance_predictor_and_tokenizer(predictor=utterance_predictor)

    def process_message(self, message:str):

        #nlu_parsed = asyncio.run(self.agent.parse_message_using_nlu_interpreter(message))
        #logging.info(f"nlu {nlu_parsed}")
        # Get tracker and initial intent ranking
        tracker = asyncio.run(self.processor.fetch_tracker_and_update_session("user"))
        user_message = UserMessage(message)
        parse_data = asyncio.run(self.processor.parse_message(user_message,tracker))

        # run our dm to get new intent prediction
        tokenized_message = word_tokenize(message)
        _, intent, score, _ = dm.main(tokenized_message, self.generator, self.tokenizer, self.agent.interpreter)
        # parse_data["intent"] = {'id': 1622048501520703154, 'name': 'bot_challenge', 'confidence': 4.4063952373107895e-05} #update parse_data with our new intnet ranking
        parse_data["intent"] = {'id': 1622048501520703154, 'name': intent, 'confidence': 4.4063952373107895e-05} #update parse_data with our new intnet and score (with dummy id for the intent)

        tracker.update(
            UserUttered(
                message,
                parse_data["intent"],
                parse_data["entities"],
                parse_data,
                input_channel=user_message.input_channel,
                message_id=user_message.message_id,
                metadata=user_message.metadata,
            ),
            self.processor.domain,
        )

        # Generate system response
        #print("parsed", tracker.current_state())
        action,prediction = self.processor.predict_next_action(tracker)
        #print("probs",list(zip(prediction.probabilities, [action_for_index(x, processor.domain, processor.action_endpoint) for x in range(len(prediction.probabilities))])))
        output_channel = CollectingOutputChannel()
        messages = asyncio.run(action.run(output_channel, self.processor.nlg, tracker, self.processor.domain))
        system_reply = messages.pop().text

        #print("tracker", tracker.current_state())
        return action.name(), system_reply


if __name__ == "__main__":
    model_path = "C:/Users/schmi/Softwareprojekt/full_model/models"
    message = "hi there"
    interface = CoreModelInterface(model_path)
    action_name, system_reply = interface.process_message(message)
    print("action name: ", action_name)
    print("system reply: ", system_reply)








