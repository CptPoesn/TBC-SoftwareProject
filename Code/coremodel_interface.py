
import argparse
import asyncio

import our_dialogue_manager as dm
from nltk import word_tokenize

from rasa.core.channels.channel import CollectingOutputChannel

from rasa.shared.core.events import UserUttered
from rasa.core.channels import UserMessage
from rasa.model import get_model
from rasa.shared.constants import DEFAULT_MODELS_PATH
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
        action,prediction = self.processor.predict_next_action(tracker)
        output_channel = CollectingOutputChannel()
        messages = asyncio.run(action.run(output_channel, self.processor.nlg, tracker, self.processor.domain))
        try:
            system_reply = messages[-1].text
        except AttributeError:
            return action.name(), (None, f"events: {messages}")

        return action.name(), system_reply



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='An interactive interface to our dialog agent.')
    parser.add_argument("model",
                        help="""Path to the "/models" directory with a model trained for NLU and core.""")
    args = parser.parse_args()

    interface = CoreModelInterface(args.model)
    while True:
        message = input("You: ")
        action_name, system_reply = interface.process_message(message)
        print("Action name: ", action_name)
        print("System reply: ", system_reply)
        print("\n--------------------------------------------\n")








