

#My first try at the problem; should be included in Clara's part below
import argparse
import asyncio

from rasa.shared.core.events import UserUttered

from rasa.core.policies.policy import PolicyPrediction

from rasa.core.actions.action import ActionListen, action_for_index, ActionUtterTemplate

import rasa
from rasa.core import constants
from rasa.core.channels import UserMessage
from rasa.shared.utils.io import json_to_string

from rasa.nlu.model import Interpreter

from rasa.model import get_model, get_model_subdirectories

from rasa.shared.constants import DEFAULT_MODELS_PATH, DEFAULT_CREDENTIALS_PATH, DEFAULT_ENDPOINTS_PATH

from rasa.cli.utils import get_validated_path
from rasa.cli.run import run as clirun

def run(core_model, message, args: argparse.Namespace):
    import rasa.run
    from rasa.core.agent import Agent
    args.endpoints = rasa.cli.utils.get_validated_path(
        args.endpoints, "endpoints", DEFAULT_ENDPOINTS_PATH, True
    )
    args.credentials = rasa.cli.utils.get_validated_path(
        args.credentials, "credentials", DEFAULT_CREDENTIALS_PATH, True
    )
    agent = Agent.load(core_model) # create agent earlier and use to parse message with interpreter
    nlu_parsed = asyncio.run(agent.parse_message_using_nlu_interpreter(message["text"]))
    print("nlu", nlu_parsed)
    processor = agent.create_processor()
    # result = processor.predict_next_with_tracker(message: UserMessage, tracker: DialogueStateTracker)
    #response = asyncio.run(processor.predict_next())
    tracker = asyncio.run(processor.fetch_tracker_and_update_session("user"))
    #asyncio.run(processor._handle_message_with_tracker(UserMessage(message["text"]),tracker))
    message = UserMessage(message["text"])
    parse_data = asyncio.run(processor.parse_message(message,tracker))
    # do our stuff to get new intent and confidence
    parse_data["intent"] = {'id': 1622048501520703154, 'name': 'bot_challenge', 'confidence': 4.4063952373107895e-05} #update parse_data with our new intnet ranking
    # parse_data["entities"] = ... # new intent ranking (probably not necessary)
    tracker.update(
        UserUttered(
            message.text,
            parse_data["intent"],
            parse_data["entities"],
            parse_data,
            input_channel=message.input_channel,
            message_id=message.message_id,
            metadata=message.metadata,
        ),
        processor.domain,
    )

    print("parsed", tracker.current_state())
    response = processor.predict_next_action(tracker)
    print("ensemble", processor.policy_ensemble)
    #agent.execute_action()
    #response = asyncio.run(agent.predict_next("1")) #agent.trigger_intent()
    print("re:" , response)
    a = PolicyPrediction
    b = ActionUtterTemplate
    print("probs",list(zip(response[1].probabilities, [action_for_index(x, processor.domain, processor.action_endpoint) for x in range(len(response[1].probabilities))])))
    print(response[1].policy_name)
    print(response[1].events)
    print(response[1].is_end_to_end_prediction)
    print(response[1].optional_events)
    print(response[0].name())
    #print(asyncio.run(response[0].run(output_channel=?, processor.nlg, tracker, processor.domain))) #TODO

    print("tracker", tracker.current_state())
    # TODO: clean up, make this the main method, integrate our stuff, make it an await and call with asyncio.run() => should us let get rid of all the asyncio.run()'s in here

def process_input(input:str, model_path:str):
    # Get model
    #model = get_validated_path("../../Softwareprojekt/rasa_test/models", "model", DEFAULT_MODELS_PATH)
    model = get_validated_path(model_path, "model", DEFAULT_MODELS_PATH)
    #try:
    model_path = get_model(model)
    """except ModelNotFound:
        print(
            "No model found. Train a model before running the "
            "server using `rasa train`."
        )
        return"""
    core_model, nlu_model = get_model_subdirectories(model_path)
    # NLU model
    interpreter = Interpreter.load(nlu_model)

    # compute NLU result
    #telemetry.track_shell_started("rasa")
    result = interpreter.parse(input)
    print(json_to_string(result))
    print(result)

    print("Core model path: ", core_model)
    # Compute response
        # set default args as in scaffold.py
    args = argparse.Namespace()
    attributes = [
        "endpoints",
        "credentials",
        "cors",
        "auth_token",
        "jwt_secret",
        "jwt_method",
        "enable_api",
        "remote_storage",
    ]
    for a in attributes:
        setattr(args, a, None)
    args.message = result

    args.port = constants.DEFAULT_SERVER_PORT
    args.connector = "cmdline" # from shell.py
    args.model = "../../Softwareprojekt/full_model/models"
    run(model_path, message=result, args=args)

model_path = "../../Softwareprojekt/full_model/models"
message = "hi there"
process_input(message, model_path)








