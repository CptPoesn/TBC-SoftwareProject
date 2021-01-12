import argparse

from rasa import telemetry

from rasa.cli.utils import get_validated_path
from rasa.core import constants
from rasa.exceptions import ModelNotFound
from rasa.model import get_model, get_model_subdirectories
from rasa.nlu.model import Interpreter
from rasa.shared.constants import DEFAULT_MODELS_PATH
from rasa.shared.nlu.interpreter import RegexInterpreter
from rasa.shared.utils.io import json_to_string
import rasa.cli.run



def process_input(input:str, model_path:str):
    # Get model
    model = get_validated_path("../../Softwareprojekt/rasa_test/models", "model", DEFAULT_MODELS_PATH)
    try:
        model_path = get_model(model)
    except ModelNotFound:
        print(
            "No model found. Train a model before running the "
            "server using `rasa train`."
        )
        return
    core_model, nlu_model = get_model_subdirectories(model_path)
    # NLU model
    interpreter = Interpreter.load(nlu_model)

    # compute NLU result
    telemetry.track_shell_started("rasa")
    result = interpreter.parse(input)
    print(json_to_string(result))

    print(core_model)
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

    args.port = constants.DEFAULT_SERVER_PORT
    args.connector = "cmdline" # from shell.py
    args.model = "../../Softwareprojekt/rasa_test/models"
    rasa.cli.run.run(args)

model_path = "../../Softwareprojekt/rasa_test"
message = "hi there"
process_input(message, model_path)