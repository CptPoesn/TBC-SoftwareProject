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
    model = get_validated_path(model_path, "model", DEFAULT_MODELS_PATH)
    model_path = get_model(model)
    _, nlu_model = get_model_subdirectories(model_path)
    interpreter = Interpreter.load(nlu_model)

    # compute NLU result
    result = interpreter.parse(input)
    # print out result
    print(json_to_string(result))



model_path = "../../Softwareprojekt/rasa_test/models"
message = "hi there"
process_input(message, model_path)
