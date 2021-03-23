# predictor = GPT-2
# from https://huggingface.co/gpt2

#imports################################################################

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

####################################################3
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from transformers import pipeline

#test 2 only imports ###################################################
from transformers import set_seed
generator = pipeline('text-generation',
                     model='gpt2')
set_seed(42)

#functions################################################################

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


#actual codes################################################################

model_path = "../../Softwareprojekt/rasa_test/models"
message = "That’s good to hear" #"I don't know"
tokenized_msg = word_tokenize(message)
cumu_msg = []   #cumulated message

print("tokens: ", tokenized_msg)


# generate up to first sentence-end marker (. or ? or !)

for word in tokenized_msg:
    cumu_msg.append(word)       # take one more word from full msg
    out = " ".join(cumu_msg)
    print("current output: ", out)
    predict = generator(out, num_return_sequences=5)    # generate 5 text (tweet) predictions 
    #print(predict)
    for item in predict:    # for each prediction text
        predfull_text = item["generated_text"]
        punc = "[" + ".?!" + "]"   #string.punctuation
        pred_text = re.split(punc, predfull_text)[0].strip()
        #print(pred_text)
        print("prediction item: ", pred_text)
        process_input(pred_text, model_path)    # classify intent
    print("\n ------------------------- \n")


#backup
# generate full prdiction
'''
for word in tokenized_msg:
    cumu_msg.append(word)       # take one more word from full msg
    out = " ".join(cumu_msg)
    print("current output: ", out)
    predict = generator(out, num_return_sequences=5)    # generate 5 text (tweet) predictions 
    #print(predict)
    for item in predict:    # for each prediction text
        predfull_text = item["generated_text"]
        punc = "[" + ".?!" + "]"   #string.punctuation
        pred_text = predfull_text.strip()
        #print(pred_text)
        print("prediction item: ", pred_text)
        process_input(pred_text, model_path)    # classify intent
    print("\n ------------------------- \n")
'''

