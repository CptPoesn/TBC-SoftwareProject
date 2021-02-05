# predictor = PPredictor
# from https://huggingface.co/huggingtweets/ppredictors

# new in test 5:
#   1. formula of adding up confidence at each time step: previous_all*0.5 + now*0.5
#   2. "TRP": as all points the threshold is exceeded


#alter these################################################################

'''
# suggested test text
[agreement] I think that's correct / That may very well be
[inform_continue] actually I lived over in Europe for a couple of years
[stalling] Well you know it's it's 
[propositional question] do you have any friends that have children
'''

input_msg = "That may very well be"    #input text
threshold = float(0.8)      # threshold: 0.9/0.8 tested
predicted = "by_sentend"    # options: "by_sentend" or "by_allpunc" or "by_fulltext" for predicted text length

trp_list = list()
earliest_word = ""
response_word_at_trp = ""
response_intent_at_trp = (0,0)

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
import operator

#test 1 only imports ###################################################
import re
import string
import nltk
from nltk.tokenize import word_tokenize

from transformers import pipeline
generator = pipeline('text-generation',
                     model='huggingtweets/ppredictors')

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
    #print(json_to_string(result))
    return result, json_to_string(result)
    
                
def intent_prob(result):
    intent_cal_list = list()
    for item in result:
        if item == "intent_ranking":
            intent_rank = json_to_string(result[item])
            n = 1
            ir_split = intent_rank.split(",")
            for thing in ir_split:
                thing = thing.strip()
                
                if "name" in thing:#thing.startswith('"name'):
                    now_intent = thing.split(":")[1].strip(string.punctuation).strip().strip(string.punctuation).strip()
                elif thing.startswith('"confidence"'):
                    now_confi = thing.split(":")[1].strip("}").strip(string.punctuation).strip().strip(string.punctuation).strip()
                    intent_cal_list.append((now_intent, now_confi))
    return intent_cal_list

#def intent_cumulative(intent, confi):
# sorry havent put this part into function


#actual codes################################################################

model_path = "../../Softwareprojekt/rasa_test/models"
message = input_msg
tokenized_msg = word_tokenize(message)
cumu_msg = []   #cumulated message
intent_dict = {}

print("tokens: ", tokenized_msg)

intent_dict = dict()


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
        allpunc = string.punctuation
        if predicted == "by_sentend":
            pred_text = re.split(punc, predfull_text)[0].strip()    # split by sentence end punc
        elif predicted == "by_allpunc":
            pred_text = predfull_text.split(allpunc)[0].strip()    # split by all punc
        elif predicted == "by_fulltext":
            pred_text = predfull_text.strip()       # use full text
        #print(pred_text)
        print("prediction item: ", pred_text)
        all_result, result_string = process_input(pred_text, model_path)    # classify intent
        #print(all_result)
        item_intent_dis = intent_prob(all_result)
        #ranked_intent = item_intent_dis.sort(key = lambda x: x[1])
        #ranked_intent = sorted(intent_dict.items(), key=operator.itemgetter(1), reverse=True)[0]  #sorted(item_intent_dis, key = lambda x: x[1], reverse=True)
        #print("ranked intent: ", ranked_intent)
        #if len(item_intent_dis) != 0:
        #    now_top_intent = ranked_intent[0]
        for (intent, confi) in item_intent_dis:
            if intent in intent_dict:
                intent_update = intent_dict[intent]*0.5 + float(confi)*0.5
                intent_dict[intent] = intent_update
                if intent_update >= threshold:
                    if earliest_word == "":
                        earliest_word = out
                    response_word_at_trp = out
                    response_intent_at_trp = sorted(intent_dict.items(), key=operator.itemgetter(1), reverse=True)[0]
                    trp_list.append((response_word_at_trp, response_intent_at_trp))
            else:
                intent_dict[intent] = float(confi)
            #print("intent: ", intent, "\tconfi: ", confi)
            #print('--------')
        print(sorted(intent_dict.items(), key=operator.itemgetter(1), reverse=True))
    print("\n ------------------------- \n")

print("full message: ", input_msg)
print("earliest possible response point: ", response_word_at_trp)
print("top intent and confi at earliest possible response point: ", response_intent_at_trp)
print("trp list: ", trp_list)

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
        all_result = process_input(pred_text, model_path)    # classify intent
        print(all_result)
    print("\n ------------------------- \n")
'''

# only intent classification
'''
for word in tokenized_msg:
    cumu_msg.append(word)       # take one more word from full msg
    out = " ".join(cumu_msg)
    print("current output: ", out)
    #predict = generator(out, num_return_sequences=5)    # generate 5 text (tweet) predictions 
    #print(predict)
    all_result = process_input(pred_text, model_path)    # classify intent
    print(all_result)
    print("\n ------------------------- \n")
'''
