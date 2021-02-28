
# imports################################################################
import logging

from rasa.cli.utils import get_validated_path
from rasa.model import get_model, get_model_subdirectories
from rasa.nlu.model import Interpreter
from rasa.shared.constants import DEFAULT_MODELS_PATH
from rasa.shared.utils.io import json_to_string
import operator

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import math
import re
import string
from nltk.tokenize import word_tokenize
from collections import defaultdict
from operator import itemgetter


# functions################################################################
def get_rasa_model(model_path: str):
    # Get model
    model = get_validated_path(model_path, "model", DEFAULT_MODELS_PATH)
    model_path = get_model(model)
    _, nlu_model = get_model_subdirectories(model_path)
    interpreter = Interpreter.load(nlu_model)
    return interpreter

def process_input(input: str, interpreter: Interpreter):
    # compute NLU result
    result = interpreter.parse(input)
    return result # , json_to_string(result)


def intent_prob(result):
    intent_cal_list = list()
    # TODO: think about whether it should be possible that we get predictions from empty string here  (which then cause there not be an key "intent_ranking" in result
    try:
        intent_rank = json_to_string(result["intent_ranking"])
        n = 1
        ir_split = intent_rank.split(",")
        for thing in ir_split:
            thing = thing.strip()

            if "name" in thing:  # thing.startswith('"name'):
                now_intent = thing.split(":")[1].strip(string.punctuation).strip().strip(string.punctuation).strip() #TODO: delete " and ' from string.punctuation
            elif thing.startswith('"confidence"'):
                now_confi = thing.split(":")[1].strip("}").strip(string.punctuation).strip().strip(
                    string.punctuation).strip()
                intent_cal_list.append((now_intent, now_confi))
    except KeyError:
        pass
    return intent_cal_list

def get_utterance_predictor_and_tokenizer(predictor="huggingtweets/ppredictors"):
    # predictor can either be "huggingtweets/ppredictors" or "gpt2"
    # start generator and tokenizer
    generator = AutoModelForCausalLM.from_pretrained(predictor, return_dict_in_generate=True, max_length=30)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return generator, tokenizer

def predict_with_score(input, generator, tokenizer, num_predictions=5):

    # make model and generate output
    input_ids = tokenizer(input, return_tensors="pt").input_ids # "pt" stands for pytorch; could also be "tf"
    generated_outputs = generator.generate(input_ids, do_sample=True, num_return_sequences=num_predictions, output_scores=True)

    # gen_sequences has shape [5, 15] TODO why 15????
    gen_sequences = generated_outputs.sequences[:, input_ids.shape[-1]:]

    # stack the logits generated at each step to a tensor and transform
    # logits to probs TODO: Why do we sometimes get an empty tensor here??
    probs = torch.stack(generated_outputs.scores, dim=1).softmax(-1)  # -> shape [5, 15, vocab_size]

    # now we need to collect the probability of the generated token
    # we need to add a dummy dim in the end to make gather work
    gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)
    #print("token probabilities: ", gen_probs)

    # now we can do all kinds of things with the probs
    # 1) the probs that exactly those sequences are generated again
    # those are normally going to be very small
    unique_prob_per_sequence = gen_probs.prod(-1)
    logging.info("unique probs:", unique_prob_per_sequence)
    if num_predictions==1:
        return [{"generated_text":tokenizer.decode(x), "score":y}
                for x,y in zip(gen_sequences, unique_prob_per_sequence)]

    # 2) normalize the probs over the three sequences
    normed_gen_probs = gen_probs / gen_probs.sum(0)
    unique_normed_prob_per_sequence = normed_gen_probs.prod(-1)
    logging.info("unique NORMED probs per sequence: ", unique_normed_prob_per_sequence)

    return [{"generated_text":tokenizer.decode(x), "score":y}
            for x,y in zip(gen_sequences, unique_normed_prob_per_sequence)]

def sigmoid(x):
    # actually quicker than pre-implemented sigmoid function, according to https://stackoverflow.com/questions/3985619/how-to-calculate-a-logistic-sigmoid-function-in-python
    return 1 / (1 + math.exp(-x))

def weight_by_utterance_probability(pred_score, scaling_weight_utterance_prediction_score):
    try:
        # tried sigmoid here (to keep total value below 1.0) but didn't find good values for other hyper-parameters
        return float(scaling_weight_utterance_prediction_score)/math.log(pred_score,10) # i.e. 10/num_decimal_zeroes for scaling_weight_utterance_prediction_score=-10
    except ValueError:
        return 1/1000 # TODO: double-check: this should be a very small number
    except ZeroDivisionError:
        return 1

def get_pred_text(input_so_far, predfull_text, predicted):
    #print("Full predicted utterance: ", predfull_text)
    punc = "[" + ".?!" + "]"  # string.punctuation
    allpunc = "[" + string.punctuation + "]"
    if predicted == "by_sentend":
        pred_text = re.split(punc, predfull_text)[0].strip()  # split by sentence end punc
    elif predicted == "by_allpunc":
        pred_text = re.split(allpunc, predfull_text)[0].strip()  # split by all punc
    elif predicted == "by_fulltext":
        pred_text = predfull_text.strip()  # use full text

    pred_text = " ".join(input_so_far + [pred_text]) # append the newly predicted text to what we already have as input
    pred_text = pred_text.replace("<|endoftext|>","") # str.removesuffix is only implemented in python>=3.9
    logging.info("prediction item: ", pred_text)

    return pred_text

def earliest_locking_time(trp_list):
    # Choose prediction at earliest point in time (i.e. shortest user utterance) and with highest score (i.e. prediction[1]).
    user_utterance, predicted_utterance, intent, score, pred_id = min(trp_list, key=(lambda x: (len(x[0]), -x[3])))
    return user_utterance, intent, score, predicted_utterance, pred_id

def last_locking_time(trp_list):
    # Choose prediction at latest point in time (i.e. longest user utterance) and with highest score (i.e. prediction[1]).
    user_utterance, predicted_utterance, intent, score, pred_id = max(trp_list, key=(lambda x: (len(x[0]), x[3])))
    return user_utterance, intent, score, predicted_utterance, pred_id

def best_locking_time(trp_list):
    # Choose prediction with best score (i.e. highest prediction[1])
    user_utterance, predicted_utterance, intent, score, pred_id = max(trp_list, key=(lambda x: x[3]))
    return user_utterance, intent, score, predicted_utterance, pred_id


def get_prediction(tokenized_msg, generator, tokenizer, rasa_model, threshold,
         update_weight_timesteps, predicted, scaling_weight_utterance_prediction_score,
         average=False, averaging_weight = 0.5, num_utterance_predictions = 5, utt_score_threshold=0.6, prediction_updates=True):
    num_updates = 0 # counter for at how many time steps a prediction passed the threshold
    cumu_msg = []  # cumulated message
    intent_dict = defaultdict(int)
    trp_list = list()

    for word in tokenized_msg:
        cumu_msg.append(word)  # take one more word from full msg
        out = " ".join(cumu_msg)
        logging.info("current utterance portion: ", out)

        _intent_dict = defaultdict(list)

        predict = predict_with_score(out, generator, tokenizer, num_predictions=num_utterance_predictions) # generate 5 text (tweet) predictions

        for item in predict:  # for each prediction text
            pred_score = item["score"] # score for utterance prediction
            utt_score = weight_by_utterance_probability(pred_score, scaling_weight_utterance_prediction_score)
            pred_text = get_pred_text(cumu_msg, item["generated_text"], predicted)
            logging.info("input to intent classifier: ", pred_text)

            # Get intents ranking for predicted utterance
            all_result = process_input(pred_text, rasa_model)  # classify intent
            # TODO: possibly double-check whether pred_text==""
            # or maybe empty string means "end of sentence", i.e. the gpt2 prediction starts with ". NextSentence"
            # TODO: break when the utterance prediction we use is "" ?
            item_intent_dis = intent_prob(all_result) # item_intent_dis = list of (intent,confi)

            for intent,confi in item_intent_dis:
                #score = float(confi) * utt_score # In Gervits they use utt_score as threshold instead of weight
                #weight = 0.5
                #score = float(confi) * (1-weight) + sigmoid(utt_score) * (weight)
                score = sigmoid(float(confi) + utt_score)
                """
                if sigmoid(utt_score) > utt_score_threshold:
                    score = float(confi)
                else:
                    score = 0.0
                """

                _intent_dict[intent].append((pred_text, score))

            #print(sorted(intent_dict.items(), key=operator.itemgetter(1), reverse=True))

        update_performed = False # flag whether a prediction passed the threshold at this time step; using flag because we want to count at how many time steps a prediction was performed and because trp_list can contain several elements per time step.

        # make final scores by averaging over the individual utterance predictions
        for intent, predictions in _intent_dict.items():
            # _intent_dict[intent] : list of (predicted_utterance, intent_score) tuples

            # get utterance and score TODO: score must be weighted by utterance weight
            highest_ranking_utterance, score = max(predictions, key=itemgetter(1))
            if average:
                # replace score by average over the scores of all utterance predictions at this time step
                average = sum([x[1] for x in predictions]) / num_utterance_predictions  # score = average_score
                score = average * (1- averaging_weight) + score * averaging_weight

            intent_dict[intent] = intent_dict[intent] * (
                        1 - update_weight_timesteps) + score * update_weight_timesteps  # in Gervits et al.: update_weight_timesteps=1.0
            if score >= threshold:
                update_performed = True
                trp_list.append((out, highest_ranking_utterance, intent, score, num_updates+1))

        if update_performed:
            num_updates += 1

        if trp_list:
            if prediction_updates:
                # Break if the highest scoring prediction in trp_list predicts the trp to be 3 or less tokens away
                user_utterance, intent, score, predicted_utterance, pred_id = best_locking_time(trp_list)
                if len(word_tokenize(predicted_utterance)) - len(word_tokenize(user_utterance)) < 4:
                    logging.info(f"Highest scoring prediction is {pred_id}th out of {num_updates} predictions.")
                    print(f"Highest scoring prediction is {pred_id}th out of {num_updates} predictions.")
                    for t in trp_list:
                        print(t)
                    break
                else:
                    print(len(word_tokenize(predicted_utterance)), len(word_tokenize(user_utterance)))
            else:
                # If we don't update predictions and trp_list is non-empty (i.e. at least one prediction has passed the threshold),
                # there's no need for further computation.
                break


    print("\n ------------------------- \n")

    # If threshold hasn't been reached,
    #   choose best intent after reading in the whole input
    if not trp_list:
        best_intent,best_score = max(intent_dict.items(), key=itemgetter(1))
        trp_list.append((out, pred_text, best_intent, best_score, num_updates))
        logging.info(f"Highest scoring prediction is 1st out of {num_updates} predictions.")
        print(f"Highest scoring prediction is 1st out of {num_updates} predictions.")
        print("default trp_list")
        print(trp_list)

    return trp_list

def main(tokenized_msg, generator, tokenizer, rasa_model, threshold,
         update_weight_timesteps, predicted, scaling_weight_utterance_prediction_score,
         average=False, averaging_weight=0.5, num_utterance_predictions=5,
         utt_score_threshold=0.6, prediction_updates=True):

    # 1. Get trp_list
    trp_list = get_prediction(tokenized_msg, generator, tokenizer, rasa_model, threshold,
         update_weight_timesteps, predicted, scaling_weight_utterance_prediction_score,
         average=average, averaging_weight=averaging_weight, num_utterance_predictions=num_utterance_predictions,
                   utt_score_threshold=utt_score_threshold, prediction_updates=prediction_updates)

    # 2. Extract return values from trp-list
    if prediction_updates:
        msg_at_locking_time, p_intent, score, p_utterance, pred_id = best_locking_time(trp_list)
    else:
        msg_at_locking_time, p_intent, score, p_utterance, pred_id = earliest_locking_time(trp_list)

    print("full message: ", " ".join(tokenized_msg))
    print("predicted utterance at locking time: ", p_utterance)
    print(f"user utterance up to locking time: {msg_at_locking_time}")
    print("top intent and score at locking time: ", p_intent, score)
    print("\n\n\n")
    #print("prediction list: ", trp_list)

    return msg_at_locking_time, p_intent, p_utterance


if __name__ == "__main__":

    # alter these################################################################

    '''
    # suggested test text (from switchboard sw00-0004) (trained with the rest of switchboard)
    [inform_pass]           They been trying these people now for twentytwo years ever since I was a child
    [inform_continue]       And here's this bum that didn't have a job
    [stalling]              I mean they're they're
    [agreement]             That may very well be
    [autoPositive]          Uh-huh
    [confirm]               Yeah
    [selfCorrection]        you would h  you would have
    [setQuestion]           Who's paying for that
    [checkQuestion]         huh
    [propositionalQuestion] because when you get the most heinous of crimes have you ever noticed you always get the most renowned defense attorney
    [answer]                you're talking to part of them that's paying for that <laughter>
    '''

    input_msg = "those three guys take the evidence go off figure it out and then come back and say whether you're guilty or not"  # inform_pass
    threshold = float(0.9)  # threshold
    predicted = "by_sentend"  # options: "by_sentend" or "by_allpunc" or "by_fulltext" for predicted text length
    update_weight_timesteps = 0.9
    scaling_weight_utterance_prediction_score = -17
    average = True
    averaging_weight= 0.8
    num_utterance_predictions = 5
    utt_score_threshold = 0.6
    update_predictions = True

    model_path = "../../Softwareprojekt/rasa_test/models"
    tokenized_msg = word_tokenize(input_msg)
    #print("tokens: ", tokenized_msg)

    # models
    generator, tokenizer = get_utterance_predictor_and_tokenizer(predictor="huggingtweets/ppredictors") # "huggingtweets/ppredictors" or "gpt2"
    model = get_rasa_model(model_path)

    main(tokenized_msg, generator, tokenizer, model, threshold,
             update_weight_timesteps, predicted, scaling_weight_utterance_prediction_score,
             average=average, averaging_weight=averaging_weight,
         num_utterance_predictions=num_utterance_predictions, utt_score_threshold=utt_score_threshold,
         prediction_updates=update_predictions)
