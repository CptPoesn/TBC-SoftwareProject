
# imports################################################################

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

# test 1 only imports ###################################################
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
                now_intent = thing.split(":")[1].strip(string.punctuation).strip().strip(string.punctuation).strip()
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
    print("unique probs:", unique_prob_per_sequence)
    if num_predictions==1:
        return [{"generated_text":tokenizer.decode(x), "score":y}
                for x,y in zip(gen_sequences, unique_prob_per_sequence)]

    # 2) normalize the probs over the three sequences
    normed_gen_probs = gen_probs / gen_probs.sum(0)
    unique_normed_prob_per_sequence = normed_gen_probs.prod(-1)
    print("unique NORMED prob per sequence: ", unique_normed_prob_per_sequence)

    return [{"generated_text":tokenizer.decode(x), "score":y}
            for x,y in zip(gen_sequences, unique_normed_prob_per_sequence)]

def sigmoid(x):
    # actually quicker than pre-implemented, according to https://stackoverflow.com/questions/3985619/how-to-calculate-a-logistic-sigmoid-function-in-python
    return 1 / (1 + math.exp(-x))

def weight_by_utterance_probability(pred_score, scaling_weight_utterance_prediction_score):
    # TODO (urgent): define meaningful heuristic!!!
    #return 1
    try:
        # tried sigmoid here (to keep total value below 1.0) but didn't find good values for other hyper-parameters
        return float(scaling_weight_utterance_prediction_score)/math.log(pred_score,10) # i.e. 10/num_decimal_zeroes for scaling_weight_utterance_prediction_score=-10
    except ValueError:
        return 1/1000 # TODO: double-check: this should be a very small number
    except ZeroDivisionError:
        return 1


def get_pred_text(input_so_far, predfull_text, predicted):
    print("predfull_text: ", predfull_text)
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
    print("prediction item: ", pred_text)

    return pred_text

def earliest_locking_time(trp_list):
    user_utterance, predicted_utterance, prediction = min(trp_list, key=(lambda x: (len(x[0]), -x[2][1])))
    intent,score = prediction
    return user_utterance, intent, score, predicted_utterance


# def intent_cumulative(intent, confi):
# sorry havent put this part into function


# actual codes################################################################



# generate up to first sentence-end marker (. or ? or !)


def main(tokenized_msg, generator, tokenizer, rasa_model, threshold,
         update_weight_timesteps, predicted, scaling_weight_utterance_prediction_score,
         average=False, averaging_weight = 0.5, num_utterance_predictions = 5, utt_score_threshold=0.6): #TODO implement num_utt_pred
    cumu_msg = []  # cumulated message
    intent_dict = defaultdict(int)
    trp_list = list()

    for word in tokenized_msg:
        cumu_msg.append(word)  # take one more word from full msg
        out = " ".join(cumu_msg)
        print("current output: ", out)

        _intent_dict = defaultdict(list)

        predict = predict_with_score(out, generator, tokenizer, num_predictions=num_utterance_predictions) # generate 5 text (tweet) predictions
        # TODO: make the number of predictions a variable

        for item in predict:  # for each prediction text
            pred_score = item["score"] # score for utterance prediction
            utt_score = weight_by_utterance_probability(pred_score, scaling_weight_utterance_prediction_score)
            pred_text = get_pred_text(cumu_msg, item["generated_text"], predicted)
            print("pred text: ", pred_text)

            # Get intents ranking for predicted utterance
            all_result = process_input(pred_text, rasa_model)  # classify intent
            # TODO: possibly double-check whether pred_text==""
            # or maybe empty string means "end of sentence", i.e. the gpt2 prediction starts with ". NextSentence"
            # TODO: break when the utterance prediction we use is "" ?
            item_intent_dis = intent_prob(all_result) # item_intent_dis = list of (intent,confi)

            """
            proposed bug fix: 
            """
            for intent,confi in item_intent_dis:
                #score = float(confi) * utt_score # In Gervits they use utt_score as threshold instead of weight
                #weight = 0.5
                #score = float(confi) * (1-weight) + sigmoid(utt_score) * (weight)
                #score = sigmoid(float(confi) + utt_score)
                if sigmoid(utt_score) > utt_score_threshold:
                    score = float(confi)
                else:
                    score = 0.0

                _intent_dict[intent].append((pred_text, score))

            #print(sorted(intent_dict.items(), key=operator.itemgetter(1), reverse=True))

        # make average
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
                trp_list.append((out, highest_ranking_utterance, (intent, score)))

        # compute output at this time step
        if trp_list:
            break

        print("\n ------------------------- \n")

        """ bugged
            
            for (intent, confi) in item_intent_dis:
                if intent in intent_dict:
                    # add to variable
                    intent_update = intent_dict[intent] * (1 - update_weight_timesteps) + float(confi) * update_weight_timesteps # TODO make variable
                    intent_update = intent_update * weight_by_utterance_probability(pred_score, scaling_weight_utterance_prediction_score) # TODO get meaningful weighting heuristic
                    intent_dict[intent] = intent_update
                    if intent_update >= threshold:
                        if earliest_word == "":
                            earliest_word = out
                        response_word_at_locking_time = out
                        utterance_prediction_at_locking_time = pred_text
                        print("pred text saved: ", pred_text)
                        response_intent_at_locking_time = max(intent_dict.items(), key=operator.itemgetter(1) )
                        trp_list.append((out, utterance_prediction_at_locking_time, response_intent_at_locking_time))
                else:
                    intent_dict[intent] = float(confi) * weight_by_utterance_probability(pred_score, scaling_weight_utterance_prediction_score)
                    
        """


    # If threshold hasn't been reached,
    #   choose best intent after reading in the whole input
    if not trp_list:
        best_intent,best_score = max(intent_dict.items(), key=itemgetter(1))
        trp_list.append((out, pred_text, (best_intent, best_score) ))

    # extract return values from trp-list
    msg_at_locking_time, p_intent, score, p_utterance = earliest_locking_time(trp_list)
    
    print("\n ------------------------- \n")
    print("full message: ", " ".join(tokenized_msg))
    print("earliest possible response point: ", msg_at_locking_time)
    print("top intent and score at earliest possible response point: ", p_intent, score)
    print("trp list: ", trp_list)

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
    averaging_weight= 0.5
    num_utterance_predictions = 5
    utt_score_threshold = 0.6

    model_path = "../../Softwareprojekt/rasa_test/models"
    tokenized_msg = word_tokenize(input_msg)
    print("tokens: ", tokenized_msg)

    # models
    generator, tokenizer = get_utterance_predictor_and_tokenizer(predictor="huggingtweets/ppredictors")
    model = get_rasa_model(model_path)

    main(tokenized_msg, generator, tokenizer, model, threshold,
             update_weight_timesteps, predicted, scaling_weight_utterance_prediction_score,
             average=average, averaging_weight=averaging_weight,
         num_utterance_predictions=num_utterance_predictions, utt_score_threshold=utt_score_threshold)
