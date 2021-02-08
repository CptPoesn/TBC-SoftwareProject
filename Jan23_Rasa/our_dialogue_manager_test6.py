
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
    generator = AutoModelForCausalLM.from_pretrained(predictor, return_dict_in_generate=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return generator, tokenizer

def predict_with_score(input, generator, tokenizer):

    # make model and generate output
    input_ids = tokenizer(input, return_tensors="pt").input_ids # "pt" stands for pytorch; could also be "tf"
    generated_outputs = generator.generate(input_ids, do_sample=True, num_return_sequences=5, output_scores=True)

    # gen_sequences has shape [5, 15] TODO why 15????
    gen_sequences = generated_outputs.sequences[:, input_ids.shape[-1]:]

    # stack the logits generated at each step to a tensor and transform
    # logits to probs
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
    # 2) normalize the probs over the three sequences
    normed_gen_probs = gen_probs / gen_probs.sum(0)
    unique_normed_prob_per_sequence = normed_gen_probs.prod(-1)
    print("unique NORMED prob per sequence: ", unique_normed_prob_per_sequence)

    return [{"generated_text":tokenizer.decode(x), "score":y}
            for x,y in zip(gen_sequences, unique_normed_prob_per_sequence)]


def weight_by_utterance_probability(pred_score, scaling_weight_utterance_prediction_score):
    # TODO (urgent): define meaningful heuristic!!!
    #return 1
    try:
        return float(scaling_weight_utterance_prediction_score)/math.log(pred_score,10) # i.e. 10/num_decimal_zeroes for scaling_weight_utterance_prediction_score=-10
    except ValueError:
        return 1/1000 # TODO: double-check: this should be a very small number


# def intent_cumulative(intent, confi):
# sorry havent put this part into function


# actual codes################################################################



# generate up to first sentence-end marker (. or ? or !)

def get_pred_text(input_so_far, predfull_text, predicted):
    print("predfull_text: ", predfull_text)
    punc = "[" + ".?!" + "]"  # string.punctuation
    allpunc = string.punctuation
    if predicted == "by_sentend":
        pred_text = re.split(punc, predfull_text)[0].strip()  # split by sentence end punc
    elif predicted == "by_allpunc":
        pred_text = predfull_text.split(allpunc)[0].strip()  # split by all punc
    elif predicted == "by_fulltext":
        pred_text = predfull_text.strip()  # use full text
    pred_text = " ".join(input_so_far + [pred_text]) # append the newly predicted text to what we already have as input
    print("prediction item: ", pred_text)

    return pred_text


def earliest_locking_time(trp_list):
    user_utterance, predicted_utterance, prediction = sorted(trp_list, key=(lambda x: len(x[0])))[0]
    intent, score = prediction
    return user_utterance, intent, predicted_utterance


def main(tokenized_msg, generator, tokenizer, rasa_model, threshold,
         update_weight_timesteps, predicted, scaling_weight_utterance_prediction_score):
    cumu_msg = []  # cumulated message
    intent_dict = dict()
    response_word_at_trp = ""
    response_intent_at_trp = (0, 0)
    trp_list = list()
    earliest_word = ""
    utterance_prediction_at_trp = ""

    for word in tokenized_msg:
        cumu_msg.append(word)  # take one more word from full msg
        out = " ".join(cumu_msg)
        print("current output: ", out)

        predict = predict_with_score(out, generator, tokenizer) # generate 5 text (tweet) predictions
        for item in predict:  # for each prediction text
            pred_text = get_pred_text(cumu_msg, item["generated_text"], predicted)

            # Get intents ranking

            all_result = process_input(pred_text, rasa_model)  # classify intent # TODO: possibly double-check whether pred_text==""
            # or maybe empty string means "end of sentence", i.e. the gpt2 prediction starts with ". NextSentence"
            # TODO: break when the utterance prediction we use is ""
            item_intent_dis = intent_prob(all_result)

            pred_score = item["score"] # score for utterance prediction

            for (intent, confi) in item_intent_dis:
                if intent in intent_dict:
                    intent_update = intent_dict[intent] * (1 - update_weight_timesteps) + float(confi) * update_weight_timesteps # TODO make variable
                    intent_update = intent_update * weight_by_utterance_probability(pred_score, scaling_weight_utterance_prediction_score) # TODO get meaningful weighting heuristic
                    intent_dict[intent] = intent_update
                    if intent_update >= threshold:
                        if earliest_word == "":
                            earliest_word = out
                        response_word_at_trp = out
                        utterance_prediction_at_trp = pred_text
                        response_intent_at_trp = sorted(intent_dict.items(), key=operator.itemgetter(1), reverse=True)[0]
                        trp_list.append((out, utterance_prediction_at_trp, response_intent_at_trp))
                else:
                    intent_dict[intent] = float(confi) * weight_by_utterance_probability(pred_score, scaling_weight_utterance_prediction_score)
            print(sorted(intent_dict.items(), key=operator.itemgetter(1), reverse=True))
        print("\n ------------------------- \n")


    # TODO: edit after Clara's update / @Clara: double-check
    if not response_word_at_trp:
        response_word_at_trp = out
    if response_intent_at_trp == (0,0):
        response_intent_at_trp = sorted(intent_dict.items(), key=operator.itemgetter(1), reverse=True)[0]
    if not utterance_prediction_at_trp:
        utterance_prediction_at_trp = pred_text
    if not trp_list:
        trp_list.append((response_word_at_trp, utterance_prediction_at_trp, response_intent_at_trp))

    print("full message: ", " ".join(tokenized_msg))
    print("earliest possible response point: ", response_word_at_trp)
    print("top intent and confi at earliest possible response point: ", response_intent_at_trp)
    print("trp list: ", trp_list)

    # extract return values from trp-list
    msg_at_locking_time, p_intent, p_utterance = earliest_locking_time(trp_list)

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

    input_msg = "I mean they're they're"  # input text
    threshold = float(0.8)  # threshold: 0.9/0.8 tested
    predicted = "by_sentend"  # options: "by_sentend" or "by_allpunc" or "by_fulltext" for predicted text length
    update_weight_timesteps = 0.5

    model_path = "../../Softwareprojekt/rasa_test/models"
    tokenized_msg = word_tokenize(input_msg)
    print("tokens: ", tokenized_msg)

    # models
    generator, tokenizer = get_utterance_predictor_and_tokenizer(predictor="huggingtweets/ppredictors")
    model = get_rasa_model(model_path)

    main(tokenized_msg, generator, tokenizer, model, threshold, update_weight_timesteps, predicted)
