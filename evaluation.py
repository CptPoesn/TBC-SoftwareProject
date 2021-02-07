"""
Outline for the evaluation
"""
import our_dialogue_manager as dm
from nltk.tokenize import word_tokenize

def get_prediction(utterance, model) -> tuple[str, int, int]:
	"""
	Runs our (i.e. Clara's) code for incremental prediction.

	Returns:
		intent : the predicted intent
		prediction_locking_time : number of tokens after which the prediction was locked with respect to real utterance length
		response_delivery_time : the predicted response time with respect to the real utterance length
	"""

	# hyper-parameters for tuning
	threshold = 0.8
	predicted = "by_sentend"  # options: "by_sentend" or "by_allpunc" or "by_fulltext" for predicted text length
	update_weight_timesteps = 0.5 # how strongly the new intent ranking is weighted compared to the previous time step
	"""
	run several experiments with different configuration: different weightings for incremental scores; 
	confidence thresholds; unique vs recurring values in training file; cutting predicted utterance at 
	different break points (punctuation, EOS, EOT, defined maximum length, ...); different utterance 
	prediction models (GPT2, PPredict, GPT3?); task-based vs chit-chat
	"""

	tokenized_msg = word_tokenize(utterance)
	print("tokens: ", tokenized_msg)


	cumu_msg_at_trp, predicted_intent, utterance_prediction_at_trp = \
		dm.main(tokenized_msg, generator, tokenizer, model, threshold, update_weight_timesteps)

	# return values
	prediction_locking_time = len(cumu_msg_at_trp) - len(tokenized_msg)
	response_delivery_time = len(utterance_prediction_at_trp) - len(tokenized_msg)
	return predicted_intent, prediction_locking_time, response_delivery_time



# train model (manually or within script whichever is easier)
model_path = "../../Softwareprojekt/rasa_test/models"
generator, tokenizer = dm.get_utterance_predictor_and_tokenizer(predictor="huggingtweets/ppredictors")
model = dm.get_rasa_model(model_path)



# TODO @Ben ab hier
# read in corpus file as csv
# corpus_file =

# run model on corpus file
# for line in corpus_file:
	# p_intent, prediction_locking_time, response_delivery_time = get_prediction(line["utterance"], model)

	# confusion_matrix[line["intent"]][p_intent] += 1
	# locking_times.append(prediction_locking_time)
	# response_times.append(response_delivery_time)


# get P,R,F1,Acc from confusion matrix

# get relevant measures for timing (Durchschnitt?)


