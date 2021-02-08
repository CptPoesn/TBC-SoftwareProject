

import our_dialogue_manager as dm
from nltk.tokenize import word_tokenize
from collections import defaultdict
import csv
import numpy as np
import pandas

def get_prediction(utterance, model, generator, tokenizer):
	"""
	Runs our (i.e. Clara's) code for incremental prediction.
	Returns:
		intent : the predicted intent
		prediction_locking_time : number of tokens after which the prediction was locked with respect to real utterance length
		response_delivery_time : the predicted response time with respect to the real utterance length
	"""

	# hyper-parameters for tuning
	threshold = 0.8
	predicted = "by_allpunc"  # options: "by_sentend" or "by_allpunc" or "by_fulltext" for predicted text length
	update_weight_timesteps = 0.5 # how strongly the new intent ranking is weighted compared to the previous time step
	scaling_weight_utterance_prediction_score = -20 # the more negative, the stronger the score's influence

	"""
	run several experiments with different configuration: different weightings for incremental scores; 
	confidence thresholds; unique vs recurring values in training file; cutting predicted utterance at 
	different break points (punctuation, EOS, EOT, defined maximum length, ...); different utterance 
	prediction models (GPT2, PPredict, GPT3?); task-based vs chit-chat
	"""

	tokenized_msg = word_tokenize(utterance)
	print("tokens: ", tokenized_msg)


	msg_at_trp, predicted_intent, utterance_prediction_at_trp = \
		dm.main(tokenized_msg, generator, tokenizer, model, threshold,
				update_weight_timesteps, predicted, scaling_weight_utterance_prediction_score)

	# return values
	cumu_msg_at_trp = word_tokenize(msg_at_trp)
	tokenized_utterance_prediction = word_tokenize(utterance_prediction_at_trp)
	prediction_locking_time = len(cumu_msg_at_trp) - len(tokenized_msg)
	print(len(tokenized_utterance_prediction),tokenized_utterance_prediction)
	print(len(tokenized_msg), tokenized_msg)
	response_delivery_time = len(tokenized_utterance_prediction) - len(tokenized_msg)

	return predicted_intent, prediction_locking_time, response_delivery_time



# train model (manually or within script whichever is easier)
#TODO

# get models
model_path = "../../Softwareprojekt/rasa_test/models"
generator, tokenizer = dm.get_utterance_predictor_and_tokenizer(predictor="huggingtweets/ppredictors")
model = dm.get_rasa_model(model_path)

# read in corpus file as csv
confusion_matrix = defaultdict(lambda: defaultdict(int))
locking_times = []
response_times = []
corpus_file_name = "CorporaTrainingEval/Switchboard/sw02-0224_DiAML-MultiTab._for_development.csv"
with open(corpus_file_name, mode='r') as csv_file:
	csv_reader = csv.DictReader(csv_file)
	for row in csv_reader:
		p_intent, prediction_locking_time, response_delivery_time = \
			get_prediction(row["utterance"], model, generator, tokenizer)

		confusion_matrix[row["intent"]][p_intent] += 1
		locking_times.append(prediction_locking_time)
		response_times.append(response_delivery_time)


# get P,R,F1,Acc from confusion matrix
true_positives = float(sum([confusion_matrix[x][x] for x in confusion_matrix]))
print("true_positives: ", true_positives)
total_sum = float(sum([sum(confusion_matrix[x].values()) for x in confusion_matrix]))
print("total sum: ", total_sum)
accuracy = true_positives / total_sum
print("accuracy: ", accuracy)

"""
import pickle

with open('list_dump', 'wb') as fp:
	pickle.dump(locking_times, fp)
	pickle.dump(response_delivery_times, fp)
	pickle.dump(total_sum)
	pickle.dump(confusion_matrix)
"""

# get relevant measures for timing
print("locking times: ", locking_times)
print("average locking time: ", sum(locking_times)/total_sum, " +/- ", np.std(locking_times))
print("response delivery times: ", response_times)
print("average response time: ", sum(response_times) / total_sum, " +/- ", np.std(response_times))



# label-specific
"""
false_negatives = dict()
for gold in confusion_matrix:
	false_negatives[gold] = sum(confusion_matrix[gold].values()) - confusion_matrix[gold][gold]
false_positives = dict()
for gold in confusion_matrix:
	for pred in confusion_matrix:
		if gold != pred:
			false_positives[pred] += confusion_matrix[gold][pred]
"""

