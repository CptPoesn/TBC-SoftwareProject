

import our_dialogue_manager as dm
from nltk.tokenize import word_tokenize
from collections import defaultdict
import csv
import numpy as np
from timeit import default_timer as timer

def get_prediction(utterance, model, generator, tokenizer):
	"""
	Runs our (i.e. Clara's) code for incremental prediction.
	Returns:
		intent : the predicted intent
		prediction_locking_time : number of tokens after which the prediction was locked with respect to real utterance length
		response_delivery_time : the predicted response time with respect to the real utterance length
	"""

	# hyper-parameters for tuning
	threshold = 1.1
	predicted = "by_allpunc"  # options: "by_sentend" or "by_allpunc" or "by_fulltext" for predicted text length
	update_weight_timesteps = 0.9 # how strongly the new intent ranking is weighted compared to the previous time step; range [0,1]
	scaling_weight_utterance_prediction_score = -17 # the more negative, the stronger the score's influence
	averaging = True # whether we take an average over all predicted utterances or only the highest scoring utterance
	averaging_weight = 0.8 # how strongly the individual score of the most successful utterance is weighted against the avarage of all scores
	num_utterance_predictions = 5

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
				update_weight_timesteps, predicted, scaling_weight_utterance_prediction_score,
				average=averaging, averaging_weight=averaging_weight,
				num_utterance_predictions=num_utterance_predictions)

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
start = timer()

# get models
model_path = "../../Softwareprojekt/rasa_test/models"
generator, tokenizer = dm.get_utterance_predictor_and_tokenizer(predictor="huggingtweets/ppredictors")
model = dm.get_rasa_model(model_path)

# read in corpus file as csv
confusion_matrix = defaultdict(lambda: defaultdict(int))
locking_times = []
response_times = []
corpus_file_name = "C:/Users/schmi/Downloads/rasa-master/unifiedCorpora/allSwitchboard._for_development.csv"
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

end = timer()
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
	# Gervits: 	baseline (=without prediction of trp) response_time = 1.4 seconds
	# 			Switchboard mean syllable duration = 200 ms
	#			English: ~1.5 syllables / word
	#			very roughly, Gervits et al.'s baseline translates into 4.6 tokens after the EOU
	#			or: English: ~100-130 words/minute => baseline translates into 3-4 tokens
print("Execution time: ", start-end)




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

