"""
Outline for the evaluation
"""


def get_prediction(utterance, model, weight=0.5) -> Tuple[str, int, int]:
	"""
	Runs our (Clara's) code for incremental prediction.

	Returns:
		intent : the predicted intent
		time_token : the token after which the prediction was locked
		time_estimate : the predicted response time = the predicted utterance length
	"""
	pass
	return intent, time_token, time_estimate



# train model (manually or within script whichever is easier)

# read in corpus file as csv

# for line in corpus_file:
	# p_intent, p_time_token, p_time_estimate = get_prediction(line["utterance"], model)
	# s.t. 	

	# confusion_matrix[line["intent"]][p_intent] += 1
	# predcition_times.append(len(line["utterance"]) - p_time_token)
	# resposne_times.append(p_time_estimate - len(line["utterance"]))


# get P,R,F1,Acc from confusion matrix
# get relevant measures for timing

"""
run several experiments with different configuration: different weightings for incremental scores; confidence thresholds; unique vs recurring values in training file; cutting predicted utterance at different break points (punctuation, EOS, EOT, defined maximum length, ...); different utterance prediction models (GPT2, PPredict, GPT3?); task-based vs chit-chat
"""
