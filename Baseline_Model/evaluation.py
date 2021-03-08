import our_dialogue_manager as dm
from nltk.tokenize import word_tokenize
from collections import defaultdict
import csv
import numpy as np
from timeit import default_timer as timer

corpus_file_name = "C:/Users/schmi/Downloads/rasa-master/unifiedCorpora/BiggerSwitchboard/dev_reformat.txt"
model_path = "C:/Users/schmi/Softwareprojekt/biggerSwitchboard/models"
log_path = "eval_log_biggerSwitchboard.csv"
write_log = True
predictor = "huggingtweets/ppredictors" # huggingtweets/ppredictors or gpt2
vad_threshold = 2 # VAD = voice activation detector; threshold indicates after how many tokens of silence we deduce an end-of-utterance
response_generation_duration = 2  # how long it takes the NLG module to produce a response given a utterance and intent; duration measured in number of tokens


def get_prediction(utterance, model, generator, tokenizer):
	"""
	Runs the framework for incremental prediction.

	Returns:
		intent : the predicted intent
		prediction_locking_time : number of tokens after which the prediction was locked with respect to real utterance length
		predicted_trp : the predicted response time with respect to the real end of utterance
		fot_estimate: an estimate of the real floor offset transfer (assuming a response computation duration of 3 tokens,
						for detailed explanation, see below)
	"""

	# hyper-parameters for tuning
	threshold = 0.8
	predicted = "by_allpunc"  # options: "by_sentend" or "by_allpunc" or "by_fulltext" for predicted text length
	update_weight_timesteps = 0.9 # how strongly the new intent ranking is weighted compared to the previous time step; range [0,1]
	scaling_weight_utterance_prediction_score = -17 # the more negative, the stronger the score's influence
	averaging = True # whether we take an average over all predicted utterances or only the highest scoring utterance
	averaging_weight = 0.8 # how strongly the individual score of the most successful utterance is weighted against the avarage of all scores
	num_utterance_predictions = 5
	utt_pred_threshold = 0.6
	update_predictions = True

	# process utterance, process i.e. input
	tokenized_msg = word_tokenize(utterance)
	msg_at_trp, predicted_intent, utterance_prediction_at_trp = \
		dm.main(tokenized_msg, generator, tokenizer, model, threshold,
				update_weight_timesteps, predicted, scaling_weight_utterance_prediction_score,
				average=averaging, averaging_weight=averaging_weight,
				num_utterance_predictions=num_utterance_predictions, utt_score_threshold=utt_pred_threshold,
				prediction_updates=update_predictions, response_generation_duration=response_generation_duration)

	# return values
	cumu_msg_at_trp = word_tokenize(msg_at_trp)
	tokenized_utterance_prediction = word_tokenize(utterance_prediction_at_trp)
	prediction_locking_time = len(cumu_msg_at_trp) - len(tokenized_msg)
	#print(len(tokenized_utterance_prediction),tokenized_utterance_prediction)
	#print(len(tokenized_msg), tokenized_msg)
	predicted_trp = len(tokenized_utterance_prediction) - len(tokenized_msg)
	# According to Gervits et al., it takes the model about 1400ms to compute its response (including VAD).
	# This translates into something between 3-5 tokens in typical English dialogue. Let's assume 2 tokens
	# for VAD-threshold and 2 tokens for generation.
	# In that case, if the model locks its prediction at t=-1 (one token before the user has finished their utterance),
	# and predicts the TRP correctly (i.e. len(tokenized_utterance_prediction) - len(tokenized_msg) == 0),
	# then it can still respond at t=1lockingTime+computationDuration=-1+2 instead of the correctly predicted 0.
	# In general, if predictedTRP>(lockingTime+2): FOT=predictedTRP, else: FOT=lockingTime+2.
	#if predicted_trp > (prediction_locking_time + response_generation_duration) :
	#	fot_estimate = predicted_trp
	#else:
	#	fot_estimate = prediction_locking_time + response_generation_duration
	fot_estimate = max(predicted_trp, prediction_locking_time + response_generation_duration)

	return predicted_intent, prediction_locking_time, predicted_trp, fot_estimate


def main():
	if write_log:
		with open(log_path, "a", encoding="utf-8") as log:
			log.write("\t".join(["utterance", "intent", "predicted intent", "locking time", "predicted TRP",
								 "estimated FTO without VAD"]))
			log.write("\n")

	start = timer()

	# get models
	generator, tokenizer = dm.get_utterance_predictor_and_tokenizer(predictor=predictor)
	model = dm.get_rasa_model(model_path)

	# read in corpus file as csv
	confusion_matrix = defaultdict(lambda: defaultdict(int))
	locking_times = []
	trps = []
	fots = []
	with open(corpus_file_name, mode='r', encoding="utf-8") as csv_file:
		#csv_reader = csv.DictReader(csv_file)
		#for row in csv_reader:
		for line in csv_file:
			#p_intent, prediction_locking_time, predicted_trp, estimated_fot = \
				#get_prediction(row["utterance"], model, generator, tokenizer)

			#confusion_matrix[row["intent"]][p_intent] += 1
			line = line.strip().split("|")
			p_intent, prediction_locking_time, predicted_trp, estimated_fot = \
				get_prediction(line[0], model, generator, tokenizer)

			confusion_matrix[line[1]][p_intent] += 1
			locking_times.append(prediction_locking_time)
			trps.append(predicted_trp)
			fots.append(estimated_fot)

			if write_log:
				with open(log_path, "a", encoding="utf-8") as log:
					print([line[0], line[1], p_intent, prediction_locking_time, predicted_trp, estimated_fot])
					log.write("\t".join([line[0], line[1], p_intent, str(prediction_locking_time), str(predicted_trp), str(estimated_fot)]))
					log.write("\n")

	end = timer()

	print_metrics(confusion_matrix, locking_times, trps, fots, end-start)

def print_metrics(confusion_matrix, locking_times, trps, fots, execution_time):

	# get P,R,F1,Acc from confusion matrix
	true_positives = float(sum([confusion_matrix[x][x] for x in confusion_matrix]))
	print("true_positives: ", true_positives)
	total_sum = float(sum([sum(confusion_matrix[x].values()) for x in confusion_matrix]))
	print("total sum: ", total_sum)
	accuracy = true_positives / total_sum
	print("accuracy: ", accuracy)

	# get relevant measures for timing
	print("locking times: ", locking_times)
	print("average locking time: ", sum(locking_times)/total_sum, " +/- ", np.std(locking_times))
	print("predicted TRPs / response delivery times: ", trps)
	print("average predicted TRPs / response time: ", sum(trps) / total_sum, " +/- ", np.std(trps))
	print("estimated floor transfer offset: ", fots)
	print("average estimated floor transfer offset: ", sum(fots) / total_sum, " +/- ", np.std(fots))
		# Gervits: 	baseline (=non-incremental, without prediction of trp) response_time = 1.4 seconds (=generation-time + vad-threshold)
		# 			Switchboard mean syllable duration = 200 ms
		#			English: ~1.5 syllables / word
		#			very roughly, Gervits et al.'s baseline translates into 4.6 tokens after the EOU
		#			or: English: ~100-130 words/minute => baseline translates into 3-4 tokens

	# Assuming a VAD module, the response generation would be started at time t = end-of-utterance + vad-threshold.
	fots_with_vad = [min(vad_threshold+response_generation_duration, x) for x in fots] # x already includes response-generation-duration
	print(f"estimated floor transfer offset with VAD threshold of {vad_threshold}: {fots_with_vad}")
	print(f"average estimated floor transfer offset with a VAD threshold of {vad_threshold}: ", sum(fots_with_vad) / total_sum, " +/- ", np.std(fots_with_vad))

	print(f"Execution time: {execution_time} seconds")

def from_log(logfile):

	confusion_matrix = defaultdict(lambda: defaultdict(int))
	locking_times = []
	trps = []
	ftos = []

	with open(logfile, "r", encoding="utf-8") as log:
		for line in log:
			if not line.startswith("utterance\t"):

				utterance, g_intent, p_intent, prediction_locking_time, predicted_trp, estimated_fto = line.split("\t")
				prediction_locking_time = int(prediction_locking_time)
				predicted_trp = int(predicted_trp)
				estimated_fto = int(estimated_fto)

				confusion_matrix[g_intent][p_intent] += 1
				locking_times.append(prediction_locking_time)
				trps.append(predicted_trp)
				ftos.append(estimated_fto)

	print_metrics(confusion_matrix, locking_times, trps, ftos, "unknown")

if __name__ == "__main__":
	from_log("eval_log_biggerSwitchboard.csv")
	#main()


"""
_confusion_matrix = defaultdict(lambda: defaultdict(int))
for intent in confusion_matrix:
	if intent.startswith("inform"):
		for _intent in confusion_matrix[intent]:
			if _intent.startswith("inform"):
				_confusion_matrix["inform"]["inform"] += confusion_matrix[intent][_intent]
			else:
				_confusion_matrix["inform"][_intent] += confusion_matrix[intent][_intent]
	else:
		for _intent in confusion_matrix[intent]:
			if _intent.startswith("inform"):
				_confusion_matrix[intent]["inform"] += confusion_matrix[intent][_intent]
			else:
				_confusion_matrix[intent][_intent] += confusion_matrix[intent][_intent]
print("controlled for two types of inform")
true_positives = float(sum([_confusion_matrix[x][x] for x in _confusion_matrix]))
print("\ttrue_positives: ", true_positives)
total_sum = float(sum([sum(_confusion_matrix[x].values()) for x in _confusion_matrix]))
print("\ttotal sum: ", total_sum)
accuracy = true_positives / total_sum
print("\taccuracy: ", accuracy)
"""









# TODO label-specific evaluation
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

