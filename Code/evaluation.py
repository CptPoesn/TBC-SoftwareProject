import argparse

import our_dialogue_manager as dm
from nltk.tokenize import word_tokenize
from collections import defaultdict
import csv
import numpy as np
from timeit import default_timer as timer



def get_prediction(utterance, model, generator, tokenizer):
	"""
	Runs the framework for incremental prediction.

	Arguments:
		utterance: real user utterance
		model: interpreter model from rasa NLU
		generator: huggingface LM for text completion to predict user utterances in each iterative step
		tokenizer: huggingface tokenizer associated with the generator

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
	update_weight_timesteps = 1.0 # how strongly the new intent ranking is weighted compared to the previous time step; range [0,1]
	scaling_weight_utterance_prediction_score = -17 # the more negative, the stronger the score's influence
	averaging = True # whether we take an average over all predicted utterances or only the highest scoring utterance
	averaging_weight = 1.0 # how strongly the individual score of the most successful utterance is weighted against the avarage of all scores for the same intent
	num_utterance_predictions = 1
	utt_pred_threshold = 0.7
	update_predictions = False
	weight_utterance_score_relative_to_intent_confidence = 0.5

	# process utterance, process i.e. input
	tokenized_msg = word_tokenize(utterance)
	msg_at_trp, predicted_intent, score, utterance_prediction_at_trp = \
		dm.main(tokenized_msg, generator, tokenizer, model, threshold,
				update_weight_timesteps, predicted, scaling_weight_utterance_prediction_score,
				average=averaging, averaging_weight=averaging_weight,
				num_utterance_predictions=num_utterance_predictions, utt_score_threshold=utt_pred_threshold,
				prediction_updates=update_predictions, response_generation_duration=response_generation_duration,
				weight_utterance_score_relative_to_intent_confidence = weight_utterance_score_relative_to_intent_confidence)

	# return values
	cumu_msg_at_trp = word_tokenize(msg_at_trp)
	tokenized_utterance_prediction = word_tokenize(utterance_prediction_at_trp)
	prediction_locking_time = len(cumu_msg_at_trp) - len(tokenized_msg)
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

	return predicted_intent, prediction_locking_time, predicted_trp, fot_estimate, len(tokenized_msg)


def main(args):
	if args.write_log:
		with open(args.log_file, "a", encoding="utf-8") as log:
			log.write("\t".join(["utterance", "intent", "predicted intent", "locking time", "predicted TRP",
								 "estimated FTO without VAD"]))
			log.write("\n")

	start = timer()

	# get models
	generator, tokenizer = dm.get_utterance_predictor_and_tokenizer(predictor=predictor)
	model = dm.get_rasa_model(args.model)

	# read in corpus file as csv
	confusion_matrix = defaultdict(lambda: defaultdict(int))
	locking_times = []
	trps = []
	fots = []
	msg_lens = []
	with open(args.file_name, mode='r', encoding="utf-8") as csv_file:
		csv_reader = csv.DictReader(csv_file)
		for row in csv_reader:
			p_intent, prediction_locking_time, predicted_trp, estimated_fot, len_original_msg = \
				get_prediction(row["utterance"], model, generator, tokenizer)

			confusion_matrix[row["intent"]][p_intent] += 1
			locking_times.append(prediction_locking_time)
			trps.append(predicted_trp)
			fots.append(estimated_fot)
			msg_lens.append(len_original_msg)

			if args.write_log:
				with open(args.log_file, "a", encoding="utf-8") as log:
					print([row["utterance"], row["intent"], p_intent, prediction_locking_time, predicted_trp, estimated_fot], "\n\n")
					log.write("\t".join([row["utterance"], row["intent"], p_intent, str(prediction_locking_time), str(predicted_trp),
										 str(estimated_fot)]))
					log.write("\n")

	end = timer()
	print("message lengths: ", msg_lens)

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
	#print("estimated floor transfer offset: ", fots)
	#print("average estimated floor transfer offset: ", sum(fots) / total_sum, " +/- ", np.std(fots))
		# Gervits: 	baseline (=non-incremental, without prediction of trp) response_time = 1.4 seconds (=generation-time + vad-threshold)
		# 			Switchboard mean syllable duration = 200 ms
		#			English: ~1.5 syllables / word
		#			very roughly, Gervits et al.'s baseline translates into 4.6 tokens after the EOU
		#			or: English: ~100-130 words/minute => baseline translates into 3-4 tokens

	# Assuming a VAD module, the response generation would be started at time t = end-of-utterance + vad-threshold.
	fots_with_vad = [min(vad_threshold+response_generation_duration, x) for x in fots] # x already includes response-generation-duration
	print(f"estimated floor transfer offset with VAD threshold of {vad_threshold} and generation duration "
		  f"of {response_generation_duration}: {fots_with_vad}")
	print(f"average estimated floor transfer offset with a VAD threshold of {vad_threshold} and "
		  f"generation duration of {response_generation_duration}: ",
		  sum(fots_with_vad) / total_sum, " +/- ", np.std(fots_with_vad))

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
	predictor = "gpt2"  # "huggingtweets/ppredictors" or "gpt2"
	vad_threshold = 2  # VAD = voice activation detector; threshold indicates after how many tokens of silence we deduce an end-of-utterance
	response_generation_duration = 2  # how long it takes the NLG module to produce a response given a utterance and intent; duration measured in number of tokens

	parser = argparse.ArgumentParser(description='This script performs quantitative evaluation of the model our_dialog_manager')
	parser.add_argument("-g", "--gold-file", dest="file_name",
						help='The corpus file you wish to run. Format: csv file with columns "intent" and "utterance".')
	parser.add_argument("-m", "--model",
						help="""Path to the "/models" directory with a trained model.""")
	parser.add_argument("-l", "--log-file",
						help="""Path to a log file created in the previous run of the evaluation script or path to the 
						log file to be created in this run of the evalutaion.""")
	parser.add_argument("--write-log", action="store_true",
						help="""Writes gold intents and predicted intents into a file. Default = false.""")
	parser.add_argument("--from-log", action="store_true")
	args = parser.parse_args()

	if args.from_log:
		from_log(args.log_file)
	else:
		main(args)


