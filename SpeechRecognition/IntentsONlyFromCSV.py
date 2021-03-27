def extract_intents(file_out, data):
    pass

def cvs_to_dict(filename):
    with open(filename, encoding="utf-8") as csvfile:
        reader = list(csv.DictReader(csvfile, delimiter=';', dialect='excel'))
        max_fstext = 0
        total_tokens = 0
        total_turns = 1
        total_utterances = 0
        for rownum in range(len(reader)):
            total_utterances += 1
            # get the number of transitions from one speaker to the next:
            try:
                if not(reader[rownum]['Sender'] == reader[rownum + 1]['Sender']):
                    total_turns += 1
            except IndexError:
                pass

            # metadata gathering
            total_tokens += len(word_tokenize(reader[rownum]["FS text"]))
            if len(word_tokenize(reader[rownum]['FS text'])) > max_fstext:
                max_fstext = len(word_tokenize(reader[rownum]['FS text']))

            for key, value in reader[rownum].items():
                # TODO make list of allowed items instead of disallowed
                # if value and key != '﻿Markables' and key != 'Markables' and key != '\ufeffMarkables' and key != 'Sender' and key != 'Addressee' and key != 'Turn transcription'and key != 'FS text' and key != 'Comments' and key!= 'other':
                if value and (key == 'Task' or key == 'autoFeedback' or key == 'alloFeedback' or key == 'turnManagement'
                              or key == 'timeManagement' or key == 'ownCommunicationManagement'
                              or key == 'partnerCommunicationManagement' or key == 'discourseStructuring'
                              or key == 'socialObligationsManagement'):
                    spl = value.split()
                    reader[rownum][key] = spl[1]

            # find unannotated statements
            this_row = reader[rownum]
            # if not (this_row['Task'] or this_row['autoFeedback'] or this_row['timeManagement'] or
            #         this_row['ownCommunicationManagement'] or this_row['partnerCommunicationManagement'] or
            #         this_row['discourseStructuring'] or this_row['socialObligationsManagement']):
            #     print(this_row['FS text'], 'tM:', this_row['turnManagement'], 'comm:', this_row['Comments'], 'from:', filename)

            if reader[rownum]['Task']:
                task_label = reader[rownum]['Task']
                if 'inform' in task_label:
                    try:
                        if reader[rownum]['Sender'] == reader[rownum + 1]['Sender']:
                            reader[rownum]['Task'] = 'inform_continue'
                        else:
                            reader[rownum]['Task'] = 'inform_pass'
                    except IndexError:
                        reader[rownum]['Task'] = 'inform_end'
    print("Total turns: ", total_turns, filename)
    print("Total tokens: ", total_tokens, filename)
    print("Maximum Utterance Length in tokens: ", max_fstext, filename)
    print("Average tokens per utterance: ", total_tokens/total_utterances, filename)
    return reader



if __name__ == '__main__':
    # Examples/Tech Demo:
    # file = 'TRAINS-1-gold_standard-MultiTab-V21.csv'
    # file = 'sw00-0004_DiAML-MultiTab.csv'

    files = ['CorporaTrainingEval/TRAINS/TRAINS-1-gold_standard-MultiTab-V21.csv',
             'CorporaTrainingEval/TRAINS/TRAINS-2-gold_standard-MultiTab-V21.csv',
             'CorporaTrainingEval/TRAINS/TRAINS-3-gold_standard-MultiTab-V2.csv',
             'CorporaTrainingEval/Switchboard/sw00-0004_DiAML-MultiTab.xlsx.txt.csv',
             'CorporaTrainingEval/Switchboard/sw01-0105_DiAML-MultiTab.csv',
             'CorporaTrainingEval/Switchboard/sw02-0224_DiAML-MultiTab.csv',
             'CorporaTrainingEval/Switchboard/sw03-0304_DiAML-MultiTab1.csv',
             'CorporaTrainingEval/MapTask/q1ec5_DiAML-MultiTab.csv',
             'CorporaTrainingEval/MapTask/q1ec6_DiAML-MultiTab.csv',
             'CorporaTrainingEval/DBOX/diana_DiAML-MultiTab-4.csv',
             'CorporaTrainingEval/DBOX/eleanor_DiAML-MultiTab.csv',
             'CorporaTrainingEval/DBOX/rihanna_DiAML-MultiTab.csv',
             'CorporaTrainingEval/DBOX/venus_DiAML-MultiTab.csv',
             'CorporaTrainingEval/DBOX/washington_DiAML-MultiTab.csv',
             'unifiedCorpora/Switchboard/allSwitchboard.csv',
             'unifiedCorpora/DBOX/allDBOX.csv',
             'unifiedCorpora/TRAINS/allTRAINS.csv']

    # Trains

    # Switchboard

    # MapTask

    # DBOX

    for file in files:
        training_file_out = file[:-3] + 'yaml'
        story_file_out = file[:-3] + 'story'
        dev_file_out = file[:-3] + "_for_development.csv"
        eval_file_out = file[:-3] + "_for_eval.csv"

        data = cvs_to_dict(file)
        train_split = int(len(data) / 100 * 80)
        dev_split = int(len(data) / 100 * 90)
        train_data = data[:train_split]
        dev_data = data[train_split:dev_split]
        eval_data = data[dev_split:]
        # print(data[1].keys())

