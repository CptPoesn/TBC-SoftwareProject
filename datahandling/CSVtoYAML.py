import csv
from collections import defaultdict as dd


def cvs_to_dict(filename):
    with open(filename, encoding="utf-8") as csvfile:
        reader = list(csv.DictReader(csvfile, delimiter=';', dialect='excel'))
        for rownum in range(len(reader)):
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
            if not(this_row['Task'] or this_row['autoFeedback'] or this_row['timeManagement'] or
                   this_row['ownCommunicationManagement'] or this_row['partnerCommunicationManagement'] or
                   this_row['discourseStructuring'] or this_row['socialObligationsManagement']):
                print(this_row['FS text'], 'tM:', this_row['turnManagement'], 'comm:', this_row['Comments'])


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
    return reader


def dict_to_yaml(filename_out, data_dict):
    # yaml_dict = dd(list)
    yaml_dict = dd(set)
    for row in data_dict:

        for key, value in row.items():
            # skips lines that aren't annotated
            if value and (key == 'Task' or key == 'autoFeedback' or key == 'alloFeedback' or key == 'turnManagement'
                          or key == 'timeManagement' or key == 'ownCommunicationManagement'
                          or key == 'partnerCommunicationManagement' or key == 'discourseStructuring'
                          or key == 'socialObligationsManagement'):
                if row['turnManagement']:
                    if value == row['turnManagement']:
                        continue
                    # yaml_dict[value].append(row['FS text'] + '\n    metadata:\n      turnManagement: '
                    #                         + row['turnManagement'])
                    yaml_dict[value].add(row['FS text'] + '\n    metadata:\n      turnManagement: '
                                            + row['turnManagement'])

                else:
                    # yaml_dict[value].append(row['FS text'])
                    yaml_dict[value].add(row['FS text'])
    with open(filename_out, 'w') as f:
        f.write('nlu:\n')  # no spaces
        for key, values in yaml_dict.items():
            if key == 'turnKeep':
                continue
            f.write('- intent: ' + key + '\n')  # no spaces
            f.write('  examples:' + '\n')  # two spaces

            for ex in values:
                f.write('  - text: |\n')  # four spaces
                f.write('      ' + ex + '\n')  # six spaces

            f.write('\n')

    # for row in reader:
    #     str = ''
    #     i = 0
    #     for key, value in row.items():
    #         if value and key != 'Markables' and key != 'Sender' and key != 'Addressee' and \
    #         key != 'Turn transcription' \and key != 'FS text':
    #             i += 1
    #             str += key + ': ' + value + ', '
    #     print(str, i)


if __name__ == '__main__':
    # Examples/Tech Demo:
    # file = 'TRAINS-1-gold_standard-MultiTab-V21.csv'
    # file = 'sw00-0004_DiAML-MultiTab.csv'

    files = []

    #Trains
    files.append('csv/TRAINS/TRAINS-1-gold_standard-MultiTab-V21.csv')
    files.append('csv/TRAINS/TRAINS-2-gold_standard-MultiTab-V21.csv')
    files.append('csv/TRAINS/TRAINS-3-gold_standard-MultiTab-V2.csv')

    #Switchboard
    files.append('csv/Switchboard/sw00-0004_DiAML-MultiTab.xlsx.txt.csv')
    files.append('csv/Switchboard/sw01-0105_DiAML-MultiTab.csv')
    files.append('csv/Switchboard/sw02-0224_DiAML-MultiTab.csv')
    files.append('csv/Switchboard/sw03-0304_DiAML-MultiTab1.csv')

    #MapTask
    files.append('csv/MapTask/q1ec5_DiAML-MultiTab.csv')
    files.append('csv/MapTask/q1ec6_DiAML-MultiTab.csv')

    #DBOX
    files.append('csv/DBOX/diana_DiAML-MultiTab-4.csv')
    files.append('csv/DBOX/eleanor_DiAML-MultiTab.csv')
    files.append('csv/DBOX/rihanna_DiAML-MultiTab.csv')
    files.append('csv/DBOX/venus_DiAML-MultiTab.csv')
    files.append('csv/DBOX/washington_DiAML-MultiTab.csv')


    for file in files:
        file_out = file[:-3] + 'yaml'

        data = cvs_to_dict(file)

        # print(data[1].keys())

        dict_to_yaml(file_out, data)
