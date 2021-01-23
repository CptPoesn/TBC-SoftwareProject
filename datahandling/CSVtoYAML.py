import csv
from collections import defaultdict as dd


def cvs_to_dict(filename):
    with open(filename) as csvfile:
        reader = list(csv.DictReader(csvfile, delimiter=';', dialect='excel'))
        for rownum in range(len(reader)):
            for key, value in reader[rownum].items():
                if value and key != 'Markables' and key != 'Sender' and key != 'Addressee' and key != 'Turn ' \
                                                                                                      'transcription' \
                        and key != 'FS text':
                    reader[rownum][key] = value.split()[1]

            if reader[rownum]['Task']:
                task_label = reader[rownum]['Task']
                if 'inform' in task_label:
                    try:
                        if reader[rownum]['Sender'] == reader[rownum + 1]['Sender']:
                            reader[rownum]['Task'] = 'inform_continue'
                        else:
                            reader[rownum]['Task'] = 'inform_pass'
                    except:
                        print('Catch error that inform is last label')
    return reader


def dict_to_yaml(filename_out, data_dict):
    yaml_dict = dd(list)
    for row in data_dict:
        for key, value in row.items():
            if value and key != 'Markables' and key != 'Sender' \
                    and key != 'Addressee' and key != 'Turn transcription' and key != 'FS text':
                yaml_dict[value].append(row['FS text'])
    with open(filename_out, 'w') as f:
        f.write('nlu:\n')  # no spaces
        for key, values in yaml_dict.items():
            f.write('  - intent: ' + key + '\n')  # two spaces
            f.write('  examples:' + '\n')  # two spaces

            for ex in values:
                f.write('    - text: |\n')  # four spaces
                f.write('      ' + ex + '\n')  # six spaces

            f.write('\n')

    # for row in reader:
    #     str = ''
    #     i = 0
    #     for key, value in row.items():
    #         if value and key != 'Markables' and key != 'Sender' and key != 'Addressee' and key != 'Turn transcription' and key != 'FS text':
    #             i += 1
    #             str += key + ': ' + value + ', '
    #     print(str, i)


if __name__ == '__main__':
    file = 'sw00-0004_DiAML-MultiTab.csv'
    file_out = file[:-3] + 'yaml'

    data = cvs_to_dict(file)

    print(data[1].keys())

    dict_to_yaml(file_out, data)
