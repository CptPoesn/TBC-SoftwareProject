from collections import defaultdict as dd

def writeToYAML(filename, dict):

    with open(filename, 'w') as f:
        f.write('nlu\n')
        for key, values in dict.items():
            f.write('- intent: ' + key.lower() + '\n')
            f.write('  examples:' + '\n')  # two spaces
            for ex in values:
                f.write('  - text: |\n')  # four spaces
                f.write('      ' + ex + '\n')  # six spaces

            f.write('\n')

if __name__ == '__main__':

    #convert all training utterances to yaml
    filename = 'train_set.txt'
    intentUtterances = dd(set)

    with open(filename) as f:
        for utt in f:
            speaker, utterance, label = utt.split('|')
            label = label[:-1]
            intentUtterances[label].add(utterance)
    print(intentUtterances.keys())
    writeToYAML('training.yaml', intentUtterances)

    #dev files:
    testFile = 'test_set.txt'





