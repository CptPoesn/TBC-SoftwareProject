import csv

with open('sw00-0004_DiAML-MultiTab.csv') as csvfile:
    reader = list(csv.DictReader(csvfile, delimiter = ';', dialect='excel'))
    for rownum in range(len(reader)):
        if reader[rownum]['Task']:
            taskLabel = reader[rownum]['Task']
            if 'inform' in taskLabel:
                try:
                    if reader[rownum]['Sender'] == reader[rownum + 1]['Sender']:
                        reader[rownum]['Task'] = 'inform_continue'
                    else:
                        reader[rownum]['Task'] = 'inform_pass'
                except:
                    print('It happened, do something')
    for row in reader:
        print(row['Task'])
