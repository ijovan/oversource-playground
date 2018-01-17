import csv

with open('corrected.csv','w') as correct:
    writer = csv.writer(correct, dialect = 'excel')
    with open('../Pop/output/questions.csv', 'rb') as mycsv:
        reader = csv.reader( (line.replace('\0','') for line in mycsv) )
        for row in reader:
            writer.writerow(row)
