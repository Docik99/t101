import csv

if __name__ == '__main__':
    names = ['audi.csv', 'hyundi.csv', 'bmw.csv', 'merc.csv', 'vauxhall.csv',
             'cclass.csv', 'skoda.csv', 'vw.csv', 'focus.csv', 'toyota.csv', 'ford.csv']
    files = []
    for name in names:
        files.append(f'{name}')

    for path in files:
        with open(path, "r") as file_in:
            with open(f"new/{path}", "w") as file_out:
                reader = csv.reader(file_in)
                writer = csv.writer(file_out)
                next(reader)

                for row in reader:
                    if int(row[4]) <= 100_000 and int(row[2]) <= 100_000:
                        writer.writerow(row)
exit(0)