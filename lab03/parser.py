"""
Parse csv file
"""
import csv
import numpy as np


def files_name():
    """
    Create correct path/name
    """
    names = ['audi.csv']
    files = []
    for name in names:
        files.append(f'archive_car/{name}')
    return files


def csv_reader(files):
    """
    Read a csv file
    """
    data = {}
    mileage = []
    price = []

    for path in files:

        with open(path, "r") as file:
            reader = csv.reader(file)
            next(reader)

            for row in reader:
                mileage.append(int(row[4]))
                price.append(int(row[2]))

        data['mileage'] = np.array(mileage)
        data['price'] = np.array(price)

    return data


if __name__ == "__main__":
    csv_path = "archive_car/audi.csv"
    with open(csv_path, "r") as f_obj:
        print(csv_reader(f_obj))