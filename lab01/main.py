import json
import argparse

rules = []

def create_args():
    """Создание аргументов командной строки

    Возвращаемые значения:
        parser: парсер введенных аргументов

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'operation',
        help='input "d" for download rules.json or "c" for check rules.json',
        type=str,
    )
    parser.add_argument(
        'file',
        help='input way to file',
        type=str,
    )
    return parser


def load_data(data_file):
    """

    :param data_file:
    :return:
    """
    rules_dict = {}
    f_json = open(data_file, 'r')
    rules = json.load(f_json)
    print(rules)
    for rule in rules:
        condition = rule['if']
        result = rule['then']
        i = 0
        for part in condition['and']:
            i += 1
            element_list = []
            if isinstance(part, list):
                for element in part:
                    if isinstance(element, dict):
                        for not_element in element['not']:
                            element_list.append(not_element * -1)
                            print(f"{not_element * -1} not_element part{i}")
                    else:
                        element_list.append(element)
                        print(f"{element} element part{i}")
            else:
                element_list.append(part)
                print(f"{part} part{i}")
            print(element_list)


if __name__ == '__main__':
    parsers = create_args()
    args = parsers.parse_args()
    load_data(args.file)
