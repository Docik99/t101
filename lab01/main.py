import json
import argparse


def create_args():
    """
    Создание аргументов командной строки
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
    Загрузка правил в словарь из файла

    :param data_file: файл с правилами
    :return: словарь с обработанными правилами
    """
    rules_dict = {}
    f_json = open(data_file, 'r')
    rules = json.load(f_json)

    # for rule in rules:
    #     condition = rule['if']
    #     result = rule['then']
    #     i = 0
    #     for part in condition['and']:
    #         i += 1
    #         element_list = []
    #         if isinstance(part, list):
    #             for element in part:
    #                 if isinstance(element, dict):
    #                     for not_element in element['not']:
    #                         element_list.append(not_element * -1)
    #                 else:
    #                     element_list.append(element)
    #         else:
    #             if isinstance(part, dict):
    #                 for not_element in part['not']:
    #                     element_list.append(not_element * -1)
    #             else:
    #                 element_list.append(part)
    #
    #         rules_dict.update({tuple(element_list): result})
    # print(rules_dict)
    # return rules_dict

    for rule in rules:
        condition = rule['if']
        result = rule['then']
        element_list = []
        if isinstance(condition, list):
            for element in condition:
                element_list.append(element)


def check_rule(check_file, rule):
    print(check_file.get(rule))


if __name__ == '__main__':
    parsers = create_args()
    args = parsers.parse_args()
    if args.operation == 'd':
        rules = load_data(args.file)
        check_rule(rules, tuple([5, 3]))
    # elif args.operation == 'c':
    #     check_rule(args.file)
