import json
import argparse
import networkx as nx
import matplotlib.pyplot as plt



def create_args():
    """
    Создание аргументов командной строки
    """
    parser = argparse.ArgumentParser()
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

    for rule in rules:
        condition = rule['if']
        result = rule['then']
        element_list = []
        graph = nx.DiGraph()
        for operation in condition:
            if operation == 'and':
                oper = -1
            elif operation == 'or':
                oper = -2
            elif operation == 'not':
                oper = -3
            for element in condition[operation]:
                graph.add_edge(element, oper)
                element_list.append(element)
            graph.add_edge(oper, result[0])
        rules_dict.update({tuple(element_list): result})
        # nx.draw(graph)
        # plt.show()
    return rules_dict


if __name__ == '__main__':
    parsers = create_args()
    args = parsers.parse_args()
    rules = load_data(args.file)
    print(rules)

