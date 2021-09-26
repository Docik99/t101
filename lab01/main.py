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
    graph_list = []
    f_json = open(data_file, 'r')
    rules = json.load(f_json)

    for rule in rules:
        condition = rule['if']
        result = rule['then']
        #element_list = []
        for operation in condition:
            graph = nx.DiGraph()
            if operation == 'and':
                oper = -1
            elif operation == 'or':
                oper = -2
            elif operation == 'not':
                oper = -3
            for element in condition[operation]:
                graph.add_edge(element, oper)
                #element_list.append(element)
            graph.add_edge(oper, result[0])
            graph_list.append(graph)
        #rules_dict.update({tuple(element_list): result})
    for g in graph_list:
        nx.draw(g, with_labels=True)
        plt.show()
    return graph_list


if __name__ == '__main__':
    parsers = create_args()
    args = parsers.parse_args()
    rules = load_data(args.file)
    print(rules)

