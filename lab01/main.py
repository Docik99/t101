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
    oper = 0
    i = 0

    for rule in rules:
        condition = rule['if']
        result = rule['then']
        i -= 1
        for operation in condition:
            graph = nx.DiGraph()

            for element in condition[operation]:
                graph.add_edge(element, i, log=operation)

            graph.add_edge(i, result[0], log=None)
            graph_list.append(graph)

    union_graph = graph_list.pop(0)
    for g in graph_list:
        union_graph = nx.compose(union_graph, g)
    nx.draw(union_graph, with_labels=True)
    plt.show()
    return graph_list


if __name__ == '__main__':
    parsers = create_args()
    args = parsers.parse_args()
    rules = load_data(args.file)
    print(rules)

