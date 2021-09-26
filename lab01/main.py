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
        oper -= 1
        for operation in condition:
            graph = nx.MultiDiGraph()

            for element in condition[operation]:
                graph.add_edge(element, oper, log=operation)
                graph.add_edge(oper, element, log=None)

            if isinstance(result, list):
                graph.add_edge(oper, result[0], log=None)
            else:
                graph.add_edge(oper, result, log=None)
            graph_list.append(graph)

    union_graph = graph_list.pop(0)
    for g in graph_list:
        union_graph = nx.compose(union_graph, g)

    # for g in union_graph:
    #     for nbr in union_graph[g]:
    #         print(f"{g} ---> {nbr}")

    nx.draw(union_graph, with_labels=True)
    plt.show()

    return union_graph


def check_rule(graphs, facts):
    if isinstance(graphs, list):
        for graph in graphs:
            for fact in facts:
                for nbr in graph[fact]:
                    all_child = -1
                    fact_child = 0
                    new_fact = 0
                    for nbr2 in graph[nbr]:
                        all_child += 1
                        if nbr2 in facts:
                            fact_child += 1
                        else:
                            new_fact = nbr2
                    if fact_child == all_child:
                        facts.append(new_fact)

    else:
        for fact in facts:
            for nbr in graphs[fact]:
                all_child = -1
                fact_child = 0
                new_fact = 0
                for nbr2 in graphs[nbr]:
                    all_child += 1
                    if nbr2 in facts:
                        fact_child += 1
                    else:
                        new_fact = nbr2
                if fact_child == all_child:
                    facts.append(new_fact)
    return facts


if __name__ == '__main__':
    parsers = create_args()
    args = parsers.parse_args()
    rules = load_data(args.file)
    answer = check_rule(rules, [11,10,3,8,9])
    print(answer)
