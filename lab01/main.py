"""
Создание экспертной системы

python3 main.py rules.json
"""
import json
import argparse
from time import time

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
    Загружает правила из json файла в ориентированный мультиграф
    :param data_file: правила записанные в json файле
    :return:
        graph: ориентированный мультиграф
    """
    r_json = open(data_file, 'r')
    rules = json.load(r_json)
    count_not_rules = 0
    oper_and = 0.0
    oper_not = 0
    graph = nx.MultiDiGraph()

    for rule in rules:
        condition = rule['if']
        result = rule['then']
        for operation in condition:

            for element in condition[operation]:
                if isinstance(result, list):
                    graph.add_edge(element, result[0], log=operation, dop=condition[operation])

                else:
                    graph.add_edge(element, result, log=operation, dop=condition[operation])

    return graph, count_not_rules


def check_rule(graph, facts, count_not_rules):
    """
    Проверка фактов по внесенным правилам
    :param graph: ориентированный мультиграф
    :param facts: список фактов
    :param count_not_rules: количество правил
    :return:
        facts: новый список фактов
    """
    facts = list(set(facts))

    for fact in facts:
        if fact in graph:
            print(graph[fact])
            keys = graph[fact].keys()
            print(list(keys))
            if len(keys) != 0:
                if graph[fact][keys[0]][0]['log'] == 'and':
                    if set(graph[fact][keys[0]][0]['dop']).issubset(facts):
                        for new_fact in graph[fact]:
                            if new_fact not in facts:
                                facts.append(new_fact)

                elif graph[fact][keys[0]][0]['log'] == 'or':
                    for new_fact in graph[fact]:
                        if new_fact not in facts:
                            facts.append(new_fact)

    nodes = list(graph.nodes())
    free_facts = set(nodes).difference(set(facts))
    free_facts = list(free_facts)
    for fact in free_facts:
        keys = list(graph[fact].keys())
        if len(keys) != 0:
            if graph[fact][keys[0]][0]['log'] == 'not':
                if not set(graph[fact][keys[0]][0]['dop']).issubset(facts):
                    for new_fact in graph[fact]:
                        if new_fact not in facts:
                            facts.append(new_fact)

            # for op in graph[fact]:
            #     if graph[fact][op][0]['log'] == 'and':
            #         new_fact = -1
            #         colvo_arg = True
            #         for nbr in graph[op]:
            #             if nbr not in facts:
            #                 if not graph.has_edge(nbr, op):
            #                     new_fact = nbr
            #                 else:
            #                     colvo_arg = False
            #                     break
            #         if colvo_arg and new_fact != -1:
            #             facts.append(new_fact)
            #
            #     elif graph[fact][op][0]['log'] == 'or':
            #         if op not in facts:
            #             facts.append(op)

    # for edge in range(count_not_rules * -1, 0):
    #     new_fact = -1
    #     colvo_arg = True
    #     for nbr in graph[edge]:
    #         if nbr not in facts:
    #             if not graph.has_edge(nbr, edge):
    #                 new_fact = nbr
    #         else:
    #             colvo_arg = False
    #             break
    #     if colvo_arg and new_fact != -1:
    #         facts.append(new_fact)

    return facts


if __name__ == '__main__':
    parsers = create_args()
    args = parsers.parse_args()

    time_start = time()

    rules, count_not_rules = load_data(args.file)
    nx.draw(rules, with_labels=True)
    plt.show()

    print(time() - time_start)

    time_start = time()
    answer = check_rule(rules, [8,9], count_not_rules)
    print(time() - time_start)

    print(answer)
