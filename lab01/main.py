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

            if operation == 'and':
                oper_and -= 0.1
                for element in condition[operation]:
                    graph.add_edge(element, oper_and, log=operation)
                    graph.add_edge(oper_and, element, log=None)

                if isinstance(result, list):
                    graph.add_edge(oper_and, result[0], log=operation)
                else:
                    graph.add_edge(oper_and, result, log=operation)

            if operation == 'not':
                count_not_rules += 1
                oper_not -= 1
                for element in condition[operation]:
                    graph.add_edge(element, oper_not, log=operation)
                    graph.add_edge(oper_not, element, log=None)

                if isinstance(result, list):
                    graph.add_edge(oper_not, result[0], log=operation)
                else:
                    graph.add_edge(oper_not, result, log=operation)

            elif operation == 'or':
                for element in condition[operation]:
                    if isinstance(result, list):
                        graph.add_edge(element, result[0], log=operation)

                    else:
                        graph.add_edge(element, result[0], log=operation)

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

    for fact in facts:
        if fact in graph:
            for op in graph[fact]:

                if graph[fact][op][0]['log'] == 'and':
                    new_fact = -1
                    colvo_arg = True
                    for nbr in graph[op]:
                        if nbr not in facts:
                            if not graph.has_edge(nbr, op):
                                new_fact = nbr
                            else:
                                colvo_arg = False
                                break
                    if colvo_arg and new_fact != -1:
                        facts.append(new_fact)

                elif graph[fact][op][0]['log'] == 'or':
                    if op not in facts:
                        facts.append(op)

    for edge in range(count_not_rules * -1, 0):
        for nbr in graph[edge]:
            if graph.has_edge(edge, nbr):
                if nbr not in facts:
                    if graph[edge][nbr][0]['log'] == 'not':
                        if isinstance(graph[nbr], list):
                            for dubl_op in graph[nbr]:
                                for dubl_el in graph[dubl_op]:
                                    if dubl_el not in facts:
                                        if graph[dubl_op][dubl_el][0]['log'] == 'not':
                                            facts.append(dubl_el)
                        else:
                            if graph[edge][nbr][0]['log'] == 'not':
                                facts.append(nbr)
                else:
                    break

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
    answer = check_rule(rules, [8, 9, 13, 99, 112, 114], count_not_rules)
    print(time() - time_start)

    print(answer)
