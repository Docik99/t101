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
    count_rules = len(rules)
    oper = 0
    graph = nx.MultiDiGraph()

    for rule in rules:
        condition = rule['if']
        result = rule['then']
        oper -= 1
        for operation in condition:

            for element in condition[operation]:
                graph.add_edge(element, oper, log=operation)
                graph.add_edge(oper, element, log=None)

            if isinstance(result, list):
                graph.add_edge(oper, result[0], log=operation)
            else:
                graph.add_edge(oper, result, log=operation)

    return graph, count_rules


def check_rule(graph, facts, count_rules):
    """
    Проверка фактов по внесенным правилам
    :param graph: ориентированный мультиграф
    :param facts: список фактов
    :param count_rules: количество правил
    :return:
        facts: новый список фактов
    """

    for fact in facts:
        if fact in graph:
            for op in graph[fact]:
                new_fact = 0

                if graph[fact][op][0]['log'] == 'and':
                    all_child = -1
                    fact_child = 0
                    for nbr in graph[op]:
                        if graph[op][nbr][0]['log'] is not None:
                            all_child += 1
                            if nbr in facts:
                                fact_child += 1
                            else:
                                new_fact = nbr
                    if fact_child == all_child:
                        facts.append(new_fact)

                elif graph[fact][op][0]['log'] == 'or':
                    for nbr in graph[op]:
                        if graph[op][nbr][0]['log'] is not None:
                            if nbr not in facts:
                                if len(graph[nbr]) > 1:
                                    for nbr2 in graph[nbr]:
                                        if nbr2 != nbr:
                                            new_fact = nbr
                                            break
                                else:  # если узел конечен => это следствие из правила а не условие
                                    new_fact = nbr
                                facts.append(new_fact)

    app = False
    block_facts = []
    for edge in range(count_rules * -1, 0):
        for nbr in graph[edge]:
            if graph.has_edge(edge, nbr):
                if nbr not in facts:
                    if graph[edge][nbr][0]['log'] == 'not':
                        if isinstance(graph[nbr], list):
                            for dubl_op in graph[nbr]:
                                for dubl_el in graph[dubl_op]:
                                    if dubl_el not in facts:
                                        if graph[dubl_op][dubl_el][0]['log'] == 'not':
                                            if nbr not in block_facts:
                                                facts.append(dubl_el)
                                                app = True
                        else:
                            if nbr not in block_facts:
                                facts.append(nbr)
                                app = True
                    else:
                        block_facts.append(nbr)
                else:
                    break

    if app:
        return check_rule(graph, facts, count_rules)
    else:
        return facts


if __name__ == '__main__':
    parsers = create_args()
    args = parsers.parse_args()

    time_start = time()

    rules, count_rules = load_data(args.file)
    nx.draw(rules, with_labels=True)
    plt.show()

    print(time() - time_start)

    time_start = time()
    answer = check_rule(rules, [345, 479, 8, 9], count_rules)
    print(time() - time_start)

    print(answer)
