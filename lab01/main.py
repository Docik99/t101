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
    f_json = open(data_file, 'r')
    rules = json.load(f_json)
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

    return graph


def check_rule(graphs, facts):
    """
    Проверка фактов по внесенным правилам
    :param graphs: ориентированный мультиграф
    :param facts: список фактов
    :return:
        facts: новый список фактов
    """
    for fact in facts:
        if fact in graphs:
            if isinstance(graphs[fact], list):
                for op in graphs[fact]:
                    new_fact = 0

                    if graphs[fact][op][0]['log'] == 'and':
                        all_child = -1
                        fact_child = 0
                        for nbr in graphs[op]:
                            all_child += 1
                            if nbr in facts:
                                fact_child += 1
                            else:
                                new_fact = nbr
                        if fact_child == all_child:
                            facts.append(new_fact)

                    elif graphs[fact][op][0]['log'] == 'or':
                        for nbr in graphs[op]:
                            if nbr not in facts:
                                if len(graphs[nbr]) > 1:
                                    for nbr2 in graphs[nbr]:
                                        if nbr2 != nbr:
                                            new_fact = nbr
                                            break
                                else:  # если узел конечен => это следствие из правила а не условие
                                    new_fact = nbr
                        facts.append(new_fact)

    count_rules = len(rules)
    for edge in range(count_rules * -1, 0):
        for nbr in graphs[edge]:
            if graphs.has_edge(edge, nbr):
                if nbr not in facts:
                    if graphs[edge][nbr][0]['log'] == 'not':
                        if isinstance(graphs[nbr], list):
                            for dubl_op in graphs[nbr]:
                                for dubl_el in graphs[dubl_op]:
                                    if dubl_el not in facts:
                                        if graphs[dubl_op][dubl_el][0]['log'] == 'not':
                                            facts.append(dubl_el)
                        else:
                            if graphs[edge][nbr][0]['log'] == 'not':
                                facts.append(nbr)
                else:
                    break

    return facts


if __name__ == '__main__':
    parsers = create_args()
    args = parsers.parse_args()

    time_start = time()

    rules = load_data(args.file)
    nx.draw(rules, with_labels=True)
    plt.show()

    print(time() - time_start)

    answer = check_rule(rules, [10, 9, 8, 11])

    print(answer)
