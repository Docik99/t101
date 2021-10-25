from random import choice, shuffle, randint
from time import time
import json
import networkx as nx


def generate_simple_rules(code_max, n_max, n_generate, log_oper_choice=["and", "or", "not"]):
    rules = []
    for j in range(0, n_generate):

        log_oper = choice(log_oper_choice)  # not means and-not (neither)
        if n_max < 2:
            n_max = 2
        n_items = randint(2, n_max)
        items = []
        for i in range(0, n_items):
            items.append(randint(1, code_max))
        rule = {
            'if': {
                log_oper: items
            },
            'then': code_max + j
        }
        # rule = f"{{ \"if\": {{ \"{log_oper}\": {items}, \"then\":{code_max + j}}}}}"
        rules.append(rule)
    shuffle(rules)
    return (rules)


def generate_stairway_rules(code_max, n_max, n_generate, log_oper_choice=["and", "or", "not"]):
    rules = []
    for j in range(0, n_generate):

        log_oper = choice(log_oper_choice)  # not means and-not (neither)
        if n_max < 2:
            n_max = 2
        n_items = randint(2, n_max)
        items = []
        for i in range(0, n_items):
            items.append(i + j)
        rule = {
            'if': {
                log_oper: items
            },
            'then': i + j + 1
        }
        rules.append(rule)
    shuffle(rules)
    return (rules)


def generate_ring_rules(code_max, n_max, n_generate, log_oper_choice=["and", "or", "not"]):
    rules = generate_stairway_rules(code_max, n_max, n_generate - 1, log_oper_choice)
    log_oper = choice(log_oper_choice)  # not means and-not (neither)
    if n_max < 2:
        n_max = 2
    n_items = randint(2, n_max)
    items = []
    for i in range(0, n_items):
        items.append(code_max - i)
    rule = {
        'if': {
            log_oper: items
        },
        'then': 0
    }
    rules.append(rule)
    shuffle(rules)
    return (rules)


def generate_random_rules(code_max, n_max, n_generate, log_oper_choice=["and", "or", "not"]):
    rules = []
    for j in range(0, n_generate):

        log_oper = choice(log_oper_choice)  # not means and-not (neither)
        if n_max < 2:
            n_max = 2
        n_items = randint(2, n_max)
        items = []
        for i in range(0, n_items):
            items.append(randint(1, code_max))
        rule = {
            'if': {
                log_oper: items
            },
            'then': randint(1, code_max)
        }
        rules.append(rule)
    shuffle(rules)
    return (rules)


def generate_seq_facts(M):
    facts = list(range(0, M))
    shuffle(facts)
    return facts


def generate_rand_facts(code_max, M):
    facts = []
    for i in range(0, M):
        facts.append(randint(0, code_max))
    return facts

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
        #print(len(facts))
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
        #print(len(facts)*(-1))
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



# samples:
# print(generate_simple_rules(100, 4, 10))
# print(generate_random_rules(100, 4, 10))
# print(generate_stairway_rules(100, 4, 10, ["or"]))
# print(generate_ring_rules(100, 4, 10, ["or"]))
# print(generate_rand_facts(100, 10))

# generate rules and facts and check time

N = 10000
M = 10
rules = generate_simple_rules(100, 4, N)
f_json = open(f"rules.json", "w")
json.dump(rules, f_json)
facts = generate_rand_facts(100, M)

# load and validate rules
# YOUR CODE HERE
time_start = time()
oper = 0
graph = nx.MultiDiGraph()
count_rules = len(rules)

for rule in rules:
    condition = rule['if']
    result = rule['then']
    oper -= 1
    for operation in condition:

        for element in condition[operation]:
            graph.add_edge(element, oper, log=operation)
            graph.add_edge(oper, element, log=None)

        if isinstance(result, list):
            graph.add_edge(oper, result[0], log=None)
        else:
            graph.add_edge(oper, result, log=None)

print("%d rules add in %f seconds" % (N, time() - time_start))

# check facts vs rules
time_start = time()

# YOUR CODE HERE
check_rule(graph, facts, count_rules)

print("%d facts validated vs %d rules in %f seconds" % (M, N, time() - time_start))
