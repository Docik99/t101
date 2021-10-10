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


# samples:
print(generate_simple_rules(100, 4, 10))
print(generate_random_rules(100, 4, 10))
print(generate_stairway_rules(100, 4, 10, ["or"]))
print(generate_ring_rules(100, 4, 10, ["or"]))
print(generate_rand_facts(100, 10))

# generate rules and facts and check time

N = 100000
M = 100
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

        if operation == 'and' or operation == 'not':
            for element in condition[operation]:
                graph.add_edge(element, oper, log=operation)
                graph.add_edge(oper, element, log=None)

            if isinstance(result, list):
                graph.add_edge(oper, result[0], log=operation)
            else:
                graph.add_edge(oper, result, log=operation)

        elif operation == 'or':
            for element in condition[operation]:
                if isinstance(result, list):
                    graph.add_edge(element, result[0], log=operation)

                else:
                    graph.add_edge(element, result, log=operation)

print("%d rules add in %f seconds" % (N, time() - time_start))

# check facts vs rules
time_start = time()

# YOUR CODE HERE
for fact in facts:
    print(len(facts))
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

# for edge in range(count_rules * -1, 0):
#     for nbr in graph[edge]:
#         if graph.has_edge(edge, nbr):
#             if nbr not in facts:
#                 if graph[edge][nbr][0]['log'] == 'not':
#                     if isinstance(graph[nbr], list):
#                         for dubl_op in graph[nbr]:
#                             for dubl_el in graph[dubl_op]:
#                                 if dubl_el not in facts:
#                                     if graph[dubl_op][dubl_el][0]['log'] == 'not':
#                                         facts.append(dubl_el)
#                     else:
#                         if graph[edge][nbr][0]['log'] == 'not':
#                             facts.append(nbr)
#             else:
#                 break

print("%d facts validated vs %d rules in %f seconds" % (M, N, time() - time_start))
