from random import choice, shuffle, randint
from time import time
import json
import networkx as nx
from numba import njit


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
M = 30
rules = generate_simple_rules(100, 4, N)
f_json = open(f"rules.json", "w")
json.dump(rules, f_json)
facts = generate_rand_facts(100, M)

# load and validate rules
# YOUR CODE HERE
time_start = time()
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

print("%d rules add in %f seconds" % (N, time() - time_start))

# check facts vs rules
time_start = time()

# YOUR CODE HERE
facts = list(set(facts))
for fact in facts:
    print(len(facts))
    if fact in graph:
        keys = list(graph[fact].keys())
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

# nodes = list(graph.nodes())
# free_facts = set(nodes).difference(set(facts))
# free_facts = list(free_facts)
# for fact in free_facts:
#     keys = list(graph[fact].keys())
#     if len(keys) != 0:
#         if graph[fact][keys[0]][0]['log'] == 'not':
#             if not set(graph[fact][keys[0]][0]['dop']).issubset(facts):
#                 for new_fact in graph[fact]:
#                     if new_fact not in facts:
#                         facts.append(new_fact)

print("%d facts validated vs %d rules in %f seconds" % (M, N, time() - time_start))
