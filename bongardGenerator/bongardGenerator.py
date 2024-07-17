import random
import pandas as pd

shapes = ['triangle', 'square', 'circle']
configurations = ['up', 'down']
relations = ['in']

class BongardExample:
    def __init__(self, index=None):
        self.index = index
        self.objects = {}  # Key is the object index, value is the object
        self.relations = []  # List of relation statements
        self.classification = None

    def __str__(self):
        string = f'bongard({self.index}, {self.classification}).\n'
        for obj in self.objects.values():
            string += f'{obj.shape}({self.index}, o{obj.index}).\n'
        for rel in self.relations:
            string += f'{rel[0]}({self.index}, o{rel[1].index}, o{rel[2].index}).\n'
        return string

    def add_object(self, obj):
        self.objects[obj.index] = obj

    def add_relation(self, relation):
        self.relations.append(relation)

    def get_object(self, index):
        return self.objects[index]

    def check_classification(self, rule):
        for obj in rule.objects.values():
            if obj.shape not in [x.shape for x in self.objects.values()]:
                self.classification = 'neg'
                return
        relation_shapes = [('in', x.shape, y.shape) for (_, x, y) in self.relations]
        for rel in rule.relations:
            if ('in', rel[1].shape, rel[2].shape) not in relation_shapes:
                self.classification = 'neg'
                return
        self.classification = 'pos'

    def __repr__(self):
        return self.__str__()


class Object:
    def __init__(self, shape, index, config=None):
        self.shape = shape
        self.index = index
        self.config = config

    def __repr__(self):
        return f'{self.shape}({self.index})'

    def get_object_id(self):
        return f"o{self.index}"


def generate_rule(rule_complexity):
    rule = BongardExample()
    ids = list(range(1, rule_complexity + 1))
    for i in range(rule_complexity):
        shape = random.choice(shapes)
        rule.add_object(Object(shape, i + 1))

    in_relation = "in"
    for i in range(max(1, rule_complexity - 1)):
        possible_ids = [x for x in ids if x not in [y[1].index for y in rule.relations]]
        if len(possible_ids) < 2:
            break
        id1 = random.choice(possible_ids)
        id2 = random.choice([x for x in ids if x != id1])
        rule.add_relation((in_relation, rule.objects[id1], rule.objects[id2]))
    return rule


def generate_pos_example(index, object_complexity, relation_complexity, rule):
    example = BongardExample(index=index)
    ids = list(range(1, object_complexity + 1))

    for i in ids:
        shape = random.choice(['triangle', 'square', 'circle'])
        new_object = Object(shape, i)
        example.add_object(new_object)

    relations_count = random.choice(range(max(1, relation_complexity - 1), relation_complexity + 2))
    for _ in range(relations_count):
        in_relation = "in"
        possible_ids = [x for x in ids if x not in [y[1].index for y in example.relations]]
        if len(possible_ids) < 2:
            break
        id1 = random.choice(possible_ids)
        id2 = random.choice([x for x in ids if x != id1])
        example.add_relation((in_relation, example.objects[id1], example.objects[id2]))

    example.check_classification(rule)
    if example.classification == 'pos':
        return example
    else:
        return generate_pos_example(index, object_complexity, relation_complexity, rule)


def generate_neg_example(index, object_complexity, relation_complexity, rule):
    example = BongardExample(index=index)
    ids = list(range(1, object_complexity + 1))

    for i in ids:
        shape = random.choice(['triangle', 'square', 'circle'])
        new_object = Object(shape, i)
        example.add_object(new_object)

    relations_count = random.choice(range(max(1, relation_complexity - 1), relation_complexity + 2))
    for _ in range(relations_count):
        in_relation = "in"
        possible_ids = [x for x in ids if x not in [y[1].index for y in example.relations]]
        if len(possible_ids) < 2:
            break
        id1 = random.choice(possible_ids)
        id2 = random.choice([x for x in ids if x != id1])
        example.add_relation((in_relation, example.objects[id1], example.objects[id2]))

    example.check_classification(rule)
    if example.classification == 'neg':
        return example
    else:
        return generate_neg_example(index, object_complexity, relation_complexity, rule)


def generate_bongard_example(num_examples, object_complexity, relation_complexity, rule_complexity, filename):
    if isinstance(rule_complexity, int):
        rule = generate_rule(rule_complexity)
    else:
        rule = rule_complexity

    examples = []
    pos = num_examples // 2
    neg = num_examples - pos

    for i in range(pos):
        examples.append(generate_pos_example(i, object_complexity, relation_complexity, rule))

    for i in range(pos, pos + neg):
        examples.append(generate_neg_example(i, object_complexity, relation_complexity, rule))

    random.shuffle(examples)

    predicates = shapes + relations + ['bongard']
    dataframes = {
        "square": {"id": [], "objectId": []},
        "circle": {"id": [], "objectId": []},
        "triangle": {"id": [], "objectId": []},
        "in": {"id": [], "objectId1": [], "objectId2": []},
        "bongard": {"id": [], "class": []}
    }

    for example in examples:
        for obj in example.objects.values():
            dataframes[obj.shape]["id"].append(example.index)
            dataframes[obj.shape]["objectId"].append(obj.get_object_id())
        for rel in example.relations:
            dataframes['in']["id"].append(example.index)
            dataframes['in']["objectId1"].append(rel[1].get_object_id())
            dataframes['in']["objectId2"].append(rel[2].get_object_id())

        dataframes['bongard']["id"].append(example.index)
        dataframes['bongard']["class"].append(example.classification)

    for key in dataframes.keys():
        df = pd.DataFrame(dataframes[key], columns=dataframes[key].keys())
        df.to_csv(filename + f"/{key}.csv", header=True, index=False)

    return rule



def generate_single_example(index, object_complexity, relation_complexity, rule):
    n = random.choice(range(object_complexity - 2, object_complexity + 3))
    example = BongardExample(index=index)
    ids = list(range(1, n + 1))
    for i in range(1, n + 1):
        shape_type = random.choice(['triangle', 'square', 'circle'])
        new_object = Object(shape_type, i)
        example.add_object(new_object)

    n = random.choice(range(relation_complexity - 2, relation_complexity + 3))
    for i in range(1, n + 1):
        in_relation = "in"
        possible_ids = [x for x in ids if x not in [y[1].index for y in example.relations]]
        if len(possible_ids) < 2:
            break
        id1 = random.choice(possible_ids)
        id2 = random.choice([x for x in ids if x != id1])
        example.add_relation((in_relation, example.objects[id1], example.objects[id2]))

    example.check_classification(rule)
    return example



# rule = generate_rule(2)
# print("Rule: ", rule)
# print(generate_neg_example(1, 5, 2, rule))
# print(generate_pos_example(1, 5, 2, rule))
# #print(generate_single_example(1,5,5,generate_rule(2)))
