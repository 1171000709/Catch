from itertools import combinations
from collections import OrderedDict


class Operation:

    # Self feature value conversion
    value_ops = ['square', 'inverse', 'log', 'sqrt', 'sigmoid', 'tanh']

    math_ops = ['add', 'subtract', 'multiply', 'divide']

    # ops_type = ['value_ops', 'add_ops', 'sub_ops', 'multi_ops',
    #             'div_ops', 'END']
    # ops_type = ['replace', 'concat', 'replace_END', 'concat_END']
    ops_type = ['replace', 'concat']
    # ops_type = ['concat', 'concat']
    # ops_type = ['replace', 'replace']
    # ops_type = ['replace', 'concat', 'replace', 'concat',
    #             'replace_END', 'concat_END']

    special_ops = [None, 'PADDING']

    rf_classification_param = OrderedDict({
        'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500, 600],
        'max_depth': [6, 8, 10, 12, 14, 16, 18, 20, 30, 40],
        'max_features': ['auto', 'sqrt', 'log2'],
        'min_samples_split': [2, 4, 6, 8],
        'min_samples_leaf': [1, 2, 3, 4],
        'bootstrap': [True, False]
    })

    @classmethod
    def get_all_ops_list(cls):
        value_ops = [i for i in cls.value_ops if i is not None]
        all_actions = value_ops + cls.ops_type + cls.special_ops
        return all_actions

    @classmethod
    def get_action_list(cls):
        # #
        # comb_value_ops = list(combinations(cls.value_ops, 2))
        # #
        # res_value_ops = list(zip(cls.value_ops, cls.value_ops))
        # two_value_ops = comb_value_ops + res_value_ops
        value_ops = []
        for single_value in cls.value_ops:
            value_ops.append((single_value, None))
            value_ops.append((None, single_value))

        # value_ops.append((None, None))

        all_actions = cls.math_ops + value_ops

        # print('comb_value_ops', comb_value_ops, len(comb_value_ops))
        # print('two_value_ops', two_value_ops, len(two_value_ops))
        # print('all_actions', all_actions, len(all_actions))
        return all_actions

NO_ACTION = []

if __name__ == '__main__':
    Operation.get_action_list()


