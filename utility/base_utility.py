import copy
from functools import reduce
from line_profiler import LineProfiler


class BaseUtility:
    @classmethod
    def list_remove_element(cls, data_list, remove_ele):
        try:
            data_list.remove(remove_ele)
        except ValueError:
            pass

    @classmethod
    def list_split_continuous_int(cls, data_list):
        new_list = []
        for i, j in zip(data_list, data_list[1:]):
            if j - i > 1:
                new_list.append(data_list[:data_list.index(j)])
                data_list = data_list[data_list.index(j):]
        new_list.append(data_list)
        return new_list

    @classmethod
    def line_profiler(cls, func, args):
        lp = LineProfiler()
        lp_wrapper = lp(func)
        lp_wrapper(*args)
        lp.print_stats()

    @classmethod
    def get_merge_dict(cls, data, discrete_col):
        discrete_df = data[discrete_col]
        fe_len = discrete_df.shape[0]
        # print('fe_len', fe_len)
        merge_dict = dict()
        categories_count = dict()
        for col in discrete_col:
            col_series = discrete_df[col]
            need_to_merge_categories = []
            count_res = col_series.value_counts(sort=True, ascending=False)
            categories_count[col] = count_res.to_dict()
            # print(col, count_res.to_dict())
            for key, value in count_res.items():
                freq = value / fe_len
                # print('freq', key, freq)
                if value < 10 or freq < 0.001:
                    need_to_merge_categories.append(key)
            if len(need_to_merge_categories) != 0:
                replace_value = \
                    [need_to_merge_categories[0]]*len(need_to_merge_categories)
                col_merge_dict = \
                    dict(zip(need_to_merge_categories, replace_value))
                # print('dictionary', dictionary)
                merge_dict[col] = col_merge_dict
        print('categories_count', categories_count)
        return merge_dict

    @classmethod
    def merge_categories(cls, data, merge_dict):
        data_ = copy.deepcopy(data)
        # print('merge_dict', merge_dict)
        data_.replace(merge_dict, inplace=True)
        return data_

    @classmethod
    def get_filter_discrete_info(cls, data, discrete_col):
        discrete_df = data[discrete_col]
        categories_count = dict()
        for col in discrete_col:
            col_series = discrete_df[col]
            count_res = col_series.value_counts(sort=True, ascending=False)
            categories_count[col] = count_res.to_dict()
        return categories_count
        # print('categories_count', categories_count)

    @classmethod
    def get_discrete_ca_num(cls, data, discrete_col):
        discrete_df = data[discrete_col]
        categories_count = dict()
        for col in discrete_col:
            col_series = discrete_df[col]
            count_res = col_series.value_counts(sort=True, ascending=False)
            categories_count[col] = len(count_res.to_dict())
        return categories_count

    #
    @classmethod
    def list_dict_duplicate_removal(cls, data_list):
        run_function = lambda x, y: x if y in x else x + [y]
        return reduce(run_function, [[], ] + data_list)



