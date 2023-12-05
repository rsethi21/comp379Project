import concurrent.futures
from itertools import repeat

def combos(**params):
    
    all_combos = []
    parameters = list(params.keys())
    values = list(params.values())
    
    def two_list_combos(list1, list2, parameter1, parameter2):
        combos_for_two = []
        for value1 in list(list1):
            for value2 in list(list2):
                combos_for_two.append({parameter1: value1, parameter2: value2})
        return combos_for_two       

    if len(parameters) == 0:
        print("No parameters provided. Empty list of combinations returned.")

    elif len(parameters) == 1:
        for value in list(values):
            all_combos.append({parameters[0]: value})

    elif len(parameters) == 2:
        all_combos = two_list_combos(values[0], values[1], parameters[0], parameters[1])

    else:
        initial_combos = two_list_combos(values[0], values[1], parameters[0], parameters[1])
        for parameter, value in zip(parameters[2:], values[2:]):
            temp_combos = []
            for v in list(value):
                copy_combos = []
                for combo in initial_combos:
                    dictionary = combo.copy()
                    dictionary[parameter] = v
                    copy_combos.append(dictionary)
                temp_combos.extend(copy_combos)
            initial_combos = temp_combos 
        all_combos = initial_combos  

    return all_combos

def grid_search_multi(evaluate, data, workers, **params):
   
    hyperparameters = combos(**params)
    print(f"Number of fits to make: {len(hyperparameters)}")
    print()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(evaluate, repeat(data), hyperparameters)

    best_model_index = -1
    best_model_performance = -1
    for i, result in enumerate(results):
        if result > best_model_performance:
            best_model_index = i
            best_model_performance = result
    
    return hyperparameters[best_model_index], best_model_performance
