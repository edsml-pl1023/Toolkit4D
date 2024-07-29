def get_top_combination(agglomerate_comb, num_agglomerates):
    sorted_agglomerate_comb = dict(sorted(
        agglomerate_comb.items(), key=lambda item: item[1], reverse=True))
    combinations = []
    record = set()
    for combination in sorted_agglomerate_comb:
        if not (record & set(combination)):
            combinations.append(combination)
            record = record | set(combination)
        if len(combinations) >= num_agglomerates:
            break
    return combinations
