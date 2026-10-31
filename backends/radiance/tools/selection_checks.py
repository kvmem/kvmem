"""Pure checks for the frozen-query controller's separately reserved suffix."""


def untrimmed_within_reserve(item, resumes, budget, reserve, replays=()):
    tokens = int(item['tokens'])
    if tokens <= budget:
        return True
    matching = [r for r in resumes if r['request'] == item.get('request')]
    # A resumed prefix is selected only when IT exceeds the history budget.
    # The new tool suffix and generation occupy the separately declared reserve.
    if (len(matching) == 1 and
            0 < int(matching[0]['frontier']) <= budget and
            0 <= tokens - int(matching[0]['frontier']) <= reserve):
        return True
    # A zero-frontier query replay has no retained history. All evaluated
    # tokens, including generation, must fit the separately declared reserve.
    # Require actual replay evidence; a short request alone is insufficient.
    zero_replays = [r for r in replays
                    if r.get('session') == item.get('session') and
                    r.get('before_sampler') == '1' and int(r['begin']) == 0 and
                    0 < int(r['end']) <= tokens]
    return len(zero_replays) == 1 and tokens <= reserve
