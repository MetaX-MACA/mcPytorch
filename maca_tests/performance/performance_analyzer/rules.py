
class GenericRule(object):
    def __init__(self, threshold_pct, large_is_better=True):
        self.threshold_pct = threshold_pct
        self.large_is_better = large_is_better

    def calculate_golden(self, max, min, avg, **kwargs):
        return max if self.large_is_better else min

    def check_regression(self, metric, max, min, avg, golden, **kwargs):
        if metric is not None and golden:
            if self.large_is_better:
                return metric < golden * self.threshold_pct
            else:
                return metric > golden * self.threshold_pct

available_rules = {
    'larger_better_90pct': GenericRule(0.9),
    'smaller_better_110pct': GenericRule(1.1, False),
}

def get_rule(rule_name=None):
    if rule_name is None:
        return available_rules['larger_better_90pct']
    return available_rules[rule_name]
