"""Necessary resource gates, not predicted schedules or global lower bounds."""
import json
from pathlib import Path


def gate(slots, savings, target=899):
    work = slots['alu'] + 8 * slots['valu']
    remaining = {
        'arithmetic_lane_units': work - savings.get('arithmetic_lane_units', 0),
        'load': slots['load'] - savings.get('load', 0),
        'flow': slots['flow'] - savings.get('flow', 0),
    }
    capacities = {'arithmetic_lane_units': 60 * target, 'load': 2 * target, 'flow': target}
    deficits = {k: max(0, v - capacities[k]) for k, v in remaining.items()}
    return {'remaining_work': remaining, 'necessary_deficits': deficits,
            'passes_relaxed_gate': not any(deficits.values())}


if __name__ == '__main__':
    data = json.loads((Path(__file__).parent / 'resource_profile_983.json').read_text())
    assert data['verified_all_outputs'] and data['full32']
    cases = {
        'unchanged_983': {},
        # Hypothetical exact stage elimination, not an implemented rewrite.
        'hypothetical_one_hash_stage': {'arithmetic_lane_units': 256 * 16},
        # An optimistic budget contract: selection/address overhead is NOT free
        # in an implementation. This establishes the maximum tolerable addition.
        'hash_stage_and_96_loads_and_40_flow': {
            'arithmetic_lane_units': 4096, 'load': 96, 'flow': 40},
    }
    result = {name: gate(data['slots'], saving) for name, saving in cases.items()}
    assert result['unchanged_983']['necessary_deficits'] == {
        'arithmetic_lane_units': 3689, 'load': 91, 'flow': 37}
    assert not result['hypothetical_one_hash_stage']['passes_relaxed_gate']
    assert result['hash_stage_and_96_loads_and_40_flow']['passes_relaxed_gate']
    print(json.dumps({'source': 'resource_profile_983.json', 'target': 899,
                      'cases': result,
                      'warning': 'Necessary only; ignores dependencies, startup, stores, register pressure and engine-specific restrictions.'}, indent=2))
