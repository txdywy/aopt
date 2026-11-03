"""Validate the actual raw-load witnesses and optimistic alias bound."""
import json
from itertools import permutations
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


def maximum_alias(source,slots):
    return max(sum(source_value==slot+offset for source_value,slot in zip(order,slots))
               for order in permutations(source) for offset in range(-16,17))


class ReversedRawRecordTests(unittest.TestCase):
    def test_all_real_source_views_and_optimistic_bound(self):
        row=json.loads((ROOT/'reversed_raw_records863_bound.json').read_text())
        self.assertEqual(64,len(row['cases']))
        for case in row['cases']:
            source,slots=case['source_offsets'],case['destination_slots']
            self.assertEqual(4,len(set(source)))
            self.assertEqual(3,max(source)-min(source))
            self.assertEqual([0,1,4,5],slots)
            self.assertEqual(2,maximum_alias(source,slots))
            self.assertEqual(2,case['optimistic_direct_aliases'])
        self.assertEqual(128,row['optimistic_minimum_copy_lanes'])
        self.assertEqual(51665-256+256+128,row['arithmetic_lanes_lower_bound'])
        self.assertEqual(864,row['arithmetic_only_cycle_lower_bound'])

    def test_contiguous_child_layout_can_alias_four_outside_this_bound(self):
        self.assertEqual(4,maximum_alias([4,5,6,7],[0,1,2,3]))


if __name__=='__main__':unittest.main()
