"""Read-only memory audit for prospective grouping; never enlarge judge memory."""
import json
from tests.frozen_problem import Tree, Input, build_mem_image, SCRATCH_SIZE


if __name__ == '__main__':
    memory = build_mem_image(Tree(10,[0]*2047), Input([0]*256,[0]*256,16))
    end = memory[6]+256
    assert len(memory) == end == 2566
    # Existing index/value regions total512 words, not an extra buffer.
    assert memory[5]+256 == memory[6]
    assert memory[6]+256 == len(memory)
    # A possible barrier-based scratch layout, before live-range accounting.
    layout = {'input_keys':256, 'input_values':256,
              'output_keys':256, 'output_values':256,
              'histogram':16, 'bucket_bases':16}
    total = sum(layout.values())
    assert total <= SCRATCH_SIZE
    print(json.dumps({'actual_memory_words':len(memory), 'tail_free_words':0,
                      'existing_index_value_words':512, 'scratch_words':SCRATCH_SIZE,
                      'conceptual_sort_scratch_words':total,
                      'remaining_before_other_live_state':SCRATCH_SIZE-total,
                      'warning':'Scratch capacity only. ISA has no general indirect scratch write; live kernel state and routing costs are not accounted for.'},indent=2))
