import codex.perf_takehome_under1000 as k
from codex.map_variant_schedule import configure_target, real_tail_ops

configure_target()
k.SCHEDULE_EXACT_CYCLES = None
b = k.KernelBuilder()
try:
    b.build_kernel(10, 2047, 256, 16)
except Exception:
    pass
ops = real_tail_ops(b.dag_ops)
keys = {
    ("negative_five_immediate", "negative_five_broadcast"),
    ("preprocess_pointer", "hash_1_shift_vector"),
    ("scalar_immediate", "hash_1_join"),
    ("scalar_zero_immediate", "cached_nodes_vector_transform"),
    ("scalar_zero_immediate", "constant_broadcast"),
    ("scalar_zero_immediate", "hash_0"),
    ("scalar_zero_immediate", "hash_1_shift_vector"),
}
for child, op in enumerate(ops):
    for parent, lag in op.parents.items():
        parent_op = ops[parent]
        if (
            (parent_op.tag, op.tag) in keys
            and parent_op.engine != op.engine
        ):
            print(
                child,
                op.engine,
                op.group,
                op.round,
                op.tag,
                "<-",
                parent,
                parent_op.engine,
                parent_op.group,
                parent_op.round,
                parent_op.tag,
                "lag",
                lag,
            )
