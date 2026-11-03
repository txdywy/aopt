# Anthropic Performance Take-Home — 864 cycles

[English](README.md) | **中文**

本仓库是 Anthropic 原版性能笔试（VLIW SIMD 模拟器上的并行树遍历 + 哈希内核）的
长期优化记录。

| 指标 | 数值 |
|---|---|
| **当前已验证最好成绩** | **864 cycles** |
| 相对仓库基线 147,734 cycles | **170.99×** |
| 相对 2 小时版起点 18,532 cycles | 21.4× |
| 官方测试 | `tests/submission_tests.py` 9/9 通过，均报 864 |
| 扩展验证 | 270/270（256 组全 32 位随机 + 8 组普通 + 6 种边界），指令经 JSON 往返后执行 |
| `tests/`、`problem.py` | 与原始仓库完全一致，未作任何修改 |

当前提交是根目录的 [`perf_takehome.py`](perf_takehome.py)，其字节相同的独立副本为
[`codex/perf_takehome_864.py`](codex/perf_takehome_864.py)。
此后约两周、数十轮的 sub-864 研究**没有得到 863 程序**；这些研究给出了
当前算法族下 864 的资源下界论证（见[下文](#为什么停在-864)），但不构成对任意算法的全局最优证明。

## 快速验证

需要 Python 3.10+（记录中使用 3.12，3.13 亦已验证）。

```bash
# tests/ 必须与原始版本一致（应无输出）
git diff 4363da3 -- tests/ problem.py
# 官方测试：应输出 9 次 "CYCLES:  864"
python3 tests/submission_tests.py
# 独立复核：指令摘要、架构写入、内存边界与最终输出
python3 -m codex.verify_compact
```

`verify_compact` 输出的紧凑指令 JSON SHA256 应为
`3f4cc919d7c5b05d105b5e41ba2ea3502020a03ec1318388a1138718abf2d4f1`。

提交文件内嵌压缩后的指令流，与运行时的树和输入值无关；所有树变换、读取和哈希都在
原始冻结模拟器上执行。适用规格固定为 `(height=10, nodes=2047, batch=256, rounds=16)`、
初始索引全零，并遵循官方测试只检查**最终 values** 的契约（不输出 indices）。

## 成绩演进

| cycles | 关键变化 | 存档 |
|---:|---|---|
| 147,734 | 仓库原始基线 | — |
| 1,939 → 1,600 | 向量化、启动段优化、分组交错调度 | git 历史 |
| 1,436 → 1,285 | `opus/` 独立重写：周期深度跟踪、浅层预载 + `vselect`、向量/标量双引擎 | [`opus/`](opus/README.md) |
| 1,023 | 极限内核实验分支 | [`codex/perf_takehome_1023.py`](codex/perf_takehome_1023.py) |
| 995 → 979 | 分支派发尾部、紧凑启动、常量暂存 | `codex/perf_takehome_98x.py`、[`_981_portable`](codex/perf_takehome_981_portable.py) |
| 880 | 新指令编译器 + word-SSA 寄存器分配 + 混合标量/向量调度 | [`codex/perf_takehome_880.py`](codex/perf_takehome_880.py) |
| 868 → 866 | 反向异或调整的四字记录、双线性寻址、锚点 bank | [`_868`](codex/perf_takehome_868.py)、[`_866`](codex/perf_takehome_866.py) |
| 865 | 三字 FLOW 广播、STORE/重读广播 | [`codex/perf_takehome_865.py`](codex/perf_takehome_865.py) |
| **864** | 常量 4 暂存、标量最深层叶读取、原始哈希重结合、计算与访存联合求解（CP-SAT） | [`codex/perf_takehome_864.py`](codex/perf_takehome_864.py) |

详细复现步骤见 [864 结果说明](codex/SUB864_864_RESULT_20261023.md)，旧版说明见
[`codex/README.md`](codex/README.md)。

## 864 内核的主要技术

- **哈希压到每轮 10 条算术**：`+`/`<<` 两段各合成一条 `multiply_add`；第 2、3 段
  重结合为两条以同一输入为源的乘加再异或；末段常量 C5 推迟到下一轮，预先异或进
  d0–d8 节点（511 lane），d9/d10 与最终输出逐元素补回。加上节点异或，每轮 11 条。
- **反向、异或调整的四字记录**：每条记录含一个节点及其两个子节点，一次 gather 覆盖两层；
  局部搜索得到的 224 字布局让 d3 与 d5 记录共享双线性地址算术，d5 记录还携带
  树块指针（锚点），省掉后续地址计算。
- **重叠 scratch 窗口 + `vselect`** 复用已读取的子节点；浅层 d0–d2 节点常驻 scratch，
  在 FLOW 引擎上选择。
- **复用已消耗内存**：输入、表头、树和索引区域在用完后被用作临时记录和选择 bank。
- **调度**：自研 word-SSA 分配器与混合标量/向量调度器；最后用 CP-SAT 联合求解计算与访存
  日历，并以精确后缀求解收尾。

可读的编译器与调度器在 `codex/`：`stride4_family.py`、`research_sub864.py`、
`research_stage864.py`、`word_ssa.py`、`packet_modes.py`、`packet_mode_search.cpp`；
已验证的 864 日历保存在 `codex/stage_astra_constant4_864_schedule.txt`。

## 为什么停在 864

864 程序的资源账本（逐包统计）：

```
容量 864 × 60 lane = 51,840      算术工作 C = 51,665 lane
空闲 = 首部 52 + 中段 7 + 尾部 116 = 175
LOAD 1,688（截止期内占满）  FLOW 851（余量约 2）  STORE 530（大量余量）
```

- 第 0 周期 scratch 全零，至多产生一条全 1 向量，**首部至少浪费 52**；最后一个周期只能
  STORE，前两周期只能喂存储，**尾部至少浪费 116**。
- 因此 863 需要 `C ≤ 863×60 − 168 = 51,612`，即**至少再削 53 lane**（约 7 条 VALU）。
- 每元素工作：哈希 16×10、节点异或 16、奇偶位 14、gather 地址 6、C5 恢复 3；
  各项均已按深度或操作码证明为局部最优。
- 已知所有"以 FLOW/LOAD 换算术"的交换，释放一单位 FLOW/LOAD 至少花 4 lane，
  而能换回的算术至多 1 lane；唯一有余量的 STORE 写出的数据必须由已饱和的 LOAD 读回。

总论证见 [`SUB863_LOWER_BOUND_ARGUMENT_20261027.md`](codex/SUB863_LOWER_BOUND_ARGUMENT_20261027.md)
与 [`SUB863_SHADOW_PRICES_20261027.md`](codex/SUB863_SHADOW_PRICES_20261027.md)。

### 仍然开放的两条路

1. **每轮哈希从 10 条压到 9 条**（一次即省 4,096 lane）。需要一种含右移、除法或跨轮存活值、
   且超出现有 3–4 条指令穷举窗口的新改写。最新的全 32 位 z3 重跑又封死 46 个窗口，0 SAT，
   仍有 26 个超时未决（[`SUB863_FULL32_RERUN_20261103.md`](codex/SUB863_FULL32_RERUN_20261103.md)）。
2. **以 STORE 余量换取算术、且不消耗 LOAD 的新访存原语**——在本 ISA 下尚未找到。

## 研究索引（`codex/SUB863_*`）

| 方向 | 主要文档 |
|---|---|
| 总论证与资源价格 | [下界论证](codex/SUB863_LOWER_BOUND_ARGUMENT_20261027.md)、[影子价格](codex/SUB863_SHADOW_PRICES_20261027.md)、[结构审计](codex/SUB863_STRUCTURAL_AUDIT_20261023.md)、[资源复审](codex/SUB863_FRESH_PASS_20261027.md) |
| 全新算法族复核 | [第一性原理复核](codex/SUB863_FRESH_ALGO_20261103.md)、[补充否定结论](codex/SUB863_NEGATIVE_NOTES_20261103.md) |
| 哈希首段 / C1 吸收 | [首段语法](codex/SUB863_HEAD_GRAMMAR_20261028.md)、[编码刚性](codex/SUB863_REENCODING_RIGIDITY_20261027.md)、[仿射模块](codex/SUB863_AFFINE_MODULE_20261027.md)、[FLOW 出口](codex/SUB863_FLOW_SINK_20261027.md) |
| 哈希中段 / 末段 / C5 | [中段跨阶段](codex/SUB863_MIDDLE_CROSS_STAGE_20261023.md)、[含右移中段](codex/SUB863_SHIFT_COMPLETE_20261024.md)、[C5 并入中段](codex/SUB863_MIDDLE_C5_20261029.md)、[C5 吸收](codex/SUB863_C5_ABSORPTION_20261028.md)、[联合尾部](codex/SUB863_JOINT_TAIL_20261024.md) |
| 除法、比较与动态移位 | [除法比较](codex/SUB863_DIVISION_COMPARISON_20261028.md)、[单除法核心](codex/SUB863_SINGLE_DIVISION_CORE_20261028.md)、[多次除法](codex/SUB863_TWO_DIVISION_20261029.md)、[动态右移量](codex/SUB863_RUNTIME_COUNT_20261029.md) |
| 第 0 轮与根节点 | [第 0 轮合并](codex/SUB863_ROUND0_MERGE_20261028.md)、[钉住移位量](codex/SUB863_ROUND0_PINNED_20261028.md)、[根边界](codex/SUB863_ROOT_BOUNDARY_20261028.md) |
| 记录、选择与活跃性 | [选择车道引理](codex/SUB863_SELECT_LEMMA_20261027.md)、[逐车道活跃性](codex/SUB863_LIVENESS_20261027.md)、[尾部推迟](codex/SUB863_TAIL_DEFERRAL_20261027.md) |
| 构图与常量 | [374 配置扫描](codex/SUB863_CONFIG_SWEEP_20261027.md)、[常量链](codex/SUB863_UNIFORM_CHAINS_20261027.md)、[标量常量](codex/SUB863_SCALAR_UNIFORMS_20261025.md) |
| 模拟器语义 | [冻结模拟器复读](codex/SUB863_SIMULATOR_REREAD_20261027.md) |

## 目录结构

| 路径 | 内容 |
|---|---|
| `perf_takehome.py` | 当前 864 提交（自包含，不需要求解器或数据文件） |
| `problem.py`、`tests/` | 原始模拟器与冻结测试（未修改） |
| `codex/` | 864 编译器、调度器、求解器脚本、全部历史提交副本与研究记录 |
| `opus/` | 独立重写的 1,285-cycle 内核及其说明 |
| `anti/` | 早期实验变体 |
| 根目录 `temp_perf*.py`、`test_hang*.py` 等 | 历史调试脚本，与当前提交无关 |

## 致谢

带符号奇偶的双线性寻址与步长四记录受
[Josu San Martin 的公开研究日志](https://github.com/josusanmartin/josusanmartin.github.io/blob/master/vliw/chat.log)
启发；[Sakauma 的 880-cycle 内核](https://github.com/Sakauma/my_original_performance_takehome/blob/cb2e0554735ef7d0dfde1a04bc2c12c4bb2ed38d/perf_takehome.py)
提供了最初的哈希调度优先级。最终 864 程序是本地生成的全新指令流，原始源码声明保留。

---

## Original take-home README

This repo contains a version of Anthropic's original performance take-home, before Claude Opus 4.5 started doing better than humans given only 2 hours.

The original take-home was a 4-hour one that starts close to the contents of this repo, after Claude Opus 4 beat most humans at that, it was updated to a 2-hour one which started with code which achieved 18532 cycles (7.97x faster than this repo starts you). This repo is based on the newer take-home which has a few more instructions and comes with better debugging tools, but has the starter code reverted to the slowest baseline. After Claude Opus 4.5 we started using a different base for our time-limited take-homes.

Now you can try to beat Claude Opus 4.5 given unlimited time!

## Performance benchmarks 

Measured in clock cycles from the simulated machine. All of these numbers are for models doing the 2 hour version which started at 18532 cycles:

- **2164 cycles**: Claude Opus 4 after many hours in the test-time compute harness
- **1790 cycles**: Claude Opus 4.5 in a casual Claude Code session, approximately matching the best human performance in 2 hours
- **1579 cycles**: Claude Opus 4.5 after 2 hours in our test-time compute harness
- **1548 cycles**: Claude Sonnet 4.5 after many more than 2 hours of test-time compute
- **1487 cycles**: Claude Opus 4.5 after 11.5 hours in the harness
- **1363 cycles**: Claude Opus 4.5 in an improved test time compute harness
- **??? cycles**: Best human performance ever is substantially better than the above, but we won't say how much.

While it's no longer a good time-limited test, you can still use this test to get us excited about hiring you! If you optimize below 1487 cycles, beating Claude Opus 4.5's best performance at launch, email us at performance-recruiting@anthropic.com with your code (and ideally a resume) so we can be appropriately impressed, especially if you get near the best solution we've seen. New model releases may change what threshold impresses us though, and no guarantees that we keep this readme updated with the latest on that.

Run `python tests/submission_tests.py` to see which thresholds you pass.

## Warning: LLMs can cheat

None of the solutions we received on the first day post-release below 1300 cycles were valid solutions. In each case, a language model modified the tests to make the problem easier.

If you use an AI agent, we recommend instructing it not to change the `tests/` folder and to use `tests/submission_tests.py` for verification.

Please run the following commands to validate your submission, and mention that you did so when submitting:
```
# This should be empty, the tests folder must be unchanged
git diff origin/main tests/
# You should pass some of these tests and use the cycle count this prints
python tests/submission_tests.py
```

An example of this kind of hack is a model noticing that `problem.py` has multicore support, implementing multicore as an optimization, noticing there's no speedup and "debugging" that `N_CORES = 1` and "fixing" the core count so they get a speedup. Multicore is disabled intentionally in this version.
