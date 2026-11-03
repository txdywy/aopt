# 全仓 UNKNOWN 盘点与真开放项重跑（2026-11-03）

**可执行最好成绩仍为 864 cycles，本轮没有得到 863 程序。** 未修改提交内核、
`problem.py` 或 `tests/`；官方测试仍报 864。

上一篇 `SUB863_FULL32_RERUN_20261103.md` 重跑了中段/尾部/joint 主文件的超时项
（46 项新封死、0 SAT）。本篇处理其余所有结果文件里的 UNKNOWN：先判断哪些早已被
后续证据关闭，再把**真正仍开放**的用大预算重跑。

## 1. 盘点：几千条 UNKNOWN，绝大多数早已关闭

扫描 `codex/` 下全部 `.json` 与 `.jsonl.gz`，共 **121 个文件**带有
`unknown/timeout/inconclusive/open` 一类标签，合计数千条。研究约定"保留原始超时，
不把超时改写成 UNSAT"，所以这些标签不等于开放问题。逐族追溯闭环链：

| 族 | 原始未决 | 后续关闭证据 |
|---|---:|---|
| head_*：C1 吸收，含两运行时乘数的 MAD | 约 330 | `head_affine_frontier` 9/10 → `head_four_xor` 10/10；`head_single_bitwise` 8/8；`head_two_bitwise` 剩 17 个顺序由 `cegis_c1_absorb863`（42,868 连线）收尾，`c1_all/onemad.unresolved = []` |
| head_nonlinear | 55+ | 同一范围，被 `head_four_xor` 10/10 取代 |
| middle_live / live_roles / cross_stage / bitwise_closure | 42 | `middle_live_closure863_results.json`：64/64 关闭（含 `close_live_middle_residual863` 的最后 8 类） |
| `grammar3/4_*`、`round0_*`、`head_c5`、`tail_c5`、division | 约 2,300 | ROUND0_PINNED、HEAD_GRAMMAR、C5_ABSORPTION、CLAUDE_REVIEW 的证书与引理 |
| `cegis_middle_c5` 236 超时 | 236 | 146 个无进位矩形证书 + 90 个精确逐位提升 |
| free_keys / keyed 各分区 | 数十 | FLOW_ENCODING、KEYED_LIVE、NONLINEAR_LIVE 的结构分区 |
| `*_control`、cross_encoding 的唯一未决 | 若干 | **正控例**，不是开放模板 |

结论：真正没有任何关闭证据的只剩 **76 项**——joint 6→5 族 74 项、
keyed-live 的 `MAD,MAD,AND` 与 `MAD,MAD,OR` 2 项。

## 2. 重跑真开放项

### joint 6→5（b→G 由 6 条压到 5 条）

来源为 `joint_direct_*`（b/bu/bv 各 profile、SHR16 与任意常量乘数实验）与
`joint_middle_tail863_{u,v}{,_extended}`。从原 78 条中剔除已被仿射折叠证书关闭的
4 条，其余 74 条按各行保存的原始命令复现，只把预算从 8 秒提到
`--seconds 200 --check-ms 60000`，z3 5.3.3，4 路并行。

| 文件 | 新封死（有限文法 UNSAT） | 仍超时 |
|---|---:|---:|
| joint_direct_b863（extended） | 8 | 20 |
| joint_direct_b863_sparse | 10 | 1 |
| joint_direct_bu_shift16863 | 2 | 5 |
| joint_direct_bv_shift16863 | 1 | 4 |
| joint_direct_arbitrary_bu863 | 1 | 6 |
| joint_middle_tail863_u | **3（全族关闭）** | 0 |
| joint_middle_tail863_u_extended | 4 | 2 |
| joint_middle_tail863_v | 1 | 1 |
| joint_middle_tail863_v_extended | 3 | 2 |
| **合计** | **33** | **41** |

**0 个全 32 位候选。** 逐查询权威结果见 `sub863_joint74_rerun_20261103.json`
（文件以运行日志为准，避免上一轮按序列名命名时的字符映射碰撞）。

### keyed-live `MAD,MAD,AND` 与 `MAD,MAD,OR`

低位域 `a < 2048` 下 `a>>19 = 0`、程序不含移位，所以宽 11 的 UNSAT 对完整 uint32
可靠。原始只跑 20 秒；本轮给 600 秒，**两项仍超时**。这印证了
`SUB863_REMAINING_ROUTES_20261023.md` 的判断：同一个通用操作数选择器模型，
延长时间没有用，需要结构分区。

可直接排除的部分：目标在低位域是置换，每一位都均衡。若末条 AND 的某个操作数是
置换（a、prior、keyed 都是），包含关系 `X ⊇ T` 加位计数相等迫使 `X = T`，这些
函数显然不等于目标；若某个操作数是常数，它只能是中性掩码，程序退化为两条 MAD，
已由 outer 审计覆盖。OR 对称。**剩下的核心情形**是"两个都不是双射的 MAD 结果，
做 AND/OR 后恰好得到一个双射"，需要按首 MAD 形式分区再逐类排除。

## 3. 新发现：一块从未搜索过的区域（region D）

`SUB863_MIDDLE_C5_20261029.md` 问：四条指令能否算出 `F(z ^ K) ^ O`（K 任意，
O 为四个可吸收键）？若成立，C5 的 **1,279 lane** 全部省掉。它在全部 T 函数运算
（MAD/XOR/AND/OR）下已封死，但明确写道**含右移的程序"这一部分没有搜索"**。

本轮给出约简论证，把这块区域缩到一个小子类：

1. 目标是 T 函数（输出第 i 位只依赖输入低 i 位）。
2. `u = v >> s` 的**每一位**都依赖更高的输入位；XOR/AND/OR/加法都无法让这种泄漏
   回到 T 函数（AND 掩码要消除所有泄漏位只能得到 0）。
3. 唯一的消除方式是乘以一个恒有至少 s 个尾随零的数：
   - 若乘数是常数 `2^s·c`，则 `(v>>s)·2^s·c = (v & ~(2^s−1))·c`，即**这条 SHR
     等价于一条 AND**，指令数不变，落回 MIDDLE_C5 已关闭的情形；
   - 剩下的**奇异子类**：u 乘以一个**恒可被 2^s 整除的运行时值**（如 `z·z+z` 恒为偶数）。
4. 对奇异子类，`u·y` 的低 w 位只依赖输入的低 w 位（u 中泄漏的高位被乘到 2^w 以上），
   因此低位宽 UNSAT 原则上可用。但有一个边界情形：若 v 的第 [w, w+s) 位对所有输入
   都恒定，乘数不必可整除，低位模拟会漏掉一个常数项。严格的低位检查必须再引入一个
   自由的高位常量 H。

所以 region D 可以搜，但要作为单独的新课题认真建模（只枚举"SHR 结果 × 运行时乘数"
的连线，并处理 H 边界情形），不在本轮范围内。考虑到 F 本身是"两条仿射、异或、
再仿射"的结构，右移没有自然用途，预计无解；但这块区域此前从未被检查过。

## 4. 合计（两篇 11-03 重跑）

| 批次 | 新封死 | 仍超时 | SAT |
|---|---:|---:|---:|
| 中段 / 尾部 / joint 主文件（上一篇） | 46 | 26 | 0 |
| joint 6→5 其余族 + keyed-live（本篇） | 33 | 43 | 0 |
| **合计** | **79** | **69** | **0** |

没有发现任何可执行的 9 条/轮哈希，也没有让中段吸收 C5 的改写。864 维持不变。
以上均为必要条件的排除证据，不是任意算法的全局最优证明。

## 复现

```bash
pip install z3-solver==5.3.3.0 numpy
# joint 6→5 某项（命令与原始记录相同，只加大预算）
python3 -m codex.search_joint_tail863 --query --sequence='>>,^,mad,mad,mad' \
  --seconds 200 --check-ms 60000 --stage b --profile extended --shift 16
# keyed-live
python3 -m codex.search_keyed_live_middle863 --sequence 'mad,mad,&' --width 11 \
  --seconds 600 --check-ms 300000 --output /tmp/keyed-and.json
```
