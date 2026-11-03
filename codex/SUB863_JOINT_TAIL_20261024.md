# 中段与末尾联合：六条改五条的构造搜索

后续：[直接 b 输入、任意常量乘数及任意两端 XOR 编码键研究](SUB863_ENCODING_FRONTIER_20261024.md)。
新增 336 项联合查询、完整自动六条对照及新的编码结构排除；实测仍为 864。

**当前可执行成绩仍是 864 cycles，没有找到更短的完整公式。**
在完成局部末段两指令排除后，本轮直接搜索整个六条中段/末尾，尝试总计五条。
局部下界不能相加为整体下界，因此这批实验仍有独立意义。

## 搜索目标和范围

实现为 `search_joint_tail863.py`。目标是

```text
b = (4097*p+0x7ed55d16) XOR ((4097*p+0x7ed55d16)>>19) XOR 0xc761c23c
u = 33*b+0xe9f8cc1d
v = 16896*b+0xaccf6200
e = 9*(u XOR v)+0xfd7046c5
G(p) = e XOR (e>>16)
```

所有计算均为完整 uint32。较早的 `p,a,a>>19,a XOR C1,b` 可读取，且它们严格来自
同一个 p。完整原程序从 b 到 G 使用 MAD、MAD、XOR、MAD、SHR、XOR 六条。

本轮采取构造性有限语法：MAD 的一个乘数从指定集合选取，另一个输入和加数
可来自现有值、新结果或指定常数；XOR 和字面量 SHR 可任意连接。每个非末条结果
要求有后续消费者。有限语法 UNSAT 不能解释为任意乘数、任意操作码的五条程序不可能。

基础乘数是 `1,9,33,512,4097,16896,-1`。扩展集合还加入复合乘数
`297,152064,16929,152361,36873,135201,65536`、若干负数及奇数乘法逆元。
扩展常数包含合并乘加得到的加数和 `D-512*B`。候选若成立，仍需在真实编译器中
计入常量构造、寄存器连续性、scratch、访存和调度，不能只凭表达式报分。

## 让求解器集中搜索连接方式

完整六条未固定参数的正向查询在 15 秒内也为 UNKNOWN，保存在
`joint_middle_tail863_unpinned_control.json`。因此不能把复杂五条查询的超时视为排除。

随后保留已知的第一条或两条原乘加，把剩余部分交给求解器。

| stage | 保留的乘加 | 新指令数 | 合计 |
|---|---:|---:|---:|
| first | 0 | 5 | 5 |
| u | u 的 1 条 | 4 | 5 |
| v | v 的 1 条 | 4 | 5 |
| branches | u、v 的 2 条 | 3 | 5 |

保留两条之后，求解器自动恢复了已知的四条后续公式，用时 0.27 秒，并通过了
全 uint32 反例检查。该记录为 `joint_middle_tail863_branches_unpinned_control.json`。
这证明简化后的查询能恢复真实程序，但四条加前缀两条仍是六条，不是新优化。

每个批量实验还检查固定原公式的完整正向对照，以及自动发现的
`(33*b+B) XOR b` 对照。候选先在 312 个完整输入上快速复算；只有全部通过后，
才进行全 uint32 符号反例查询。有限样本通过不会被标为等价。

## 实际结果

以下是查询次数，并非互不相交的全局程序分类。所有原始状态分别保存，未覆盖历史。

| 范围 | 查询数 | 有限语法 UNSAT | UNKNOWN |
|---|---:|---:|---:|
| first，基础集合，SHR16 | 50 | 26 | 24 |
| u，基础集合，SHR16 | 32 | 29 | 3 |
| v，基础集合，SHR16 | 32 | 30 | 2 |
| branches，基础集合，SHR16 | 12 | 12 | 0 |
| u，扩展集合，SHR16 | 32 | 26 | 6 |
| v，扩展集合，SHR16 | 32 | 27 | 5 |
| branches，扩展集合，SHR16 | 12 | 12 | 0 |
| branches，扩展集合，SHR7/9/12/19 | 48 | 48 | 0 |
| 合计 | 250 | 210 | 40 |

没有得到通过完整 32 位验证的五条候选。各阶段的 JSON 都以
`joint_middle_tail863_` 开头；`joint_middle_tail863_results.json` 是首轮完整五条搜索。
基础和扩展集合的超时不可直接比较，因为扩展语法更大，且求解次序也会影响耗时。

## 验证和复现

`test_joint_tail863.py` 的 **6 项测试通过，0.791 秒**：
原公式在全部前缀模式及两个常数集合中均与冻结哈希一致；每条已知公式有完整
32 位 SMT 等价检查；720 次随机程序执行交叉比较冻结 ISA、独立整数解释器和
Z3 表达式；自动正向查询恢复真实程序；还检查了阶段成本、常数范围和历史 UNKNOWN。

```bash
PYTHONPATH=/tmp/aopt-synthesis:. python3.12 -m codex.search_joint_tail863 \
  --stage first --seconds 8 --check-ms 5000 --output /tmp/joint-first.json
PYTHONPATH=/tmp/aopt-synthesis:. python3.12 -m codex.search_joint_tail863 \
  --stage u --profile extended --seconds 12 --check-ms 10000 --output /tmp/joint-u.json
PYTHONPATH=/tmp/aopt-synthesis:. python3.12 -m codex.search_joint_tail863 \
  --stage branches --profile extended --shift 7 \
  --seconds 8 --check-ms 6000 --output /tmp/joint-branches7.json
PYTHONPATH=/tmp/aopt-synthesis:. python3.12 -m unittest codex.test_joint_tail863
```

仍开放的构造空间包括额外常数、非常量 MAD 乘数、更多操作码、其他连接顺序，
以及不保留原乘加的整体改写。直接以 b 为独立输入、缩小无关操作数池，也可能让
求解更有效率。根目录提交、864/865 副本和冻结模拟器均未改动。
