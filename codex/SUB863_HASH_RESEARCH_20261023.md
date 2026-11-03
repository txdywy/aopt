# 864 之后的哈希改写研究

本轮**没有生成低于 864 周期的新程序**。提交版、独立 864 副本和 865
备份的 SHA256 均与已验证结果一致；冻结模拟器和测试目录未修改。
下文是符号搜索及排除证明，不是新的比赛成绩。

## 三个目标

[结构审计](SUB863_STRUCTURAL_AUDIT_20261023.md) 已排除相同指令图的 863
排程。在首尾边界不变时，需要减少至少 53 个算术 lane。因此本轮检查
能否将三个哈希片段分别从三条指令缩短为两条。所有公式按 uint32 计算。

| 目标 | 表达式 |
|---|---|
| transport | `4097 * (x ^ (x >> 16)) + 0x7ed55d16` |
| middle-tail | `9 * (x ^ (512*x + K)) + 0xfd7046c5` |
| head-xor | `x ^ (x >> 19) ^ 0xc761c23c` |

`K = (0xaccf6200 - 512*0xe9f8cc1d) mod 2**32`。

transport 来自 `L(x)=x^(x>>16)` 的线性及自反性：`L(L(x))=x`，
`L(e ^ L(node))=L(e)^node`。若能压缩它，有机会搬移跨轮状态变换。
但普通轮次还需要解码分支位，不能直接将局部节省乘以全部轮数；第 10
到 11 轮的根节点重置边界更适合作为第一处实验。目前没有候选新内核。

middle-tail 使用换元 `x=33*b+0xe9f8cc1d`。33 在模 `2**32` 下可逆，
且 `16896*b+0xaccf6200 = 512*x+K`。新增测试验证了整个 uint32 域上的
换元和 transport 恒等式，不是对部分输入的近似。

## 搜索范围和结果

`search_hash_transport863.py` 的第一条指令可选择：固定右移 16、任意
固定右移 1 到 31、`x//D`、`x%D`、`cdiv(x,D)`、`D*x+E`、`x*x+D`、
`x^D`、`x&D`、`x|D`。常量任意，除数非零。

第二条枚举原始 ISA 的 14 种操作：加、减、异或、与、或、乘、MAD、
左移、右移、除法、余数、向上取整除法、小于和相等。每个操作数可选
x、第一条结果、任意常量 K0 或 K1。这不涵盖所有两指令程序，例如第一条
的变长移位、比较和 `x*x+x` 未纳入。

| 目标 | 第一条指令族数 | 模板数 | 排除方式 |
|---|---:|---:|---|
| transport | 10 | 140 | 全部 SMT UNSAT |
| middle-tail | 9 | 126 | 125 个 SMT UNSAT，1 个独立多项式证明 |
| head-xor | 10 | 140 | 全部 SMT UNSAT |
| 合计 | 29 | 406 | 405 个 SMT 排除，1 个独立证明 |

middle-tail 未重复检查固定右移 16，它包含在任意固定右移族中。其他
目标的这两个族也重叠；406 是实际模板检查数，不是互不重叠的算法数。

每个模板先使用 61 个确定性边界、单比特及随机样本。样本约束 UNSAT
足以排除该模板全部常量选择；若 SAT，则执行全宽反例查询并将反例回填。
每批两个正控制，共 58 项全部通过全宽等价验证。

`cdiv` 使用 33 位中间加法，匹配模拟器的 Python 整数语义；不能提前
截断 `a+b-1`。除零显式无效。测试将全部 14 种操作与未经修改的冻结
模拟器对照，每种覆盖 31 组输入。

## 超时项的独立证书

transport 的 `cdiv → MAD` 首轮超时，追加预算后获得 UNSAT。
middle-tail 的 `affine → MAD` 追加到 60 秒仍超时，原始记录保留在 JSON
的 `followup` 中，未改写成 UNSAT。

后一模板一定属于 `A*x*x+B*x+C`，因为最后 MAD 的操作数仅来自 x、
仿射结果和常量。取目标低 11 位，x=0、1、2 强制：

```text
C = 1733
A + B = 521 (mod 2048)
2*A = 0 (mod 2048)
```

只剩 `(A,B,C)=(0,521,1733)` 和 `(1024,1545,1733)`。在 x=513 时，
两者均输出 718，目标输出 1742。因此所有二次及更低次数多项式都被排除。
`prove_middle_quadratic863.py` 输出系数和反例证书；测试另用独立 SMT
约束及全宽目标的低位确认，没有只信任证书生成器自身。

## 文件、复现和验证

原始结果保存在 `hash_transport863*_result.json`、
`hash_middle_tail863*_result.json`、`hash_head_xor863*_result.json`。
独立证书为 `hash_middle_tail863_quadratic_certificate.json`，汇总及
文件摘要为 `hash_synthesis863_summary.json`。

```bash
python3.12 -m pip install --target /tmp/aopt-synthesis -r codex/requirements-synthesis.txt
PYTHONPATH=/tmp/aopt-synthesis:. python3.12 -m codex.search_hash_transport863 \
  --objective transport --first cdiv --seconds 45 --output /tmp/transport-cdiv.json
PYTHONPATH=/tmp/aopt-synthesis:. python3.12 -m codex.search_hash_transport863 \
  --objective middle-tail --first affine --seconds 60 --output /tmp/middle-affine.json
PYTHONPATH=/tmp/aopt-synthesis:. python3.12 -m codex.search_hash_transport863 \
  --objective head-xor --first '%' --seconds 12 --output /tmp/head-remainder.json
python3.12 -m codex.prove_middle_quadratic863 --output /tmp/middle-certificate.json
PYTHONPATH=/tmp/aopt-synthesis:. python3.12 -m unittest codex.test_hash_transport863
```

7 项新增测试通过。提交程序完全未改动，因此未重复其此前已通过的
9 项官方测试和 270 例扩展验证。本轮重新检查了三个源码摘要，以及
`git diff origin/main -- tests/ problem.py` 为空。

继续代数搜索需要超出这些两指令族，或共同改写节点编码、路径位和跨轮
状态；改变数据布局则须重算 LOAD、FLOW、地址生成及首尾开销。
这些排除结果不证明 864 是全局最优。
