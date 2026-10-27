# 四指令异或及单次 AND/OR 搜索

**最好可执行结果仍是 864 cycles，尚未取得 863 程序。** 本批继续研究如何
消去首轮 C1 异或，并检查交换子节点在原始记录布局中的实际搬运成本。
内核、865 备份、`problem.py` 和 `tests/` 均未修改。

## 1. 固定输入键的 MAD/XOR 四指令范围全部排除

保留前批记号，`M` 是有两个任意运行时乘数和任意加数的乘加，`X` 是异或。
目标为 `F(x XOR 0xc761c23c) XOR O`，其中 O 任意。

上批剩下 `M M X M` 的 12 个模板。本批按第二条 MAD 的二次项及加数分区：

```text
x*x + {x,t,K}; x*t + {x,t,K}; t*t + {x,t,K}
```

若第二条为仿射值且第三条与原输入或常量异或，已由前批完整的二元多项式
模整数筛选覆盖。若第三条与首个 MAD 结果异或，还需检查自由仿射第二条。

因此生成 110 个固定图查询，全部返回 UNSAT。求解器默认要求所有中间结果
有后继使用；其中 6 个查询的首条实际是死指令，单独归入此前的三指令证明。
其余 104 个查询均有独立小位宽正例。审计对 12 个模板的 480 个原始
第二条连接做了完整覆盖；常量别名和乘法交换均有完整 uint32 测试。

结合已有九种顺序，**三个 MAD 加一个 XOR 或两个 MAD 加两个 XOR 的十种
顺序在这一单输入、无移位、任意输出 XOR 键的范围内全部排除**。
证据索引是 `head_four_xor863_audit.json`，原来的超时查询仍保留。

## 2. 将位运算扩展为 AND/OR

新增的一轮覆盖恰好包含三个 MAD 和一个 AND 或 OR 的八种顺序。
这是比上一批更大的操作符范围，仍只覆盖单个输入及任意常量。

| 位运算位置 | 证明 |
|---|---|
| 首条 | 四点无进位矩形对任意逐位前缀和后续整数多项式成立 |
| 第二条 | 完整枚举 123,008 组 `t AND x` 参数和 3,973,120 组归一化 `t AND K` 参数；每组覆盖 512 个归一化输出键 |
| 第三条 | 306 个末条连接查询得到 286 UNSAT、20 UNKNOWN；常量别名合并后以 90 个固定二次项查询关闭，仿射部分由前述模整数筛选覆盖 |
| 末条 | 对首条仿射情况放宽为 `P4 AND Q2` 或 `P4 OR Q2`，两次均 UNSAT；非仿射首条由直接查询排除 |

OR 的模整数筛选复用 AND，因为 `t OR z = t+z-(t AND z)`；这里 t 对 x
仿射，z 是 x 或常量，所以变换双向保持二元多项式的总次数。枚举同时覆盖
所有输出键的必要条件，未观察到低位幸存项。

`audit_head_single_bitwise863.py` 验证三批常量枚举的覆盖、源文件指纹和
正例，并枚举第三条 AND/OR 顺序的 151,200 个仿射首条连接模式。
完整索引为 `head_single_bitwise863_audit.json`。原始 20 个超时不被改写。

这 **不** 排除两条位运算中包含 AND/OR 的结构、移位、除法、其他 live 值、
联合整段哈希、不同内存布局，或整个内核存在更快解。

为继续扩展范围，还对两条 MAD 加两条 XOR/AND/OR（至少一个 AND/OR）的
48 种顺序做了 96 个首条分区查询：69 个 UNSAT、27 个 UNKNOWN，没有低位
候选。原始 UNKNOWN 中七项由逐位前缀矩形证明关闭；按完整顺序计，当前
28 种关闭、20 种开放。进一步按首个位运算的操作数细分，70 个查询得到
46 个 UNSAT、24 个 UNKNOWN，又关闭三种顺序；因此当前为 **31 种关闭、
17 种仍开放**。这 24 个 UNKNOWN 中有八项可因恒等或常量位运算降为三条，
其余 16 项折叠常量别名后是八个未决类别。见 `head_two_bitwise863_refined_audit.json`。

## 3. 交换原始子节点的顺序

之前把 256 个 d8 子节点由预编码改为原始值，会额外消耗状态恢复及搬运。
这次对更宽松的交换方案重新计算上界：保留 864 图的 64 条八字树预读和
每个八字写入包含两条四字记录，允许四个子节点在每次写入中任意排列，
甚至允许跨记录交换。

每组四个子节点在同一 vload 中占连续四个位置，而 vstore 的子节点槽是
`[0,1,4,5]`。无论两个 scratch 视图如何平移，最多两个子节点可直接别名；
64 组至少需要 128 个真实搬运 lane。去除的 256 个预编码 XOR 与新增的
256 个状态恢复 lane 正好抵消，因此在只用 ALU/VALU 搬运且其他图成本固定时：

```text
算术 lane >= 51665 - 256 + 256 + 128 = 51793
周期 >= ceil(51793 / 60) = 864
```

这是一项**下界，不是新的可运行成绩**。它允许比真实读取选择更多的重排，
因此比单纯检查当前反序排列更乐观。`reversed_raw_records863_bound.json`
记录 64 个真实源视图；测试也检查若改为连续四个子节点槽，可直接别名四个，
明确表明不同 stride/地址布局仍然值得研究。FLOW 搬运、拆分原始加载、
不同 stride、不同状态编码及其他图变化均不包含在此下界中。

## 4. 验证与复现

13 项测试通过，包含 370 个小位宽自动合成正例、完整 uint32 恒等式、
AND 加速器与独立 Python 模整数消元的交叉比对、冻结 ISA 正例和 64 条
真实预读/写回视图的覆盖。源文件、实验结果、环境和保护文件指纹保存在
`bitwise_closure863_validation.json`。

本批未重新运行未变化的官方内核评测；原 864 结果沿用此前验证。
已核对提交内核与独立 864 备份 SHA256 相同，`git diff origin/main -- tests/ problem.py` 为空。

```bash
export PYTHONPATH=/tmp/aopt-synthesis:/Users/yiwei/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages:/tmp/aopt-solver:.
python3.12 -m codex.scan_head_final_quadratic863 --output /tmp/final-quadratic.json
python3.12 -m codex.audit_head_four_xor863 --output /tmp/four-xor-audit.json
python3.12 -m codex.scan_head_single_bitwise863 --output /tmp/single-bitwise.json
python3.12 -m codex.scan_head_and_module863 --source input --output /tmp/and-input.json
# shard 取 0、1、2，分别保留三个输出。
python3.12 -m codex.scan_head_and_module863 --source key --shards 3 --shard 0 --output /tmp/and-key-0.json
python3.12 -m codex.scan_head_and_final863 --output /tmp/and-final.json
python3.12 -m codex.scan_head_and_final863 --refine /tmp/and-final.json --output /tmp/and-refine.json
python3.12 -m codex.audit_head_single_bitwise863 --output /tmp/single-bitwise-audit.json
python3.12 -m codex.scan_head_two_bitwise863 --output /tmp/two-bitwise.json
python3.12 -m codex.audit_head_two_bitwise863 --output /tmp/two-bitwise-audit.json
python3.12 -m codex.scan_head_two_bitwise_refine863 --prior /tmp/two-bitwise.json --output /tmp/two-bitwise-refine.json
python3.12 -m codex.audit_head_two_bitwise_refined863 --output /tmp/two-bitwise-refined-audit.json
python3.12 -m codex.bound_reversed_raw_records863 --output /tmp/reversed-bound.json
python3.12 -m unittest codex.test_head_four_xor863 codex.test_head_single_bitwise863 codex.test_reversed_raw_records863 codex.test_head_two_bitwise863
```

审计脚本默认读取仓库中保存的证据；重跑时应核对结果的配置和文件指纹。
