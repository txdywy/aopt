# 混合与非线性首乘加、跨 C1 边界、d9 直接读取

后续：[四指令多项式放宽与原始子节点记录实验](SUB863_POLYNOMIAL_FRONTIER_20261026.md)。
后续可执行最好成绩仍为 864。

**可执行最好成绩仍为 864 cycles，尚未得到 863 程序。** 本轮完成了新的结构搜索、
有限范围排除证据，以及一个经过冻结模拟器回放的读取改图。目标“突破 864”
保持进行中；这些结果不构成全局最优证明。

## 混用已有值与运行时乘法

沿用三个真实相关的 live 输入，而不是把它们当成独立变量：

```text
a = 4097*prior + C0
keyed = a XOR (a>>19) XOR K
target = F(a XOR (a>>19) XOR C1) XOR O
```

K、O 任意。F 为原中段四条公式。低 11 位模型使用合法完整输入子域
`0 <= a < 2048`，因此 `a>>19=0`。此域的 UNSAT 能排除完整恒等式；
低域 SAT 本身不能成为内核候选。

此前只关闭了首条 `A*s+B` 的范围。本轮新增：

- 4 种混合形式：`A*a+keyed`、`A*prior+keyed`、`A*keyed+a`、`A*keyed+prior`。
- 24 种运行时乘积形式：从三个 live 字中选两个相乘，加数为任一 live 字或任意常量。

第二条对首结果与 a/prior/keyed/任意常量执行 XOR、AND 或 OR；末条 MAD
可任意使用已有结果、初始值和两常量。336 项粗查询得到 **333 UNSAT、3 UNKNOWN**。
三个超时均为混合首条、XOR 常量、最后 MAD。把末条连接拆成此前验证过的
50 个角色后，新增 **150/150 UNSAT**。原始超时完整保留。

结合此前仿射首条与矩形证据，现已关闭这三个顺序的任意首末 MAD 连接：
`MAD/XOR/MAD`、`MAD/AND/MAD`、`MAD/OR/MAD`。若中间位运算不读取首 MAD，
可提前到多项式计算之前；重复位运算操作数或未使用结果也由已有多项式证据覆盖。
首条的常量别名可以放宽为独立常量；`A*a+prior` 可归入 a 的任意仿射函数。

审计实际枚举每个顺序 **308,700** 个连接，分别归入：多项式前缀 282,450、
仿射 12,950、混合 2,800、运行时乘积 10,500。此数含常量命名对称和死代码，
不是不同函数的数量，更不表示整个 live 语法已经关闭。

源文件为 `scan_keyed_live_nonlinear863.py`、`audit_keyed_live_nonlinear863.py`；
原始数据为 `keyed_live863_nonlinear.json`、`keyed_live863_nonlinear_refinement.json`，
证据 SHA256 索引为 `keyed_live863_nonlinear_audit.json`。

## 保留四条中段，尝试删除前面的 C1 XOR

新目标为 `P(x)=F(x XOR C1) XOR O`，输出键 O 任意。
搜索扩展到三 MAD 一 XOR、或两 MAD 两 XOR 的一般 SSA 连接，MAD 允许两个
运行时值相乘；不同于此前只检查八个具体仿射结构。

从 MAD 开始有六种顺序，首 MAD 分为仿射、平方加常量、平方加原输入。
18 项查询得到 **11 UNSAT、7 UNKNOWN**；对七个未决类枚举第二条操作数，
160 项得到 **105 UNSAT、55 UNKNOWN**。没有发现低域候选，也没有完整 32 位候选。

### 四个输入排除平方前缀后接多项式

取模 2048，令输入为 `0,32,64,96`。对每个 B，`x*x+B` 在横轴只翻转 bit 10，
纵轴不变；x 自身的两轴分别翻转 bit 5、bit 6。`x*x+x` 同样具有互不相交的
两轴变化集合 `{5,10}`、`{6}`。

任意 XOR/AND/OR 前缀保持每个输出位只依赖一个轴。因此每个字的横向数值差
是 32 的倍数，纵向差是 64 的倍数，任意后续整数多项式的混合差分均为零模 2048。
但目标在这四点的低位为：

```text
F(x XOR C1) mod 2048: 1382, 70, 934, 646
D(F XOR O) mod 2048:  1024，对全部 2048 个 O 都成立
```

`screen_head_square863.py` 实际检查全部 2048 个 B 和全部 2048 个 O。
这关闭平方首条、XOR、两 MAD 的两个剩余超时角色，甚至允许位运算前缀之后
任意次数的多项式运算。它不适用于后面再次交错位运算的结构。

本轮数据与此前证据合起来，在上述十个四指令顺序中 **4 个已关闭、6 个仍开放**。
所有六种 MAD 开头顺序的平方首条均已关闭；尚有五种仿射首条和
`XOR/MAD/MAD/XOR`。160 项中的 **53 项原始 UNKNOWN 仍未关闭**，不能视为不可行。
其中最值得继续研究的是新的仿射连接与常量归一化，而不是重复平方首条查询。

`XOR/MAD/XOR/MAD` 可把第一 XOR 结果看作新 live 模型的 keyed 输入，借用本轮
三条排除；`XOR/MAD/MAD/MAD`、`XOR/XOR/MAD/MAD` 由已有位运算前缀矩形证据排除。
另一个关闭顺序是本轮直接排除的 `MAD/MAD/XOR/XOR`。

源文件与结果以 `scan_head_nonlinear863`、`head_nonlinear863_` 开头。
`audit_head_nonlinear863.py` 逐条验证覆盖、引用和保留的未决状态。
本范围没有包含 AND/OR 四指令图、移位、额外 live 输入或全局布局变化。

## 实际改图：d9 直接载入连续节点向量

检查基线发现，第 10 层已经使用 scalar load 并合并节点 XOR；仍可改的是第 9 层。
`research_direct_d9_load863.py` 实现了如下真实图：

- 将 d9 的四个地址常量再增加 3，将 256 次 vload 改为 payload 的 scalar load。
- 原来的 256 条标量节点 XOR 合并为 32 条向量 XOR。
- 为维持 d10 地址，实际生成 `-6` 向量，并将已提取的奇偶值再 OR 此向量。

最后一步不能省略。若新 d9 指针 `p9'=p9+3`，原 d10 地址是
`2*p9' + parity - 6`。直接把 `state OR -6` 当成 `parity-6` 是错误的：
state=4 时得到 -2，而正确值是 -6。正确的直接补偿每组多一条操作，还需 mask。

| 指标 | 864 原图 | 本次直接 d9 图 |
|---|---:|---:|
| ALU | 2,921 | 2,665 |
| VALU | 6,093 | 6,158 |
| C=ALU+8*VALU | 51,665 | 51,929 |
| LOAD / FLOW / STORE | 1688 / 851 / 530 | 1688 / 851 / 530 |

单算算术量就需要 `ceil(51929/60)=866` 周期，因此这个具体图不能突破 864。
这不是对所有 d9 布局的排除。两个随机 uint32 完整冻结回放都通过，scratch 为
1111/1536。记录中的 `serial_control.verified=11892` 是串行校验日历长度，绝非分数。

## 验证、环境与继续方向

32 项组合测试通过（11.388 秒），随后包含新增矩形验证的 4 项测试通过
（0.197 秒），共覆盖 33 个不同测试。正向验证包括 28 种首条形式的完整 32 位
固定见证、自动发现的混合/乘积小域程序、18 个四指令自动对照，以及冻结 ISA
和随机位运算前缀/多项式。两个证据审计均通过。

组合测试首次运行时，临时依赖目录中的 NumPy 源文件缺失，仅剩目录和缓存，
导致三个数组测试导入失败；改用应用随附 NumPy 2.3.5 后全部通过，没有改变
测试断言或搜索模型。运行版本为 Python 3.12.15、Z3 5.1.0。

提交内核、864/865 备份、`problem.py` 与 `tests/` 均未改变。
当前提交与 864 副本 SHA256 仍为
`98d0f4ce1ecae152378e31ddae70d52f0ed4cb6ace5b027ce191d4027346a884`；
865 为 `58ce2efca049a18ef343ba82ceaeb72fd74f638bd7c8bbc23a14e6e796d72f44`。

复现需 Z3；涉及原有数组测试时还需一个完整的 NumPy 安装。下列命令的
PYTHONPATH 适用于本次本地环境；其他机器可使用安装研究依赖的 Python 环境。

```bash
export PYTHONPATH=/tmp/aopt-synthesis:/Users/yiwei/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages:/tmp/aopt-solver:.
python3.12 -m codex.scan_keyed_live_nonlinear863 --output /tmp/live-nonlinear.json
python3.12 -m codex.scan_keyed_live_nonlinear863 --refine /tmp/live-nonlinear.json --output /tmp/live-refined.json
python3.12 -m codex.audit_keyed_live_nonlinear863 --output /tmp/live-audit.json
python3.12 -m codex.scan_head_nonlinear863 --output /tmp/head-first.json
python3.12 -m codex.scan_head_nonlinear863 --seconds 4 --refine /tmp/head-first.json --output /tmp/head-second.json
python3.12 -m codex.screen_head_square863 --output /tmp/square-certificate.json
python3.12 -m codex.audit_head_nonlinear863 --output /tmp/head-audit.json
python3.12 -m codex.research_direct_d9_load863 --output /tmp/direct-d9.json --seeds 2
python3.12 -m unittest codex.test_head_nonlinear863 codex.test_free_middle_keys863 codex.test_keyed_live_nonlinear863 codex.test_keyed_live_middle863
```

未完成的突破路线仍包括上述六种四指令顺序、带完整高位 live 值的移位改写、
更大范围的哈希合并与新的记录布局。任何真正候选都必须再过完整 uint32 等价、
实际资源成本、调度和冻结提交测试，才能更新 864 成绩。
