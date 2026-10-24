# 直接 b 输入与任意 XOR 编码键的继续研究

**已验证可执行成绩仍为 864 cycles，没有得到 863 程序。**
本轮扩充了构造搜索和编码范围，并保留全部 UNKNOWN。根目录提交、864/865
独立副本、模拟器和冻结测试均未修改。

## 1. 缩小无关操作数池，再开放任意乘数和常量

记 uint32 中段及末段为

```text
u = 33*b + 0xe9f8cc1d
v = 16896*b + 0xaccf6200
e = 9*(u XOR v) + 0xfd7046c5
G(b) = e XOR (e>>16)
```

`search_joint_tail863.py` 新增 b、bu、bv、bbranches 四个模式。
b 本身遍历完整 uint32；另三个模式分别免费提供已经执行的一条或两条原乘加，
并把前缀成本计入总计五条的预算。它们不读取更早的 p、a、a>>19，因而是
较早全 live 输入搜索的子范围，不能把这里的排除推广到额外输入。

原程序由六条运算组成。直接 b 输入的自动六条正向查询，在较大的常量集合下
仍超时；仅保留原乘数及必要加数的 sparse 集合后，自动恢复原公式用时
**8.031 秒**，随后通过完整 32 位反例检查。记录为
`joint_direct_b863_sparse_control.json`。自动恢复 bu、bv 原公式分别约
2.94、4.87 秒；这提高了模型可发现真实程序的证据，仍不是新优化。

另外开放每条 MAD 的任意 uint32 **常量乘数**，以及两个任意共享常量。
此处“任意乘数”仍指整份程序固定的乘数，不包括两个运行时变量相乘。
后面的任意编码键通用模型允许运行时变量乘法，二者范围不同。

实际新查询如下。次数存在包含和重叠，不是互不相交的全局程序分类。
每条查询保存实际初始样本；其中加入了低 16 位相同、目标低位不同的输入，
没有用截断后的值执行右移。

| 范围 | 查询数 | 原始 UNSAT | 原始 UNKNOWN |
|---|---:|---:|---:|
| b，扩展集合，SHR16 | 50 | 22 | 28 |
| b，sparse 集合，SHR16 | 50 | 39 | 11 |
| bu/bv，扩展集合，SHR16 | 64 | 52 | 12 |
| bu/bv，扩展集合，SHR7/9 | 128 | 128 | 0 |
| bbranches，任意常量乘数及两个新常量 | 12 | 11 | 1 |
| bu，任意常量乘数及两个新常量 | 32 | 22 | 10 |
| 合计 | 336 | 274 | 62 |

### 关闭“仿射 + 一次右移”的四个超时

所有初始 b、u、v 都是 b 的仿射函数。若新程序只有常量乘数 MAD，以及
恰好一次 SHR16，无论 MAD 有多少条，结果一定形如

```text
P*b + Q*((A*b+B mod 2**32)>>16) + R
```

令 b 依次取 0、65536、131072。括号中的商模 65536 是等差数列，所以结果
低 16 位的二阶差分必须为零。冻结哈希给出的实际目标是

```text
完整输出：1818560943, 687623478, 2954089945
低 16 位：2479, 19766, 54745
二阶差分 mod 65536：17692
```

因此该范围不可能成立，允许无限多个任意预载常量也无济于事。
`audit_joint_affine863.py` 将此证书用于两个任意参数实验：bbranches 的
12 项全部关闭，bu 的 32 项关闭 25 项、仍有 7 项 UNKNOWN。原始 JSON 未改写，
细化结果单独保存在 `joint_direct_arbitrary863_refinement.json`。
336 项原始查询经此细化累计为 278 项关闭、58 项未决；没有五条候选。

证书不覆盖变量乘法、额外 XOR/AND/OR、多次右移或非仿射的前序 live 值。

## 2. 新空间：输入、输出 XOR 编码键都可自由选择

此前跨编码实验只检查固定 C1、C5 的吸收。本轮改为寻找三条程序 P 和任意
uint32 键 I、O，使

```text
P(x) = F(x XOR I) XOR O
F(x) = 9*((33*x+B) XOR (16896*x+D))+C4
```

若找到完整 32 位恒等式，输入键可吸收到既有 C1 常量，输出键则经
`L(t)=t XOR (t>>16)` 转成新的状态编码。具体关系为

```text
C1_new = C1 XOR I
state_encoding_new = C5 XOR L(O)
```

这有机会把四条中段压到三条，但仍须重新构造节点编码、选择器奇偶、最终恢复
和常量初始化，并经真实调度及冻结模拟器验证；公式本身不是周期成绩。

### 通用三条模型：仍然未决

`search_xor_equivalent_middle863.py` 支持 MAD/XOR/AND/OR，任意 SSA 连接、
每条两个任意常量、MAD 的两个任意运行时乘数，以及未知 I、O。
低位投影只用于没有移位的中段，UNSAT 才能排除对应完整 32 位语法；
SAT 必须先穷尽整个低位域，仍不能作为完整程序。

`MAD,MAD,XOR`、`MAD,XOR,MAD`、`XOR,MAD,MAD` 三项低 11 位通用查询均为
UNKNOWN，约 22 秒；没有把它们写成排除。
初次用线程直接共享 Z3 默认上下文在求解前报错，记录为
`middle_free_keys863_thread_attempt.json`；重试使用独立子进程，三项查询正常结束。

正向对照包含：自动发现带非零键的低五位三条程序并穷尽 32 个输入，以及
固定原四条程序在低 11 位和完整 32 位上的验证。前者只标为 reduced-domain。

### 全部编码键不能使 F 成为整数多项式

`screen_free_keys_polynomial863.py` 枚举低 11 位全部 **4,194,304 对 I、O**。
对每一对，检查整数多项式必要条件：在 x、x+h、x+k、x+h+k 上的混合差分
必须被 h*k 整除。49 个逐步保留的矩形约束后，剩余键对为零，用时约 0.09 秒。

这排除了任意次数的纯整数多项式程序在两端任意 XOR 编码下表示 F，因而包括
三条纯 MAD；不排除内部包含位运算的程序。原始每步计数及仿射正向对照在
`middle_free_keys863_polynomial.json`。

### 两种固定仿射结构：任意两端键下均不成立

为减少选择器开销，`search_affine_xor_keys863.py` 单独搜索：

```text
parallel: (A*x+B) XOR (D*x+E)
nested:   D*((A*x+B) XOR K)+E
```

两端键 I、O 及结构内常量均可变。parallel 由 x=0 精确推出 O；
nested 由 x=0 精确推出 E。目标是逐低位可逆的置换，因此 parallel 两个斜率
奇偶相反，nested 两个斜率均为奇数。

parallel 低 11 位查询 **UNSAT，0.364 秒**。nested 的原查询为 UNKNOWN；
增加函数保持的归一化后为 **UNSAT，6.820 秒**，保存在
`middle_free_keys863_nested_normalized.json`，原超时记录保留。

nested 的归一化为：I、O、K 的最高及最低位为零，B 的最高位为零。
输入、输出或内部键取反可吸收到相邻任意仿射变换；最高位翻转则可吸收到加数。
`normalize_nested` 显式实现这些变换，保持暴露函数
`P(x XOR I) XOR O` 不变。测试穷尽小字宽输入并交叉检查完整 uint32 随机实例，
没有把归一化当成未经验证的启发式限制。

两个固定结构的自动正向对照均找到并穷尽验证有效程序。另对低九位为零的
8 组端点键，用旧的精确嵌套结构枚举器检查每组 8,388,608 个归一化参数元组，
均无幸存者；这只是该子范围的独立交叉证据。

以上不排除非仿射 MAD、末 MAD 同时读取 x/前序结果、不同位运算或其他连接，
也不是三条程序或整个 864 内核的最优性证明。

## 3. 内核资源与公开资料复查

`joint_direct863_resource_audit.json` 保存本轮只读核算：算术量仍为 51,665 lane；
1,684 条 LOAD 各自至少有 21 周期的后续依赖，给出 863 的 LOAD 必要界；
全部 851 条 FLOW 至少有 10 周期的后续依赖，给出 861 的 FLOW 必要界。
因此在这些依赖不变时，863 只容纳两条额外的准备阶段 FLOW，新增常量读取也
必须置换已有 LOAD。成对常量读取加 FLOW 的设想不能只按省下的 ALU 报收益，
仍需组合其他图变换并重查启动损失。

本轮再次读取了[官方仓库](https://github.com/anthropics/original_performance_takehome)，
也检索了更快公开实现。排行榜页面和 Josu 博客正文未成功获取；GitHub 日志页
只返回文件元数据，未返回正文。没有新获得可核验的低于 864 实现，亦未核实当前
全网排名。检索结果和访问失败不能当成不存在更快实现的证明。

## 验证和复现

**19 项测试通过，2.383 秒**。覆盖完整已知程序、随机程序的冻结 ISA/整数/Z3
交叉执行、任意新常量和乘数、三点证书、编码键归一化、NumPy 筛选与独立小域
枚举、正向对照及低位 SAT 的状态区分。提交程序未变，未重复既有的官方 9 项和
270 例扩展内核测试。

```bash
PYTHONPATH=/tmp/aopt-synthesis:. python3.12 -m codex.search_joint_tail863 \
  --stage bu --profile sparse --free-factors --free-constants 2 \
  --output /tmp/joint-arbitrary.json
PYTHONPATH=/tmp/aopt-synthesis:. python3.12 -m codex.search_xor_equivalent_middle863 \
  --sequence 'mad,^,mad' --output /tmp/free-keys.json
PYTHONPATH=/tmp/aopt-synthesis:. python3.12 -m codex.search_affine_xor_keys863 \
  --family nested --seconds 35 --check-ms 32000 --output /tmp/nested.json
PYTHONPATH=/tmp/aopt-solver:/tmp/aopt-synthesis:. python3.12 \
  -m codex.screen_free_keys_polynomial863 --output /tmp/free-key-polynomial.json
PYTHONPATH=/tmp/aopt-solver:/tmp/aopt-synthesis:. python3.12 \
  -m unittest codex.test_free_middle_keys863 codex.test_joint_tail863
```

864 源码 SHA256 仍为
`98d0f4ce1ecae152378e31ddae70d52f0ed4cb6ace5b027ce191d4027346a884`；
865 备份仍为 `58ce2efca049a18ef343ba82ceaeb72fd74f638bd7c8bbc23a14e6e796d72f44`。
