# 864 以下：地址与常量交换、任意 XOR 编码的完整无移位检查

**可执行最好成绩仍为 864 cycles；本轮没有找到 863 程序。**
提交内核、864/865 独立副本、模拟器及冻结测试均未改变。这里的周期下界、
低位域 SAT、串行正确性回放都不是新的竞赛成绩。

本轮两项主要结果：42 个不同配置的真实指令图均不能达到 863；对单个中段
输入，即使输入、输出 XOR 编码键任意选择，三条 `{MAD,XOR,AND,OR}` 指令的
全部 64 种顺序仍被排除。后者只在明确限定的语法内给出四条下界，
不包含移位、额外 live 值或全局布局变化，也不证明整个内核最优。

## 1. 实际构图：用地址计算换 FLOW，再改变常量物化

新增 `research_flow_stage863.py`，直接修改研究 IR，所有指令都来自冻结 ISA。
第 7 层的二位地址数字可由两次选择

```text
select(b5, select(b6,3,2), b6)
```

改为 `2*b5+b6`，其中 b5、b6 均为 0/1。这每组增加 8 个算术 lane，
释放两条 FLOW，可与常量广播替换联合使用。实验还包含：

- 将常量的八次 STORE 加一次 vload，用于替代原生向量定义。
- 用三个相同源字和一次 vselect 构造八个相同字；实际复制开销也计入。
- 为失去向量别名的标量消费者重新构造独立根，避免错误地免费复用已删除值。
- 将已有常量 LOAD 换成 FLOW；将输入指针 2318 改为 `2310 OR 10`。
- 将周期 0 的一个 LOAD 改为读取 4097，同时重新构造第二个输入指针。

40 次组合扫描与 4 次提前加载实验，合计 44 次评估、42 个不同配置。
空配置的两个扫描入口各重复一次。每次均重新清理指令、展开可选标量/向量
模式、重建依赖，检查前六周期、后三周期及 LOAD/FLOW/STORE 的必要界。
另重新构造全部配置并对指令及操作数角色取哈希，确认确有 42 个不同的图，
记录为 `flow_stage863_graph_audit.json`。

| 必要周期下界 | 不同配置数 |
|---|---:|
| 864 | 8 |
| 865 | 22 |
| 866 | 10 |
| 867 | 2 |

代表结果如下。C 表示 `ALU + 8*VALU`，不是最终调度长度。

| 配置 | C | 前六周期必要空闲 | LOAD 下界 | 总下界 |
|---|---:|---:|---:|---:|
| 864 原图 | 51,665 | 52 | 863 | 864 |
| 十个常量全部 STORE/reload，融合一组数字 | 51,590 | 220 | 867 | 867 |
| 十个常量全部三字 FLOW 物化，融合五组数字 | 51,644 | 156 | 863 | 866 |
| 周期 0 加载 4097 | 51,658 | 68 | 863 | 865 |
| 周期 0 加载 4097，另外暂存常量 9 | 51,651 | 68 | 863 | 864 |

省下 75 lane 的配置仍失败，原因是启动延迟与关键 LOAD 增多。
这说明不能只将省下的广播数量换算成周期收益。
原始数据为 `flow_stage863_scan.json`、`flow_stage863_early4097.json`。

另对七个代表配置执行了完整内核正确性回放，每个使用三个随机 uint32
树与输入，共 **21 次冻结模拟器验证通过**。scratch 使用 1,027–1,108 字，
未超过 1,536。包括全十常量两种极端构造，验证了改图、地址、连续视图及
内存暂存的实际可执行性。这里使用拓扑串行日历；JSON 的 `verified` 数字
是该串行日历的周期数，绝不是优化后的竞赛分数。

验证记录见 `flow_stage863_controls.json`、`flow_stage863_extreme_controls.json`
及提前加载文件中的 `serial_control`。

## 2. 对这一组交换方式，乐观预算也只允许省 49 lane

863 在原首尾必要空闲 52/116 下，要求 C ≤ 51,612，亦即至少省 **53 lane**。
`bound_flow_materialization863.py` 检查一个明确限定的组合菜单：

- 原有十个常量向量与初始根向量可改变，ones 保持原定义。
- 每个向量可用原生 8C、一次暂存读取、2C+1FLOW 三字构造；还允许每对
  6C+1FLOW 的乐观成对构造，后者仅参与预算，没有声称已实现。
- 最多四个旧 LOAD 可交换：前三个改 FLOW，第四个用一次 ALU 构造输入指针。
  关键 LOAD 总量不得增加；863 仅有两条额外 FLOW 容量。
- 融合每组地址数字付出 8C、收回两条 FLOW。
- **额外放宽**：免费提供六个独立标量根，其中包括实际仍需构造的 4097；
  无条件赠送删除向量 256 的 8C 收益；忽略所有新增启动延迟。

1,029 个整数配置的枚举上界为 **49 lane**，独立 CP-SAT 模型返回
`OPTIMAL=49`。最乐观仍比所需收益少 4 lane。
原始证据为 `flow_stage863_menu_bound.json`。

这不是所有常量优化的上界。它不包含删除其他锚点包、不同记录布局、新的哈希
恒等式或新的物化原语。十常量暂存实验省下 75C 也不矛盾：它增加了关键 LOAD，
不属于要求 863 可容纳的上述菜单。

## 3. 三条无移位中段：任意两端 XOR 编码下仍不可能

定义所有运算模 2^32：

```text
F(x) = 9*((33*x+0xe9f8cc1d) XOR (16896*x+0xaccf6200))+0xfd7046c5
P(x) = F(x XOR I) XOR O
```

I、O 均可任意选择。MAD 允许两个运行时操作数相乘，并允许任意预载常量。
输入键有机会吸收到上一阶段的 C1；输出键可通过末尾线性变换
`L(t)=t XOR (t>>16)` 转成状态编码。这样扩大了此前固定键的搜索范围。

所有排除在低 11 位给出必要条件。由于这里只允许 MAD/XOR/AND/OR，任何完整
uint32 程序都会投影为相同语法的低位程序；低位 UNSAT 可以排除完整程序。
低位 SAT 则仍须提升和完整验证。本轮未产生新的完整程序。

| 三条指令的分类 | 顺序数 | 排除依据 |
|---|---:|---|
| 三条 MAD | 1 | 所有输入、输出键的整数多项式矩形证书 |
| 两条 MAD、一条 XOR | 3 | 新的完整连接分类、分区 UNSAT、已有归一化固定结构证书 |
| 两条 MAD、一条 AND/OR | 6 | 无进位矩形、末操作数包含关系、完整中间操作分区 |
| 一条 MAD、两条位运算 | 27 | 19 项直接 UNSAT；8 项原始 UNKNOWN 由结构证书排除 |
| 三条位运算 | 27 | 位局部性反例，任意两端键都不能消除 |
| 合计 | 64 | 全部排除；原始 UNKNOWN 保留 |

完整证据索引与源文件 SHA256 在 `free_keys863_complete_grammar.json`；
重建审计脚本为 `audit_free_key_frontier863.py`。
原四条 `MAD,MAD,XOR,MAD` 有已保存的完整 uint32 等价验证。
因此，在这套单输入语法中，**对任意编码键搜索后，最少仍为四条**。
加、减、乘都能用单条 MAD 表示，故单独加入它们不会改变此结论。

### 3.1 XOR 分区如何覆盖原先的超时

40 个新增通用分区查询得到 34 个原始 UNSAT、6 个 UNKNOWN；查询之间包含
重叠，不是 40 个互斥类别。逐步固定首 MAD、XOR 来源、末 MAD 角色后，
剩下的 `D*((A*x+B) XOR K)+E` 超时直接归入此前已完成的归一化结构证书。

`audit_free_key_orders863.py` 为三个顺序枚举 42,300 种连接方式，并将每一种
映射到实际证据。此数含常量命名对称和死代码，**不是不同函数的数量**。
首 MAD 分为仿射、x²+B、x²+x；末 MAD 的任意乘数、加数读取均被覆盖。

### 3.2 无进位矩形为什么也约束位运算前缀

整数多项式的混合差分
`P(b+h+k)-P(b+h)-P(b+k)+P(b)` 必须被 h*k 整除。
再限制 h、k 为不同二进制位，且 b 在这两位均为零、四点不溢出。
任何逐位变换 T(x)，包括多个 XOR/AND/OR 的组合，在这些矩形的两轴上
分别只改变 0 或 ±h、0 或 ±k。因此，任意整数多项式
`P(x,T1(x),T2(x),...)` 也满足这个必要条件。

新增筛选分别检查 `F(x XOR I) XOR O` 与再 XOR x 的版本。
每项枚举 **4,194,304 对低 11 位 I、O**，均无幸存键。
原始矩形与逐步淘汰数保存在 `free_keys863_bypass_polynomial.json`。

这既排除先位运算后两条 MAD，也排除两条逐位前缀后一条 MAD。
对 `XOR,MAD,XOR`，最后再 XOR 的原输入或 `x XOR K` 可化成同一个
XOR x 旁路筛选，常量 K 吸收到任意输出键。

### 3.3 末条 AND/OR 的必要包含关系

在低 11 位，F 是置换且翻转输入最高位会翻转输出最高位；任意 I、O 保留
这些性质。若最后是 AND，另一个操作数必须包含目标的每个 1；若是 OR，
另一个操作数不能包含目标没有的 1。

当另一个操作数是首 MAD 的 `A*x+B`：A 为奇数时，每一位与目标一样恰有
一半输入为 1，包含关系强制两者完全相等，而任意键下的仿射表示已被多项式
证书排除。A 为非零偶数时，操作数不随输入最高位变化，但该输出位仍均衡；
目标的最高位翻转迫使该位永远是中性值，产生矛盾。A=0 只允许全 1 的 AND
掩码或全 0 的 OR 掩码，结果退化为多项式。

首 MAD 的另外两类也可排除：x²+B 在 x 与 x XOR 1024 上不变，但在
x=0、32 上最高位相反，不可能恒为中性位。x²+x 的最低位恒零，其余每位
均衡，且在 x 与逐位取反的 x 上相等；AND 在最低位已矛盾，OR 则要求
目标在这对输入上仅最低位可能不同。但 `F(0) XOR F(2047)=1899`，反例不依赖
端点键。审计对这些低位域性质作了完整枚举检查。

对 `XOR,MAD,AND/OR`，另一操作数是 x、x XOR K 或常量。前两者均为置换，
包含关系迫使目标等于该逐位函数；常量仅能是中性掩码。两种情况分别由位
局部性或无进位矩形证书排除。

### 3.4 中间 AND/OR 的最后一个超时

AND/OR 的六个粗查询均为 UNKNOWN。细化首 MAD 和中间来源后，仅剩仿射
输入与常量掩码；再划分末 MAD 的乘数、加数后，只剩
`D*((A*x+B) OR K)+(A*x+B)` 的一个超时。

整个表达式只通过 t=A*x+B 读取输入，目标的可逆性要求 A 为奇数。
输入键取反可吸收到 `A -> -A, B -> B-A`，故 I 最低位也可归零。
加入这些必要条件后，该项 **UNSAT，约 8.95 秒**。
AND/OR 共 47 次原始查询为 34 UNSAT、13 UNKNOWN，全部超时由更细证据覆盖。
`audit_free_keys_lattice863.py` 还分别枚举了两个中间顺序的 13,500 种连接。

不足三条及含死代码的程序不会遗漏：零 MAD 程序是逐位函数；两条纯 MAD
是多项式；至多两条且只有一个 MAD 的程序可补一条恒等 MAD，落入已关闭的
两 MAD 顺序。所有这些结论仍只属于这一语法。

## 4. 保留四条中段，但尝试吸收最终恢复 XOR

另一路不减少中段指令数，而是令其输出包含
`PRE_C5 = 0xb55a4f09 XOR (0xb55a4f09>>16) = 0xb55afa53`，使末尾恢复 XOR
能够被线性尾变换吸收。输入键依然任意选择。

| 固定结构 | 新查询 | 结果 |
|---|---:|---|
| G*((A*x+B) XOR (D*x+E))+H | 低 11 位，G 任意或固定 9 | 两项 UNSAT |
| (A*x+B) XOR ((D*x+E) AND K) | 低 11、12 位 | 两项 UNSAT |
| D*(x XOR ((A*x+B) AND K))+E | 低 11、12 位 | 两项 UNSAT |

每个结构均有自动发现已知函数的低位域正向对照；它们只验证搜索模型，
不构成新内核。结果在 `absorb_restore863_*.json`、`masked_restore863_*.json`。
这些实验不覆盖所有四条程序或联合前后阶段的改写。

## 5. 公开资料、验证与仍然开放的方向

本轮成功通过 GitHub 读取了此前未取到正文的
[Josu 原始研究日志，第 2035–2065 行](https://github.com/josusanmartin/josusanmartin.github.io/blob/master/vliw/chat.log#L2035-L2065)。
这份历史日志同样记录了 864 附近的算术与访存约束、4097 初始化对启动的影响，
以及地址锚点构造不能只按局部省指令数估计收益。它提供了交叉参考，没有提供
已验证的新 863 程序，也不能用于认定当前排行榜仍以 864 为首。

新增验证覆盖：归一化函数保持性、多个逐位前缀的矩形必要条件、小字宽独立
包含关系穷举、完整低位域正向对照、冻结 ISA/整数模型交叉执行。
**23 项测试全部通过，耗时 3.406 秒**，环境版本与提交文件校验见
`flow_encoding863_validation.json`。
75 个分区正向程序和另外三个单 MAD 正向程序均由求解器自动恢复。
提交内核未变，因此不重复既有官方 9 项与 270 例扩展内核回归。

最值得继续保留的方向是：利用此前已计算的额外 live 值联合改变多段哈希；
包含移位或其他 ISA 操作的编码方案；改变树记录及其后继地址的布局；
通过新物化方式或删除锚点包同时省下算术与关键 LOAD/FLOW。
这里没有将这些更大范围标为不可能。

### 复现入口

实验依赖 NumPy、Z3、OR-Tools。当前环境分别放在 `/tmp/aopt-solver` 与
`/tmp/aopt-synthesis`，换环境时需要安装对应依赖。

```bash
PYTHONPATH=/tmp/aopt-solver:/tmp/aopt-synthesis:. python3.12 \
  -m codex.research_flow_stage863 --output /tmp/flow-stage-scan.json
PYTHONPATH=/tmp/aopt-solver:/tmp/aopt-synthesis:. python3.12 \
  -m codex.research_flow_stage863 --controls --output /tmp/flow-stage-controls.json
PYTHONPATH=/tmp/aopt-solver:. python3.12 \
  -m codex.bound_flow_materialization863 --output /tmp/flow-stage-menu.json
python3.12 -m codex.audit_free_key_frontier863 --output /tmp/free-key-frontier.json
PYTHONPATH=/tmp/aopt-synthesis:. python3.12 \
  -m codex.search_xor_equivalent_middle863 --sequence 'mad,|,mad' \
  --first-form affine --middle-source key --last-operands 2,3,1 \
  --seconds 25 --check-ms 23000 --output /tmp/or-final-refinement.json
PYTHONPATH=/tmp/aopt-synthesis:. python3.12 \
  -m codex.search_absorb_restore863 --output /tmp/restore.json
PYTHONPATH=/tmp/aopt-solver:/tmp/aopt-synthesis:. python3.12 \
  -m unittest codex.test_restore_frontier863 codex.test_free_middle_keys863
```

864 源码 SHA256：
`98d0f4ce1ecae152378e31ddae70d52f0ed4cb6ace5b027ce191d4027346a884`。
865 备份 SHA256：
`58ce2efca049a18ef343ba82ceaeb72fd74f638bd7c8bbc23a14e6e796d72f44`。
