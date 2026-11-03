# 允许八个前序值的末段两指令搜索

后续联合改写实验见 [六条变五条构造搜索](SUB863_JOINT_TAIL_20261024.md)。

**可执行最好成绩仍为 864 cycles，没有得到 863 内核。**
本轮扩大了局部改写的输入范围：每个候选都可以免费读取实际已经存在的前序值。
完成了末段两条算术指令、14 个操作码的 **196/196** 项排除。
这是针对一个明确局部边界的结果，不是整个程序的下界；它不能排除把更多阶段
一起改写、改变节点数据布局、使用访存或 FLOW 指令的方案。

## 1. 实际输入与目标

所有运算均为 uint32。唯一独立变量 p 是进入本轮哈希的状态，前序值严格由同一个
p 推导，不允许把中间量当成互不相关的输入。

```text
a = 4097*p + 0x7ed55d16
s = a >> 19
mix = a XOR 0xc761c23c
b = mix XOR s
u = 33*b + 0xe9f8cc1d
v = 16896*b + 0xaccf6200
z = u XOR v
e = 9*z + 0xfd7046c5
G(p) = e XOR (e >> 16)
```

完整原哈希是 `G(p) XOR 0xb55a4f09`。本轮的末段目标为 G，原实现从 z 到 G
需要 MAD、SHR、XOR 三条指令；尝试用两条替代。免费操作数池为
`p,a,s,mix,b,u,v,z`，另允许任意字面量和前一条新指令的结果。

操作码为 `+,-,*,MAD,XOR,AND,OR,SHL,SHR,UDIV,UREM,CDIV,LT,EQ`。
MAD 是一次向量乘加的标量表达式；UDIV/UREM/CDIV 使用无符号整数，除数不能为零。
CDIV 按冻结模拟器语义先以未溢出的整数计算 `a+b-1`，再除以 b。
查找是必要条件模型，尚未涉及真实寄存器连续性与调度。

`search_full_live_shift863.py` 也尝试了较早的中段 F：允许免费使用
`p,a,s,mix,b`，将四条缩为三条，指令中含一个 MAD 和一个 SHR。
这批 48 项的原始结果为 **10 排除、38 UNKNOWN**，保存于
`full_live_middle_shift863_results.json`。这与先前单输入版本 48/48 的结果不矛盾：
本轮操作数范围更宽，且始终保留 prior 的全部高位供新右移使用。

## 2. 反向构造测试输入

随机输入不是唯一的筛选方式。目标 G 是一个置换，可以精确求逆，构造输出恰为
0、1、全一、最高位和各进位边界的真实输入。

先利用 `L(x)=x XOR (x>>16)` 自反，从 G 得到 e。再反演外层奇数乘加得到 z。
令 `K=0xaccf6200-512*0xe9f8cc1d (mod 2**32)`，则

```text
z = u XOR (512*u+K)
```

从 u=0 开始迭代 `u=z XOR (512*u+K)` 四次，每次确定更多九位，便恢复完整 u。
之后依次反演乘数 33、XOR-SHR19、乘数 4097，即得 p。
实际代码同时验证目标的两个方向，并在 20,202 个完整输出样本上检查逆函数。

末段最初含 SHR 的 27 种两指令顺序得到 24 UNSAT、3 UNKNOWN。
增加输出反向样本后，余下 `SHR/MAD`、`CDIV/SHR`、`SHR/CDIV` 全部 UNSAT。
原始和加强查询分别保存在 `full_live_tail_shift863_results.json`、
`full_live_tail_shift863_refinement.json`，没有修改历史 UNKNOWN。

## 3. 同一低位输入类，目标却有八种输出

选择 `a=j*65536, j=0..7`，再反演为真实 p。此时 s=0，并且八个免费操作数
各自的低 16 位在八组输入上都相同：

```text
p,a,s,mix,b,u,v,z 的共同低16位：
746, 0, 0, 49724, 49724, 54745, 55808, 4057

目标 G 的低3位：6, 5, 0, 7, 2, 1, 4, 3
```

加减乘、MAD、XOR、AND、OR 保持模 65536 的同余关系。
比较指令只有 0、1 两种结果，最多把每个同余类分为两类。
SHL 也最多分成两类：在一个类内移位量低 16 位固定；若实际位数过大，低 16 位
结果为零，否则其位数及来源低 16 位均固定，只能给出另一个确定结果。

所以由上述十种操作码组成的任意两条指令，输出低 16 位最多有四种值，
无法实现有八种低 3 位值的 G。这一次排除 **100 个顺序**，包括变量位数 SHL。
独立完整 32 位 SMT 查询验证了“同余输入经 SHL 不会出现三个不同低位输出”的性质。

## 4. 高输出值强制末条操作直接复制分子

当输出严格大于 2**31 时，以下四种末条操作均要求输出等于分子：

- SHR：位数必须为零。
- UDIV、CDIV：除数必须为 1；除数至少为 2 时，最大结果分别不超过
  2**31-1、2**31。
- UREM：余数比除数小；若余数大于 2**31，发生过一次减除数就会使原分子
  超过 uint32 上界，故余数必须就是原分子。

这四个条件及其精确形式均由独立完整 32 位 SMT 检查，另与完整小整数域枚举对照。

最终分子若取原有免费值，每个来源都有实际反例；若取常量，两个不同高输出值
已经矛盾。于是分子必须来自第一条新指令。
对全部 14 种第一指令，使用 61 个真实高输出反向样本检查“前缀直接实现 G”这个
必要条件，全部 UNSAT。加上布尔末条输出、原始完整查询和第 3 节条件，
将 196 种组合缩至 24 种未决。

高输出前缀的 MAD、SHR、UREM、CDIV 正向对照均为 SAT，并在冻结模拟器逐个
执行样本复核。最初的 UREM 对照误包含不可由余数产生的 0xffffffff 输出，
被正确拒绝；随后改为 `x<0xffffffff`，原错误及原因保留在结果文件。
这不影响目标函数查询。

## 5. 所有非零 uint32 除数的区间枚举

剩余顺序主要是先除法/余数，再乘加。短时和加强查询记录分别为
`full_live_tail_division863_results.json`、`full_live_tail_division863_refinement.json`。
加强后仍有四项 UNKNOWN：`UREM/MAD`、`UDIV/MAD`、`CDIV/MAD`、`CDIV/ADD`。
以下证据独立关闭这些分支，没有把超时解释为不可行。

### 5.1 第一条的除数只能是常量

八个免费操作数各自都有一个合法 p 使其为零。完整程序不能在这些输入上执行
除零，所以第一条的除数必须是任意非零字面量 D。
这些零点实际保存并复算；UDIV、UREM、CDIV 均适用。
若分子也为常量，第一结果固定，第 3 节的八点已经将其排除。

### 5.2 最后 MAD 只能是商的仿射函数

在第 3 节的同余类上，除法结果记为 q。除 q 以外的所有来源，低 16 位均固定。
最终 MAD 的全部操作数组合因此属于以下三类：

```text
A*q+B；q*q+C；q*q+q       (mod 65536)
```

第二类在模 4 下最多有两个取值，第三类恒为偶数。实际 G 遍历全部四种模 4 余数，
故只剩任意仿射系数 A、B。这是放宽条件，不要求 A、B 在真实操作数池中可取得。

### 5.3 用商不变的区间覆盖每一个 D

对固定分子 S，`floor(S/D)` 随 D 单调，只有约 `2*sqrt(S)` 个不同区间。
构造所有边界：小除数逐个保留，再加入 `floor(S/k)+1`（k<=sqrt(S)）。
对八个分子取边界并集，并加入 1、2**32，即把所有
`D=1..2**32-1` 分成不重叠的闭区间。

程序对每个区间的两端独立求商并检查相等，利用单调性确认整个区间的商向量不变；
同时检查所有区间长度之和精确等于 4,294,967,295。
CDIV 使用 `ceil(S/D)=floor((S-1)/D)+1`，S=0 单独保持为零。

在每个区间，选择目标差值为奇数的一对点。如果对应商差为偶数，不能拟合；
否则商差可逆，唯一决定 A 模 65536，第一点再决定 B。逐个检查其余六点即可。
这里没有把模环当成域：只对已经确认是奇数的量取逆。

八种分子来源、floor/ceil 两种函数共检查 **3,233,054 个区间**，
全部无幸存者，主枚举及正向对照约 **1.113 秒**。
每个来源均用真实 `17*quotient+23` 作为正向对照，16/16 都保留了真实除数。

### 5.4 余数也被同一证据覆盖

八个 S_i 低 16 位相同，记为 s；令 `Q_i=floor(S_i/D)`，则

```text
R_i = S_i % D = s-D*Q_i                    (mod 65536)
A*R_i+B = (-A*D)*Q_i+(A*s+B)              (mod 65536)
```

所以余数的任意仿射函数必然也是商的某个仿射函数，已经被 floor 区间枚举排除。
最终加、减、乘都可用一次 MAD 表示，因此 CDIV/ADD 也随之关闭。

完整结果在 `full_live_tail_divisor_intervals863_results.json`；
`full_live_tail_closure863_results.json` 汇总原始查询与独立证据，最终 **196/196**。

## 验证与复现

组合验证 **26 项测试全部通过，7.419 秒**。覆盖：冻结哈希与完整 32 位目标、
逆函数、自动发现的完整 32 位正向控制、零点、移位同余性质、高输出恒等条件、
真实两指令程序、MAD 的全部小字宽操作数角色，以及区间算法与逐个除数、逐个系数
暴力枚举的独立对照。全部原始 UNKNOWN 保留。

```bash
PYTHONPATH=/tmp/aopt-solver:/tmp/aopt-synthesis:. python3.12 -m codex.screen_live_division_tail863 \
  --output /tmp/tail-divisors.json
PYTHONPATH=/tmp/aopt-solver:/tmp/aopt-synthesis:. python3.12 -m codex.audit_full_live_tail863 \
  --searches codex/full_live_tail_shift863_results.json \
    codex/full_live_tail_shift863_refinement.json \
    codex/full_live_tail_division863_results.json \
    codex/full_live_tail_division863_refinement.json \
  --division-certificate /tmp/tail-divisors.json --output /tmp/tail-closure.json
PYTHONPATH=/tmp/aopt-solver:/tmp/aopt-synthesis:. python3.12 -m unittest \
  codex.test_live_divisor_intervals863 codex.test_full_live_shift863 \
  codex.test_full_live_tail_audit863 codex.test_hash_transport863
```

扫描时曾修复两处记录问题：以减号开头的 opcode 串要使用 `--sequence=...`，
错误结果和正常结果的顺序字段统一为列表。中段两次失败调用保留在
`failed_attempts`，修正后均已续跑；没有把它们计为算法排除。

根目录提交、864/865 独立副本、模拟器和冻结测试没有改动。
仍有价值的后续范围是把整个中段与末尾 XOR-SHR 联合改写，例如六条缩为五条；
局部“中段三条不够”和“末段两条不够”不能合并为整个六条的最优性证明。
