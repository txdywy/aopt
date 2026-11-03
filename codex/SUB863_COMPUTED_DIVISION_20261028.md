# 计算前缀后的除法：范围枚举、奇前缀碰撞与小分子证书

**最好可执行代码仍为 864 cycles，本轮没有得到 863。** 本轮处理上一轮明确
保留的“首条计算 P，第二条除法，最后带旁路运算”的若干范围。提交内核及
864/865 独立版本、冻结评测文件未变。结论仍是指定九字、根预备常量的局部
语法排除，不是全内核下界，也不是所有三指令 ISA 的完整枚举。

记号与已有字同 `SUB863_DIVISION_COMPARISON_20261028.md`：

```text
V=[e,z,u,v,b,prior,a,s19,mix]
Y=4097*(e XOR(e>>16) XOR N)+C0  (uint32)
raw_root=0x12345678, N=0xa76e1971
T=MAD/XOR/AND/OR，P 是一条 T 的结果，q 是第二条除法的结果
```

## 1. 从 18 个 UNKNOWN 转为完整逆系数枚举

若最后是 `Y=G*q+W`，W 为九个已有字之一，G 必须为奇数：翻转 b16 时 W 的
LSB 不变，目标奇偶改变，偶数 G 无法实现。令 `A=G^-1`，则
`q=A*(Y-W)`，前缀如何计算暂且完全放开。

有两个独立必要条件：

- 任意 uint32 分子被非零常数 D≥2 做 floor/ceil，商都 ≤2^31。
- 任意 DIV/REM/CDIV 的结果若 >2^31，必等于分子。因此当分子是同一个预备
  常数时，所有高商只能取同一个值。

直接向 Z3 提交九来源 ×两条件，18 项各 15 秒均为 UNKNOWN，原始文件保留。
随后 `enumerate_middle_residual863.cpp` 对每个来源完整枚举全部奇数 A，
NEON 只加速模 2^32 乘法与计数，不改变定义。删除重复残差后，每个 A 只要
产生两个不同的高 q，就违反第二条件；第一条件是它的更强版本。

**九来源各 2^31 个参数，合计 19,327,352,832 个，全部无剩余模型。** 单来源
约 11 秒，四线程执行。原始 UNKNOWN 未改写；新证据是单独的完整枚举结果。
有限数据、对应合法主变量、二进制与源码指纹、每来源已检查参数数均保存。

这排除带已有字旁路的常数乘数尾部，涵盖任意计算分子/预备除数的 floor、
ceil，以及预备分子/计算除数的三种除法。REM 的计算分子情形分两半：
D≤2^31+1 时余数也满足上述范围；D 更大时 `q=P-D*k, k∈{0,1}`，八状态
切片上 T 部分的低 16 位固定，输出最多两种低位，少于目标八种。D=1 的
floor/ceil 则退化为无向下传位的 T 运算。

另有直接的零分子反例：九个初始分子各有合法零像，在这些点目标不等于任何
已有字。因此 `q*任意系数+已有字` 的大量计算除数情形无需枚举参数。

## 2. 一个恰为 2^31 的混合差分

新找到的四个合法 b 是：

```text
[3597240495, 3597174959, 3597371567, 3597306031]
```

九个已有字在四点上低 16 位相同，混合差分
`Δf=f0-f1-f2+f3 (mod2^32)` 都是 0，而 **ΔY=2^31**。
任意两个输入增量的乘积都被 2^32 整除，故一条任意 MAD 产生的 P，以及这些
字与 P 的任意整数多项式 R，也有 ΔP=ΔR=0。

`audit_middle_mad_quotient863.py` 因此关闭两个完整族：

```text
P=任意一条 MAD；q=DIV/REM/CDIV(P,D)
Y=R(已有字,P,预备常量)+q
或 Y=P+G*q
```

第二式的 G 必为奇数，奇数乘法的逆仍把 2^31 映到 2^31。若 D≥4 做 floor，
或 D≥5 做 ceil，商的四点混合差分绝对值小于 2^31，立即矛盾。剩下小的奇数
D 对 `D*q=P±余项` 取混合差分，左边为 2^31，右边绝对值至多 2(D−1)，
仍矛盾。D=2/4 时商的最低位只读 P 的低三位，不能产生目标的 b16 依赖。

REM 的偶数 D 直接不传递所需奇偶信息；奇数 D 由范围先迫使 D>2^30，再写成
`q=P-D*k, k≤3`。Δk 的绝对值至多 6，而奇数乘法不可能把它变为 2^31。
这覆盖所有非零 uint32 常数 D，包括最后 MAD 以 q 为加数的所有多项式乘积。
它不覆盖一般的计算除数。

## 3. 首条 P 作为最后乘数：为所有根参数寻找共同碰撞

考虑 `Y=P*q+W`，W 是已有字，q 只依赖 P 与根预备常量。P 在 b16 对上
奇偶相同，目标却翻转，所以有效 P 必须处处为奇数。首条 MAD/位运算的
702 个规范连接中，仅 **64 个私有常量奇偶分支**能满足此必要条件。

`audit_odd_prefix_collisions863.py` 为每分支找两个输入，使 P 在**任意同奇偶
根参数**下相等，但目标减 W 不相等。由此不论 q 如何计算，都无法修复旁路。

MAD 把 `K_i=p_i+2*t_i` 代入后，关于 t0/t1 次数至多二；六个 Newton 网格点
完全确定其整数前向差分，不需要除以非单位 2。位运算则逐位检查所有参数
位的组合。随后 `verify_odd_prefix_collisions863.py` 对每对输入独立提交完整
32 位参数反例查询，**64 次全部 UNSAT**。

初次生日碰撞发现关闭 63/64，原记录保留。最后一项是 `P=e²+z`：

```text
P=81*z²+(18*C4+1)*z+C4²
z'=-z-(18*C4+1)/81  (mod2^32)
```

精确反射给出共同 P 的输入对，关闭最后一项；没有把随机未发现碰撞当作证明。
该结果同时排除 `q=DIV/REM/CDIV(P,D)` 和 `q=DIV/REM/CDIV(C,P)` 的这一尾部，
所有 D/C 都可预备。若分子是另一个初始字、P 作除数，则用上一节的零分子
反例关闭同样的已有字加数情形。

## 4. 计算除数，最后为 P*q+H

`audit_small_numerator863.py` 允许任意根预备 H。初始分子 V=0 时将 H 钉住；
对七个双射来源，再构造 V 为 1..128 的合法输入。

- floor：V=1 时 P*floor(1/P) 只能为 0/1，真实目标减 H 却大于 1。
- ceil：P≤n 时乘积至多 2n−1；更大残差迫使 P>n、商为 1，直接钉住 P。
- REM：P≤n 时乘积小于 n²；更大残差迫使 P>n、余数为 n。取奇数 n 后，
  可从模 2^32 乘积唯一反推 P。

反推得到偶 P 的分支立即失败；其余强制值既不属于五基础字的 21 系数二次
多项式放宽，也不属于任意单条位运算。v、s19 两个非双射分子另用两个零像
给出不同目标，直接否定同一个 H。九来源三种除法全部关闭。

额外关闭融合尾部 `q*已有字+q`：除了 v，其他已有字的加一因子都会在合法
奇偶类变偶。v+1 恒奇，故可反推 q；219 个高 q 要求计算分子 P 已经等于 q，
但任意单条 MAD/位运算均无法匹配。常量分子则不能产生这 219 个不同高值。
平方 q 的尾部因模 4 像太小，以及 `P*q+P`、`P*q+q` 的零像/奇偶限制，也在
最终账本中单独记录。

## 5. 验证与下一步

**54 项研究回归通过**，含本轮 9 项新测试。C++ 后端在 3/8/12/17 位上与
NumPy 完整奇数域逐项比较，包含乘法回绕、阈值恰等于半字、重复残差拒绝及
阳性控制。Newton/逐位网格与小字宽全参数穷举比较；60 组三指令程序在冻结
模拟器上验证混合差分、除法及进位恒等式；小分子强制过程保留真实可行控制。

提交、864、865 源文件指纹未变，`tests/` 与 `problem.py` 对 origin/main 的
diff 为空。本轮未重跑官方测试，成绩沿用同一提交代码之前的九项 864 记录。
没有产生新完整日历，也没有把串行回放或必要条件通过算成比赛成绩。

剩余重点进一步缩小为：计算除数下的 `q+乘积` 或 `P+G*q`，尤其是有初始字
旁路的常量分子；以及首条位运算、预备除数、尚未被归约的加数尾部。还包括
多次除法、除法与移位/比较、更多 live 字、较大联合窗口、状态编码和其他
内核算法。没有声称这些方向必有解，也没有把九字局部证据推广为全局最优。

```bash
clang++ -O3 -std=c++20 -pthread codex/enumerate_middle_residual863.cpp -o /tmp/residual-enum
python -m codex.run_middle_residual_enum863 --binary /tmp/residual-enum \
  --workers 4 --output /tmp/residual-enum.json
python -m codex.audit_middle_mad_quotient863 --output /tmp/mad-quotient.json
python -m codex.audit_odd_prefix_collisions863 --output /tmp/odd-prefix.json
python -m codex.verify_odd_prefix_collisions863 --source /tmp/odd-prefix.json --output /tmp/verified.json
python -m codex.audit_small_numerator863 --output /tmp/small-numerator.json
python -m codex.audit_middle_runtime_factor863 --output /tmp/runtime-factor.json
python -m codex.close_computed_division863 --output /tmp/closed.json
python -m unittest codex.test_middle_residual863 codex.test_division_comparison863 \
  codex.test_runtime_shift_core863 codex.test_joint_live_boundary863 \
  codex.test_early_boundary863 codex.test_live_boundary863 \
  codex.test_hash_transport863 codex.test_modular_polynomial
```
