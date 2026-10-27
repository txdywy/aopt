# 输出恢复四指令范围关闭，转向 C1/C5 联合改写

**实际可执行最好成绩仍为 864 cycles，没有得到 sub-864 内核。**
本轮关闭了上一轮剩余 176 个四指令结构，建立了联合改写的实际编译器接入，
并继续搜索更大的五指令范围。提交文件、864/865 副本和冻结评测均保持原样。

## 1. 现在能够成立的限定结论

令所有计算按 uint32 回绕：

```text
F(x) = 9*((33*x+0xe9f8cc1d) XOR (16896*x+0xaccf6200))+0xfd7046c5
PRE_C5 = 0xb55afa53
P(x) = F(x XOR I) XOR PRE_C5
```

只有一个 live 输入 x、允许任意预置常量、每条指令属于 MAD/XOR/AND/OR 时，
**即使 I 任意，P 也不能用至多四条直线指令实现。** 已知 F 的四条指令加
一次 XOR 是完整 32 位的五条上界，所以这个明确语法内的最少条数是五。

这不包含移位、除法、其他 live 值、跨轮状态、访存布局或指令调度。
它尤其**不是 864 周期的全局最优性证明**。可执行内核仍可能通过其他改写更快。

上一轮完整枚举了 248,712 个规范连线，已有证据关闭 248,536 个，余 176 个。
本轮 `audit_restore_shallow863.py --last-families --final-pair` 将这 176 个全部关闭。
审计保留原始 UNKNOWN、每项归约依据和源文件 SHA256，没有把超时改写成 UNSAT。

## 2. 新关闭的 176 个连线

| 依据 | 新关闭数 |
|---|---:|
| G*(P1 XOR Q2)+H，任意常数 G/H | 53 |
| G*(P1 XOR Q2)+R2，任意常数 G、二次残差 R2 | 69 |
| G*(P1 AND Q2)+R2 | 13 |
| G*(P1 OR Q2)+R2 | 21 |
| Kissat 对具体连线的必要样本 UNSAT，和上述证据取差集 | 12 |
| 最后四个精确公式族及两种结构归约 | 8 |
| 合计 | 176 |

Pj 表示任意系数、次数至多 j 的整数多项式，不是域上的多项式。
系数使用下降阶乘基底，按 j! 的 2-adic 赋值消除函数等价系数。

`audit_restore_shallow863.py` 对真实指令展开精确符号多项式：x、位运算结果 q、
八个私有常量各自是独立不定元。只有最终式没有 x*q、q² 和高次残差时，才归入
常数 G 乘 q 加二次残差；不会把运行时乘数误当成常量。

四个主要查询在低 11 位 UNSAT，耗时分别约 2.24、91.19、16.69、18.04 秒。
额外单位缩放子类约 52.95 秒 UNSAT。Kissat 检查全部 65 个依赖深度不超过三的
剩余图，60 UNSAT、5 UNKNOWN；其独立新增排除数是表中的 12，不是 60。

### 归一化的适用范围

多项式族对 x→~x、x→x+2^(w-1) 封闭，因此只在这些族内，允许清除任意输入键
I 的最低位和最高位。没有将该限制施加到任意 SSA 连线上。
同时取反/翻转 XOR 两臂可规范其中一个常数；AND/OR 查询未套用 XOR 的对称性。

XOR 残差族还使用置换性的必要条件：

```text
G*(p1+q1)+r1 为奇数
G*q2+r2 为偶数
```

这里 q2、r2 是 x*(x-1) 的系数。它们分别来自最低位和最高位的对角条件。
240 个随机多项式程序、每个穷尽 64 个输入，检查了输入归一化保持 exposed
function；XOR、AND、OR 的已知模型都通过了求解并独立穷尽 2,048 个输入。

### 最后八项

`search_restore_last863.py` 和 `search_restore_final_pair863.py` 检查：

```text
(A*x+B) XOR (C*(x XOR K)+D)                         # 覆盖 3 项
C*u*u+D*u+E, u=(A*x+B) XOR K                       # 1 项
u XOR (C*u+D), u=(A*x+B) XOR K                     # 1 项
C*x+((A*(x XOR K)+B) XOR L)                        # 1 项
```

四族低 11 位分别在约 0.25、1.62、0.32、25.57 秒 UNSAT。
最后一族规范输入时会引入 H；它的**精确轨道**为
`H∈{0,C,C*high,C*(high+1)}`。查询保留这个轨道，既没有错误固定 H=0，也没有
将它当作任意额外常量。119 次逆变换控制检查了返回原无 H 形式。

另一个 OR 图通过恒等式归入已排除的 AND 图：

```text
t' = ~t, K' = ~K, D' = C-D-1
t XOR (C*(t OR K)+D) = t' XOR (C*(t' AND K')+D')
```

剩余连续常量 XOR 合并后降为此前已关闭的三指令任意端点键语法。
四个公式族各有 80 个程序×64 输入的归一化控制和低 11 位已知模型控制。
最终证据索引为 `restore_shallow863_closed_audit.json`。

## 3. 继续寻找真正的五指令改写

新的目标固定输入键，不再允许免费把它放回前段：

```text
b0 = a XOR (a>>19)
P5(b0) = F(b0 XOR C1) XOR PRE_C5
C1 = 0xc761c23c
```

现有联合计算是 C1 XOR、四条 F、一次恢复，共六条。若找到五条 P5，可实际
删除一次运算。新的四指令排除不否定这种五条改写。

`search_joint_restore863.py` 检查十个具体五指令族。下式各系数均任意：

```text
G*((A*x+B) XOR (C*x+D) XOR K)+H
G*(u XOR (C*u+D))+H, u=(A*x+B) XOR K
(A*x+B) XOR (G*((C*x+D) XOR K)+H)
G*((A*x+B) XOR ((C*x+D) AND K))+H
G*(t XOR (C*(t AND K)+D))+H, t=A*x+B
(A*x+B) XOR (G*((C*x+D) AND K)+H)
G*((A*x+B) XOR (C*(x XOR K)+D))+H
G*((A*x+B) XOR (C*(x AND K)+D))+H
G*((A*(x XOR K)+B) XOR (C*(x XOR K)+D))+x
x*((A*(x XOR K)+B) XOR (C*(x XOR K)+D))+H
```

十族低 11 位全部 UNSAT，约 0.06–2.30 秒；第一族再把第二臂放宽到任意二次
多项式，仍在约 12.39 秒 UNSAT。这不是全部五指令程序的枚举。

掩码族显式保留旁路，未使用“某个 AND 丢信息就使整个图不可逆”的错误推论。
必要的奇偶限制来自逐位可逆性；全零/全一掩码退化到已关闭的四条或更短结构。
归一化也不改变固定输入，1,000 个随机程序×64 输入检查了保持性；
11 个正向求解控制各穷尽 2,048 输入，40 项完整 32 位检查核对公式到宏连线的翻译。

`scout_joint_wirings863.py` 另做明确标记为抽样的连线搜索：首条 MAD 的
174 个顺序各抽 12 图，共 2,088 图；首条 XOR 且保留原输入旁路的 67 个顺序
各抽 64 图，共 4,288 图。MAD 可使用两个运行时乘数，依赖深度不超过四。
首批分别为 2,081/4,280 个 UNSAT 和 7/8 个 UNKNOWN；对后 15 项延长至
30 秒求解，全部 UNSAT。原始超时与延长求解分别保存，不能把这 6,376 图当成
全部五指令空间。

## 4. 实际编译器接入与评分边界

`build_restore_middle863.py --joint-control` 用已知六条公式做完整正确性控制。
I=C1 时，原 C1 XOR 变成 XOR 零，工具现在检查其唯一消费者，实际删除指令，
并把消费者改读原输入。第 8、9、15 轮共有 96 处删除。

三组完整 uint32 树/输入经未修改的冻结模拟器回放通过，scratch 1,151 字。
对照图 C=ALU+8*VALU 从 51,665 变为 51,705：多出的 40 是实际均匀常量构造，
不是免费常量。串行回放长度 12,080 **不是竞赛成绩**，六条控制也没有提速。

未来真实五条候选仍须核算新常量、scratch、LOAD 截止期与调度。原中段深度三，
更深公式可能拖延第 15 轮关键 LOAD；工具支持仅替换第 8、9 轮来分别定价。
本轮官方测试 9 项通过，真实成绩仍为 864；保护文件的 SHA256 未变。
另有 22 项恢复/多项式研究回归测试通过。

通过 GitHub 连接核对，Claude 分支仍与
[f816608](https://github.com/txdywy/aopt/commit/f81660838f0a3d6a10dfd2ff812a87cd10f5c64d)
相同，没有本轮尚未阅读的新提交。

## 5. 复现与证据

研究依赖 Python 3.12、Z3 5.1.0、NumPy 2.3.5；CDCL 补查使用 PySAT 1.9.dev17
中的 Kissat 4.0.4。提交内核不依赖这些研究包。

```bash
python -m codex.audit_restore_shallow863 --last-families --final-pair \
  --output /tmp/restore-closed-audit.json
python -m codex.search_restore_final_pair863 --family sum --seconds 120 \
  --output /tmp/restore-sum.json
python -m codex.search_joint_restore863 --family input_mixed --seconds 90 \
  --output /tmp/joint-input-mixed.json
python -m codex.scout_joint_wirings863 --first '^' --per-order 64 \
  --max-depth 4 --seconds 0.75 --workers 6 --output /tmp/joint-bypass.json
python -m codex.build_restore_middle863 --joint-control --seeds 3 \
  --output /tmp/joint-ir-control.json
```

运行环境、指纹与最终抽样汇总另存于 `restore_closure863_validation.json`。
下一步仍需寻找不同五指令拓扑、更多 live 字或更大范围的改写；当前证据不能
承诺这些方向一定产生 sub-864。
