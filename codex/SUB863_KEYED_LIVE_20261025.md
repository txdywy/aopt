# 任意编码与前段中间值：继续突破 864 的实验记录

后续：[56 个常量标量化与 LOAD 交换指令图](SUB863_SCALAR_UNIFORMS_20261025.md)。
后续验证最好成绩仍为 864。

**可执行最好成绩仍为 864 cycles，没有得到 863 程序。** 本轮有新的构图、
冻结 ISA 回放和搜索排除证据，属于研究进展；目标“突破 864”仍未完成。
提交内核、864/865 独立副本、模拟器和 `tests/` 均未修改。

## 1. 把前段中间值与任意编码键同时加入搜索

此前的任意键结果只有一个运行时输入。本轮允许免费读取三个已有字：

```text
a = 4097*prior + 0x7ed55d16
bK = a XOR (a>>19) XOR K
available = [a, prior, bK]
target = F(a XOR (a>>19) XOR 0xc761c23c) XOR O
F(x) = 9*((33*x+0xe9f8cc1d) XOR (16896*x+0xaccf6200))+0xfd7046c5
```

K、O 都是任意常量，MAD 允许两个运行时操作数相乘。新增
`search_keyed_live_middle863.py` 同时提供完整 uint32 语义与低位必要条件。
低位检查将真实的 32 位 a 限定在 `[0,2**w)`；w≤16 时右移 19 的值确为零，
而 prior 由模逆唯一得到。因此低域 UNSAT 可以排除对应完整程序；
低域 SAT 不表示找到可提交程序。

最初九个“两 MAD 加一位运算”通用查询均超时，原始记录保留为 UNKNOWN。
后续没有重复加长相同查询，而是按操作数结构分区。

### 已关闭的明确范围

第一条为 `t=A*s+B`，其中 s 为 a、prior 或 bK；第二条为
`q=t OP u`，OP 为 XOR/AND/OR，u 为初始 live 字或任意常量；第三条 MAD
可读取所有初始字、t、q 和两任意常量。在这个范围内，三条指令均被排除。

低 11 位中 `prior=a-C0`，所以第一条以 prior 为输入的情形可把偏移吸收到 B。
最后 MAD 使用 q 的所有操作数连接归入 50 个角色：6 个常数缩放、30 个
运行时乘积、14 个 q 作为加数的角色。常量同名情形包含在允许独立常量的模型中。

| 常量位运算分区 | 直接 UNSAT | 引用已有单输入证明 | 原始 UNKNOWN 后细化 |
|---|---:|---:|---:|
| XOR，第一输入 a/bK | 77 | 21 | 2 |
| AND，第一输入 bK | 29 | 21 | 0 |
| OR，第一输入 bK | 29 | 21 | 0 |
| 合计 | 135 | 63 | 2 |

另外，XOR 的六个运行时来源粗查询、AND/OR 的十四个已决粗查询直接 UNSAT。
AND/OR 第一输入为 a、第二操作数为任意常量的两类已在粗查询中排除。
原来的额外 prior 查询是冗余交叉检查，不重复计入上表。

63 个单输入归约令 x=bK，目标成为 `F(x XOR (K XOR C1)) XOR O`；
后两条不再读取 a/prior，可引用上一轮完整的单输入三指令排除结果。
证据逐项检查及 SHA256 索引见 `keyed_live863_frontier_audit.json`，
审计入口为 `audit_keyed_live_frontier863.py`。

### 两个超时的实际关闭方法

这两项的末条分别是 `D*a+q`、`D*prior+q`。专用模型为：

```text
((A*(a XOR I)+B) XOR K) + D*a + E = F(a XOR C1) XOR O
```

先放宽 E 为任意值：A 为偶数的分区 UNSAT，A 为奇数仍 UNKNOWN。
然后恢复实际的 E：a 旁路对应 E=0，prior 旁路对应 E=-D*C0（模 2048）。
两项奇数 A 查询分别在约 0.48/0.34 秒返回 UNSAT。没有把放宽模型的超时
改写为证明，也没有要求实际程序承担新增的常数操作。

使用的归一化均有独立函数验证：A+D 必须为奇数；I、K、O 的最低、最高位
可以清零。最高输出位通过 K 吸收，因此不破坏 E=0 或 E=-D*C0 的约束。
正向对照覆盖两种 A 奇偶、两种实际旁路及任意偏移版本。

## 2. 矩形必要条件也适用于这些 live 字

`screen_keyed_live_rectangles863.py` 寻找 a/prior 的四点矩形，要求两轴引起的
位变化彼此不重叠，且分别为 h、k 的倍数。a XOR K 的轴变化与 K 无关。
任何逐位前缀的每个输出位只依赖其中一个轴；因此后续任意整数多项式的
混合差分都必须被 h*k 整除。

低 11 位对全部 2048 个输出键检查：无旁路、最后再 XOR a、最后再 XOR prior
三种情况均无幸存键。这排除位运算前缀后接纯 MAD 的相关结构，但不排除
任意交错位运算和乘加。原始矩形、逐步淘汰数与正向对照在
`keyed_live863_rectangles.json`。测试另以标量穷举核对向量化筛选，并用随机
位运算前缀与多项式检查矩形条件。

## 3. 保留四条中段，但吸收前面的 C1 XOR

另搜索 `P(x)=F(x XOR C1) XOR O`，输入键固定为 C1、输出键任意。
若有合适恒等式，就可能删除每次哈希的一条前段 XOR；实际收益还需计入
常量初始化、状态编码和调度。

`search_head_merge863.py` 检查八类固定结构：

- 两个仿射臂、XOR、外层仿射，共四条。
- 两个仿射臂 XOR 后再 XOR 原输入，共四条。
- 仿射、XOR 原输入/常量、仿射、XOR 原输入/首结果/中间结果，共六类。

八类均在低 11 位 UNSAT，0.07–0.88 秒；八个自动正向对照均通过完整低 8 位
穷举。完整 uint32 随机程序另与冻结 ISA 交叉验证。
通用四条 `MAD,XOR,MAD,XOR` 的选择器模型仍 UNKNOWN；它允许更一般的运行时
乘法，不能用上述八类替代其全部搜索范围。

## 4. 真实指令图：将输出 vstore 拆成逐字 store

新增 `research_output_stream863.py`，修改最后 0/1/2/8/32 个组，每组使用
八次实际 STORE 和七次实际指针加法，重新清理图、展开标量/向量选项、重建
依赖并计算必要界。C 为 ALU+8*VALU。

| 逐字输出组数 | C | 后三周期必要空闲 | 必要周期下界 |
|---|---:|---:|---:|
| 0，原图 | 51,665 | 116 | 864 |
| 1 | 51,672 | 115 | 864 |
| 2 | 51,679 | 114 | 865 |
| 8 | 51,721 | 108 | 865 |
| 32 | 51,889 | 156 | 869 |

拆分少量组确实减少一点尾部空闲，但地址开销更大；全部拆分又损失向量写回
吞吐。本轮五种配置均不能达 863，不是对所有输出布局的全局证明。
1 组与 32 组各进行一次完整随机 uint32 冻结模拟器回放，均通过，scratch
使用 1108/1536 字。原始记录为 `output_stream863_bounds.json`。
其中 `serial_control.verified` 的 12097/12531 是串行验证日历长度，绝非新分数。

## 5. 验证、资料复查与未关闭范围

本轮三组研究测试 **38 项通过，7.191 秒**，包括冻结 ISA、32 位参考函数、
合法低域投影、原四条公式完整 uint32 等价证明、自动正向对照、操作数覆盖与
归一化保持函数。内核未改变，未重复运行此前已经通过的原提交测试及 270 项
扩展验证。

公开检索没有找到可核验的 863 程序。GitHub 连接器复查的
[Josu 最近提交](https://github.com/josusanmartin/josusanmartin.github.io/commit/20195bbf6e299197ff98e31a6640725c0fba28e1)
是聊天归档的暗色显示调整，未提供新内核。此结果不证明当前排行榜或全网最优。

仍需探索的是首 MAD 混合不同 live 字、两个 live 字相乘、其他指令顺序/操作码、
更大范围的哈希改写或改变记录布局。本轮的“200 分区关闭”只指上面明确给出的
仿射首条范围，绝不等于证明整个 live 输入语法或 864 全局最优。

```bash
PYTHONPATH=/tmp/aopt-synthesis:/tmp/aopt-solver:. python3.12 -m unittest \
  codex.test_keyed_live_middle863 codex.test_free_middle_keys863 codex.test_restore_frontier863
PYTHONPATH=/tmp/aopt-synthesis:. python3.12 -m codex.audit_keyed_live_frontier863 \
  --output /tmp/keyed-live-audit.json
PYTHONPATH=/tmp/aopt-synthesis:. python3.12 -m codex.scan_keyed_live_frontier863 \
  --output /tmp/keyed-live-partitions.json
PYTHONPATH=/tmp/aopt-synthesis:/tmp/aopt-solver:. python3.12 -m codex.research_output_stream863 \
  --output /tmp/output-stream.json --controls
```

受保护源码 SHA256：当前提交与 864 副本均为
`98d0f4ce1ecae152378e31ddae70d52f0ed4cb6ace5b027ce191d4027346a884`；
865 副本为 `58ce2efca049a18ef343ba82ceaeb72fd74f638bd7c8bbc23a14e6e796d72f44`。
