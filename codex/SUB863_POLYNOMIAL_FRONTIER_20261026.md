# 四指令代数放宽与原始子节点记录实验

**可执行最好成绩仍为 864 cycles，本轮没有找到 863 程序。**
本轮增加了搜索排除、完整连接分类和两种实际指令图的回放证据。
这些结果属于研究进展，不证明 864 全局最优。提交内核、864/865 独立副本、
模拟器与冻结测试均未改变。

## 1. 关闭带额外 live 值的 MAD/MAD/XOR

沿用上一轮的真实 live 输入模型：

```text
a = 4097*prior + C0
available = [a, prior, a XOR (a>>19) XOR K]
target = F(a XOR (a>>19) XOR C1) XOR O
```

K、O 和各 MAD 常量任意，乘数也可以是两个运行时字。低 11 位查询限制真实
uint32 的 a 在 `[0,2048)`；此时右移 19 确为零，且 `prior=a-C0 (mod 2048)`。
因此低域 UNSAT 可以排除相应完整程序，低域 SAT 本身不是可执行方案。

最后 XOR 如果读取初始字或常量，已有矩形证书适用。唯一需要新综合搜索的
情形是最后 XOR 同时读取两条 MAD 的输出。

| 新搜索 | 查询数 | 直接 UNSAT | 原始 UNKNOWN |
|---|---:|---:|---:|
| 两种仿射首条及 28 种混合/乘积首条 | 30 | 29 | 1 |
| 对剩余 keyed 首条固定第二 MAD 操作数 | 86 | 86 | 0 |

第二 MAD 共有 126 种规范连接，其中 40 种只读取 keyed 字、首结果及常量，
归入已完成的单输入、任意端点键、三指令证明；其余 86 种全部新查。
原始超时保留，没有将 UNKNOWN 直接算作 UNSAT。

`audit_keyed_live_outer863.py` 检查查询配置、证据 SHA256、分区完整性，并枚举
**264,600 种连接方式**，覆盖别名和死代码。这是连接数量，不是不同函数数量
或新查询数量。完整证据在 `keyed_live863_outer_audit.json`。

这也关闭了单输入四指令 `XOR,MAD,MAD,XOR` 的吸收 C1 路线：首条 XOR 产生
x XOR K，后面三条属于上述模型；额外 prior 输入只会扩大模型。

## 2. 用多项式放宽替代昂贵的选择器乘积

记 Pj(x) 是任意系数、次数至多 j 的整数多项式。搜索使用下降阶乘基：

```text
c0 + c1*x + c2*x*(x-1) + c3*x*(x-1)*(x-2) + ...
```

第 j 项对所有整数 x 都被 j! 整除，因此可以消除模 2^w 下的系数别名。
通过前五个连续取值推导四次多项式时，先检查有限差分的二进制整除性，
仅对 j! 的奇数部分求模逆；没有把模 2^w 当作域来除以偶数。

下表均是低 11 位必要条件。独立的多项式系数放宽了实际指令图约束：
UNSAT 可以排除被包含的图，SAT 仍不是实际指令实现。

| 放宽表达式 | 输入键 | 输出键 | 结果 |
|---|---|---|---|
| P4 XOR Q2 | C1 | 任意 | UNSAT，0.351 秒 |
| G*(P1 XOR Q2)+H | C1 | 任意 | UNSAT，2.616 秒 |
| (P1 XOR Q2)+R4 | C1 | 任意 | UNKNOWN，15 秒；插值版 20 秒仍 UNKNOWN |
| P4 XOR Q2 | 0 | PRE_C5 | UNSAT，0.190 秒 |
| G*(P1 XOR Q2)+H | 0 | PRE_C5 | UNSAT，1.958 秒 |
| (P1 XOR Q2)+R4 | 0 | PRE_C5 | UNSAT，12.375 秒 |
| (P1 XOR Q2)+R4 | C1 | PRE_C5 | UNSAT，15.104 秒 |

`PRE_C5=0xb55afa53` 用于尝试把最终恢复常量吸收到中段。
固定输出键查询不会使用改变该键的归一化。

第一行关闭了 `MAD,MAD,MAD,XOR` 的仿射首条分区：三条 MAD 的次数依次至多
1、2、4，最后 XOR 至多有一个四次臂，另一个至多二次。平方首条的两种分区
此前已有独立排除，因而现在这个指令顺序整体关闭。

第二行只关闭 `MAD,MAD,XOR,MAD` 中最后执行常数缩放及偏移的子类。
最后 MAD 使用运行时乘数或多项式旁路的更大范围仍开放。

又利用精确恒等式 `(P XOR Q)+R = S-2*(P AND Q)`，其中 S=P+Q+R，建立
`search_head_and_basis863.py`。AND 操作数只需低 w-1 位，最高位被乘二消去。
自动低 8 位正向对照在 2.484 秒找到解，低 11 位已知见证也通过；真实目标
20 秒内仍为 UNKNOWN，没有因此扩大排除范围。

控制实验的限制全部保留：早期旁路模型的两个自动低 8 位控制均超时，不能
称为成功发现。直接模型后来通过自动低 5 位控制和固定见证的低 11 位控制；
新 AND 模型通过自动低 8 位控制。测试另核对完整 uint32 的冻结 ISA 见证、
随机系数归一化保持性，以及符号化的 AND 窄化恒等式。

## 3. 当前四指令范围：6/10 排除，4 种开放

这里只统计单输入、任意常量和输出 XOR 键，且为“三 MAD 加一 XOR”或
“两 MAD 加两 XOR”的十种顺序。M 表示 MAD，X 表示 XOR。

| 状态 | 顺序 |
|---|---|
| 已排除 | XMMM、XXMM、XMXM、XMMX、MMMX、MMXX |
| 仍开放 | MXXM、MXMX、MXMM、MMXM |

`head_polynomial863_audit.json` 组合新旧证据，保留原始超时记录。
其他操作码、额外 live 字、整个哈希及记录布局均不在这个 6/10 结论内。
这不是四指令程序的一般下界。

## 4. 原始子节点记录的实际搬运成本

`research_raw_d8_records863.py` 尝试让 d7 记录中的 256 个 d8 子节点保持原始
值，在 32 组主循环中恢复状态向量。只看 XOR 数量，删除 256 条预处理标量
XOR，再增加 32 条向量 XOR，算术 lane 数似乎不变。

实际构图指出这个估计漏了 scratch 连续性：第一条记录的两个原始子节点来自
同一个 vload 的第 **7、6** 个位置；vstore 需要第 **0、1** 个位置。前者要求
两字地址差为 -1，后者要求 +1，不能直接别名。最初的直接别名尝试被 SSA
连续布局检查拒绝，未进入正确性或周期结论。

可执行实验保留 256 条真实标量复制完成重排，再增加状态恢复；它没有释放
原来估计的 256 条强制 ALU，反而增加算术开销。

| 实际配置 | C=ALU+8*VALU | 强制 ALU | 算术必要下界 |
|---|---:|---:|---:|
| 原始记录方案 | 51,921 | 2,905 | 866 |
| 联合已有常量暂存及标量化 | 51,879 | 9,119 | 865 |

仅算术量已足够排除 863，无需再解首尾或访存界。这些下界也不代表一定能
排到 865/866。两个配置各用两组随机 uint32 树和输入执行完整冻结模拟器
回放，共 **4 次通过**，scratch 分别为 1111、1060 字。
`serial_control.verified` 中的 12115、17549 是串行验证日历长度，不是竞赛分数。

证据在 `raw_d8_records863_results.json`。这个负面结果只覆盖已实现的搬运
方式；改变整个记录方向、偏移和后继地址仍需另行实际构图。

## 5. 验证、公开资料与复现

五组研究测试共 **38 项通过，11.697 秒**，覆盖修改后的搜索 API、多项式
放宽、AND 表示、分区审计和原四指令见证。另有上述四次冻结内核回放。
提交内核未改动，没有重复先前已通过的官方测试及 270 例扩展回归。
文件校验、版本和测试入口见 `polynomial_frontier863_validation.json`。

本次公开检索没有取得可核验的低于 864 的实现。
[官方 README](https://github.com/anthropics/original_performance_takehome) 仍未给出
人类最佳成绩的具体数字；检索到的
[0xquinto 公开实现](https://github.com/0xquinto/vliw-kernel-optimization) 自述 1285
cycles，没有提供本轮可采用的突破方案。这不证明当前排行榜仍以 864 为首。

示例命令需要 Python 3.12、Z3 和 NumPy，正常安装环境无需 PYTHONPATH：

```bash
python3.12 -m codex.scan_keyed_live_outer863 --output /tmp/live-outer.json
python3.12 -m codex.scan_keyed_live_outer863 --refine /tmp/live-outer.json \
  --seconds 4 --output /tmp/live-outer-refined.json
python3.12 -m codex.audit_keyed_live_outer863 --output /tmp/live-outer-audit.json
python3.12 -m codex.search_head_polynomial863 --family polynomial_xor \
  --output /tmp/polynomial-xor.json
python3.12 -m codex.search_head_and_basis863 --output /tmp/and-basis.json
python3.12 -m codex.audit_head_polynomial863 --output /tmp/head-audit.json
python3.12 -m codex.research_raw_d8_records863 --controls --output /tmp/raw-records.json
python3.12 -m unittest codex.test_head_polynomial863 codex.test_head_and_basis863 \
  codex.test_keyed_live_nonlinear863 codex.test_keyed_live_middle863 codex.test_head_nonlinear863
```

剩余重点是四种开放顺序中的运行时乘积与旁路、更大范围的哈希重写，以及能
同时减少实际搬运和后继地址开销的新布局。当前证据不支持继续给原图增加
调度随机种子来寻找 863。
