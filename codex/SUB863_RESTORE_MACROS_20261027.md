# 吸收输出恢复：四指令连线搜索与真实内核接入（2026-10-27）

**可执行最好成绩仍为 864 cycles。本轮没有得到 sub-864 程序。**

本轮扩大此前只检查几个固定图的输出恢复搜索，允许任意输入 XOR 编码、
不同连接和非线性 MAD 操作数。同时建立了实际编译器接入与冻结模拟器回放。
所有未解项保留，未把超时、低位域程序或串行回放当作比赛成绩。

## 1. 目标及实际收益条件

现有中段为四条指令：

```text
F(x) = 9*((33*x+0xe9f8cc1d) XOR (16896*x+0xaccf6200))+0xfd7046c5
```

寻找四条程序 P 与任意常数 I，使

```text
P(x) = F(x XOR I) XOR 0xb55afa53
```

前段既有 C1 XOR 改用 `C1 XOR I`，无需新增该处指令。令 L(e)=e XOR(e>>16)，
`L(e XOR 0xb55afa53)=L(e) XOR 0xb55a4f09`，于是第 8、9、15 轮不再需要单独
恢复 C5。三轮共有 96 个向量，每个省 8 lane，名义收益 768 lane；新增常量的
构造和布局成本必须从中扣除。

这还不是可调度性的保证：原四条中段依赖深度为三，新图可能串行四层，影响
关键 LOAD 的截止期。接入工具支持只替换第 8、9 轮，以独立核算最后一轮的影响。

## 2. 完整枚举的范围与第一轮结果

`search_restore_wirings863.py` 枚举四条 `{MAD,XOR,AND,OR}` 指令的连接。
每条 MAD 最多两个任意私有常量，两个乘数都可为运行时值；位运算也可读取
任意已有值。只有一个 live 输入 x，**不包含其他较早 live 值、移位、除法或访存**。

- 首条 MAD 归为任意仿射、x²+C、x²+x 三类，其余等价形式都包含在这三类中。
- 末条 XOR 常数退化为已排除的三指令任意输出键问题；末条 AND/OR 常数若非
  中性掩码，则目标的置换性已不可能满足。
- 死指令或常量首条同样退化为已排除的 ≤3 条任意输入/输出键问题。
- 四条纯 MAD 已被整数多项式证书排除；四条纯位运算不能产生 F 的跨位依赖。

| MAD 数 | 完整枚举连线 | 字宽阶梯直接 UNSAT | 原始 UNKNOWN |
|---|---:|---:|---:|
| 1 | 25,542 | 25,542 | 0 |
| 2 | 90,792 | 90,659 | 133 |
| 3 | 132,378 | 131,657 | 721 |
| 合计 | 248,712 | 247,858 | 854 |

原始结果为 `restore_wiring863_all{1,2,3}.json`。这些是规范连接数量，不能解释
为互不等价函数数量。由于只含无移位操作，低位投影成立，必要样本上的 UNSAT
足以排除相应完整 uint32 程序；低位 SAT 则必须先穷尽该低位域，再向更高字宽提升。
本轮没有得到目标的完整 32 位公式。

## 3. 结构归约、补查与多项式放宽

`audit_restore_wirings863.py` 对原始 854 个 UNKNOWN 逐项应用证据：

| 证据 | 新关闭连线 |
|---|---:|
| 位局部前缀后接纯多项式，所有端点键的无进位矩形筛选 | 301 |
| 常量 AND/OR 构成真正单字信息割点，只能为中性掩码 | 27 |
| 首条 XOR 形成割点、可吸收到任意输入键 | 24 |
| 多项式与原输入 XOR 旁路的矩形证书 | 1 |
| 更密样本、8→10→11 位阶梯的显式连接补查 | 236 |
| 下述独立多项式放宽 | 65 |

这一步之后剩 **200** 个 UNKNOWN。**割点检查显式保留原输入旁路**，没有使用
“首条不可逆即整个程序不可逆”的错误论证。

三条 MAD 后接末 XOR 时，输出必为两个整数多项式的异或。即使首条允许平方，
也包含于独立的 `P8(x) XOR Q4(x)`，下标为次数上限。`relax_restore_polynomial863.py`
将全部系数放开、输入键仍任意，低 11 位的必要样本约束 UNSAT（约 28 秒）。
更窄的 P4 XOR Q2 在约 1.6 秒即 UNSAT。前者一次关闭剩余 65 条实际连线。

系数采用下降阶乘基底，按 j! 的 2-adic 赋值消除函数等价系数，未把模 2^w
当作域。四个显式正向模型、三个残差插值正向模型均 SAT，并穷尽 2048 个输入。
另一个完全不固定系数的 P4 XOR Q2 正向对照自动得解并穷尽验证。

扩大为多项式与位运算结果的乘积/旁路后仍有 UNKNOWN；60 秒和残差插值后的
90 秒超时都保留在 `restore_wiring863_polynomial_*.json`，没有当作证明。

## 4. 使用 Claude 新研究中的独立 SAT 引擎

本轮重新 fetch 并阅读了 Claude 分支至
[f816608](https://github.com/txdywy/aopt/commit/f816608) 的新研究，包括选择车道引理、
仿射编码限制、374 配置扫描和尾部记账修正。它们仍未提供 sub-864 可执行程序。
其中尾部“109”是死车道被排到结束之后的假象；允许输出地址晚算时的一般界
应区分于当前地址早已被输入预读使用的具体图。

借鉴其 `cnf_cegis863.py` 的方法，新增 `sat_restore_wirings863.py`：将命名
Boolean 常量位的必要约束 bit-blast 为 CNF，再用 CaDiCaL 1.9.5 / Kissat 4.0.4
求解。每个 SAT 模型都独立运行全部 2048 个输入；反例回填后重建 CNF。
每例在独立子进程内设定时限，超时保留为 UNKNOWN。

两个引擎均通过固定原三乘数、求解加数与输入键的正向对照（全域低位验证），并与 Z3 在
一个已知 UNSAT 结构上交叉一致。最终新增排除数及剩余结构以
`restore_wiring863_complete_audit.json` 为准，所有记录保留来源哈希。
本次 CaDiCaL 对全部 200 项补查，新增排除 **24** 项；Kissat 的前 16 项补查中
两项 UNSAT 均与 CaDiCaL 重合，没有额外减少未决数。最终仍有 **176** 项 UNKNOWN，
总计 248,536 项已由必要约束或结构证据排除，没有完整 32 位候选。

## 5. 已接入实际编译器，并完成整核回放

`build_restore_middle863.py` 在接入任何宏之前，先做完整 uint32 反例检查。
旧四指令中段这种无 PRE_C5 的无效候选会被拒绝。内置五指令等价宏仅作为
控制实验，将真实 864 编译器中的 96 处中段替换，并实际删除外侧 C5 恢复。

- 3 组完整 32 位随机树和输入均通过未经修改的冻结模拟器。
- scratch 1,151 字，低于 1,536。
- 对照图算术为 **51,705 lane**，比原图 51,665 多 40，故它不是提速。
- 原图的 33/16896 等系数采用交替车道打包。通用宏需要额外五个均匀向量，
  这是实际初始化成本，未按“已有常量”免费计算。
- 不再使用的 C5 暂存包（8 STORE+1 reload）也已按协议清除。
- 12,080 是该控制实验的串行回放长度，**不是优化后的比赛周期数**。

报告为 `restore_wiring863_ir_control.json`。提交内核与 864/865 副本均未修改。

## 6. 复现入口

依赖 Python 3.12、NumPy、Z3；独立 SAT 补查另需 `python-sat`。本轮版本记录在
各结果文件中。运行时提交不依赖这些研究包。

```bash
python -m codex.search_restore_wirings863 --mads 2 --seconds 1 --workers 3 \
  --skip-controls --output /tmp/restore-all2.json
python -m codex.search_restore_wirings863 --mads 3 --seconds 1 --workers 3 \
  --skip-controls --output /tmp/restore-all3.json
python -m codex.relax_restore_polynomial863 --degrees 8,4 --seconds 60 \
  --output /tmp/restore-polynomial84.json
python -m codex.build_restore_middle863 --seeds 3 \
  --output /tmp/restore-ir-control.json
python -m codex.sat_restore_wirings863 --input codex/restore_wiring863_final_audit.json \
  --engine cadical195 --seconds 12 --workers 6 --output /tmp/restore-cadical.json
```

本轮仍留下明确未决的连接。即使最终关闭这一单输入、无移位的四指令语法，
也不能据此推断任意跨阶段或跨轮算法都无法突破 864。
