# 吸收首轮异或：九种顺序排除，剩余十二种模板

**可执行最好成绩仍为 864 cycles，尚未获得 863 程序。** 本批工作从 2026-10-26
持续到 2026-10-27，增加了完整的参数枚举、可重放的排除证据和正例验证。
这些结果缩小了搜索范围，不证明整个内核的全局最优性。

提交内核、独立 864/865 文件、`problem.py` 和 `tests/` 均未修改。

## 1. 本批排除范围

令 `M` 表示任意两个操作数的乘加，`X` 表示异或。本批研究的目标仍是：

```text
F(x) = 9 * ((33*x + 0xe9f8cc1d) XOR (16896*x + 0xaccf6200)) + 0xfd7046c5
candidate(x) = F(x XOR 0xc761c23c) XOR O
```

所有算术按 uint32 回绕，O 任意。范围为单输入、任意常量和运行时乘法的
四指令图：三个 M 与一个 X，或两个 M 与两个 X，共十种顺序。
低 11 位无解是完整 uint32 同结构程序无解的必要证据；低位有解则不是可执行候选。

| 新排除的顺序 | 方法 |
|---|---|
| M X X M | 四种异或配对 × 13 种末条连接，52 个查询全部 UNSAT |
| M X M X | 150 个固定第三条查询，144 UNSAT、6 UNKNOWN；六个未决项由已有多项式证明和三个新查询关闭 |
| M X M M | 分离输入位的矩形筛选，加上完整的二元四次多项式模整数一致性检查 |

结合此前六种顺序的证明，当前 **9/10 排除**。剩余顺序为 `M M X M`。
完整索引及证据 SHA256 见 `head_affine_frontier863_audit.json`。
原始 UNKNOWN 文件保留，没有把超时改写成 UNSAT。

## 2. 两个异或顺序的覆盖

`audit_head_xor_pairs863.py` 枚举 18,900 个仿射首条后的连接模式。
去掉可约为三条指令或可重新排序的情形后，两次异或只需四种配对；
末条同时使用两个异或结果时，常量别名折叠后有 13 种角色，因此需要 52 个固定图查询。
独立测试使用完整 uint32 常量，检查 256 个非平凡连接到四种模板的精确变换。

`audit_head_inner_product863.py` 枚举 15,750 个 `M X M X` 连接模式。
原始六个 UNKNOWN 分为不读取中间异或的多项式、缩放异或结果、以及
`t*(t XOR K)+D` 三类。最后一类分别固定末条与 x、t、中间结果异或，三个查询均 UNSAT。
重复操作数、死指令和常量输出编码也在覆盖中。

## 3. 从四百万组仿射参数筛到可枚举规模

设首条为 `t=A*x+B`，令 `x=8*y+r`，r 从 0 到 7。
此时 t 的低三位固定，高八位为：

```text
(A*y + floor((A*r+B)/8)) mod 256
```

对 y 的矩形四角，若 x 和 t 的横、纵变化占用互不相交的位，则任意逐位
XOR/AND/OR 前缀仍保持这两个轴分离。随后的任意整数多项式，其混合差
含有两个原始步长的乘积；当该乘积被 2048 整除时，混合差必须为零。

初版只使用对所有输出异或常量都矛盾的矩形：单个余数筛出 5,504 个高位参数对，
八个余数联合后保留 219,008 个完整低 11 位 `(A,B)`。

随后保留每个参数对允许的**同一个输出常量**，跨矩形、跨余数求交，
剩余缩至 **123,008 个 `(A,B)`**。这是必要条件，不是程序候选。
扩展到三至八维立方体没有继续减少，本批如实保留这个负面结果。

结果文件使用带 SHA256 的压缩位图保存完整集合；不是抽样或摘要。
审计器从记录的矩形/立方体重放全部 65,536 个高位参数组合，并重新构造
八个余数在原始四百万参数空间中的交集。

## 4. 完整模整数线性消元

当 `u=t XOR x` 或 `u=t XOR K`，后面两条 MAD 的结果必然属于 x、u 的
总次数不超过四的整数多项式。因为 t 本身对 x 仿射，可以放宽为全部
15 个单项式独立系数的集合。

固定前缀参数后，寻找这些系数是 `Z/(2^11)` 上的线性一致性问题。
消元每步选取 2-adic valuation 最小的主元，显式检查非单位主元对应的
整除条件。目标与 O 的异或，可表示成 O 各位的线性组合，因而同一消元
覆盖全部归一化输出常量；不需要为每个 O 启动 SMT。

| 分支 | 完整枚举规模 | 结果 |
|---|---:|---|
| `u=t XOR x` | 30,752 个归一化 `(A,B)`，每个覆盖 512 个 O | 全部排除 |
| `u=t XOR K` | 7,712 个归一化 `(A,B)`，展开为 1,986,560 个 `(A,B,K)`，每个覆盖 512 个 O | 三批全部排除 |

输入分支约 21 秒；常量分支分为三批，每批约 327 秒。这些是研究工具的
宿主运行时间，**不是内核周期成绩**。

归一化的依据是：删除 A、B 的最高位只给 u 增加仿射项；对 t 或 u 取补码
仍是可逆仿射替换；A 为偶数时其低常量位独立于输入；任意多项式的取补码
和最高位平移允许 O 的最低/最高位置零。所有替换保持次数上界。

加速器为 `screen_head_bivariate863.cpp` 和 `screen_head_keyed_bivariate863.cpp`，
Python 入口负责枚举、覆盖和证据保存。C++ 仅用于研究枚举，提交解仍是原 Python 内核。
每个 UNKNOWN 型号与完整模整数排除证据分别保存。

## 5. 剩下的 M M X M

设 `t=A*x+B`，p 是第二条 MAD 的输出，z 是第三条 XOR 的输出。
153 个固定末条查询得到 **115 UNSAT、38 UNKNOWN**。
其中 15 个 UNKNOWN 已被此前 `G*(P1 XOR Q2)+H` 的多项式放宽证明覆盖；
其余 23 个常量别名合并为以下 **12 个模板**：

| z | 尚未关闭的末条表达式 |
|---|---|
| `p XOR x` | `t*z+C`，`C*t+z`，`C*z+t` |
| `p XOR t` | `C*p+z`，`C*z+p` |
| `p XOR K` | `C*x+z`，`t*z+C`，`C*t+z`，`p*z+t`，`C*z+x`，`C*z+t`，`C*z+p` |

这里 p 必须来自一条真实 MAD；“未关闭”表示仍有未决查询，不表示存在正确程序。
审计器枚举这个顺序的 75,600 个仿射首条连接模式。若第三条不读取第二条，
可交换顺序归入已经排除的 `M X M M`；死指令和恒定异或归入此前三指令排除。
因此表中保留的是本范围内确实尚未关闭的形式。

下一步应对这 12 个模板按第二条 MAD 的二次项结构继续分区，或者转向
移位、额外 live 值、整段哈希或实际布局改图。以上排除不覆盖这些方向。

## 6. 验证与复现

本批 20 项测试通过，包含 91 个自动小位宽合成正例、全部低 11 位输入上
C++ 目标与冻结 ISA 的比对、24 组矩阵 × 512 个输出常量的 Python/C++ 一致性、
非单位主元的小模数穷举、仿射归一化和矩形证据重放。
已知多项式正例在输入和常量两个分支均保留。

最终环境与文件指纹记录在 `affine_frontier863_validation.json`。
本轮没有重复运行未变化的官方提交评测；864 沿用此前验证结果，并核对内核 SHA256
仍与独立 864 文件一致。`git diff origin/main -- tests/ problem.py` 为空。

```bash
export PYTHONPATH=/tmp/aopt-synthesis:/Users/yiwei/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages:/tmp/aopt-solver:.
python3.12 -m codex.scan_head_xor_pairs863 --output /tmp/xor-pairs.json
python3.12 -m codex.scan_head_inner_product863 --output /tmp/inner-product.json
python3.12 -m codex.screen_affine_head_keys863 --joint --max-dimension 8 --output /tmp/cubes.json
python3.12 -m codex.screen_head_bivariate863 --prior /tmp/cubes.json --output /tmp/input-module.json
# shard 取 0、1、2，三个输出分别保留。
python3.12 -m codex.scan_head_keyed_module863 --prior /tmp/cubes.json --shards 3 --shard 0 --output /tmp/keyed-0.json
python3.12 -m codex.scan_head_final_product863 --output /tmp/final-product.json
python3.12 -m codex.audit_head_affine_frontier863 --output /tmp/frontier-audit.json
python3.12 -m unittest codex.test_head_xor_pairs863 codex.test_head_affine_frontier863 codex.test_head_bivariate863 codex.test_modular_polynomial
```

审计入口默认核验仓库中保存的证据；重新生成实验后，应显式核对文件配置及指纹。

本次也按公开资料继续检索，没有取得可复现的低于 864 的实现。
[Sambhav Gupta 的原始复盘](https://sambhavg.github.io/blog/performance) 和
[Stephen Cagle 的原始复盘](https://stephencagle.dev/posts-output/2026-03-09-anthropic-performance-takehome-writeup/)
提供的是更早阶段的调度、预载和哈希合并经验，没有改变本批实验方向。
这不能用于判断当前排行榜第一名或证明不存在更快方案。
