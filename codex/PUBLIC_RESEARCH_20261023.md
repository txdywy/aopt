# 公开资料复查与两指令搜索补全

当前本地完整验证的最佳成绩仍为 **864 周期**。本轮没有新内核，也没有
取得排行榜的新提交结果。此次工作补充参考资料，并把上一轮孤立哈希
片段的搜索范围扩展到完整的两条算术指令范围，条件如下。

## 一手资料及采用判断

| 来源 | 本次实际读到的内容 | 对当前研究的作用 |
|---|---|---|
| [Josu 原始研究日志](https://josusanmartin.com/vliw/chat.log) | 日志末段报告 864，并讨论继续减少工作量、访存与转置成本 | 与我们已有的同图下界一致；没有读到已验证的 863 |
| [HMA 论文 §4.5](https://arxiv.org/html/2610.13494v1#S4.SS5) | 单次 48 小时实验报告 878；引用 2026-10-08 排行榜首位为 864，并说明不是验证后的榜单名次 | 较新的历史分数参照，不能视为今天的实时排名 |
| [HMA 代码](https://github.com/humanfia/hma)及[实验说明](https://github.com/humanfia/hma/blob/main/docs/experiments.md) | 公开交替运行与实验复现框架 | 已查看的页面没有提供可直接替换当前提交的低于 864 实现 |
| [Helo3301 实现说明](https://github.com/Helo3301/vliw-simd-optimization) | 报告 1138；介绍 MAD 合并、标量分流与阶段错开 | 主要改写当前内核已包含；其全局下界声明不能直接用于本题当前值输出评分配置 |
| [AI-Comp](https://github.com/fiigii/ai-comp) | 可配置的多层 IR、优化 pass 和寄存器压力诊断 | 保留为编译器组织参考；本次 README 没有建立低于 864 的成绩 |
| [Souper 原论文](https://arxiv.org/html/1711.04422v2) | 数据流切片、路径条件、候选综合和独立等价验证 | 采用其明确输入域与表达式范围的方法，补全我们自己的 ISA 搜索 |
| [egg 项目](https://egraphs-good.github.io/) | 等价程序的紧凑表示与优化提取 | 后续跨片段重写的框架参考；本轮未引入依赖，也没有运行 egg |

Helo 的 README 中有“适用于任意程序”的 854/1044 周期下界说法。
我们不采用这种范围：只按 VALU 计算忽略了 ALU 的承载能力；在少数
gather/fusion 策略之间求最优也没有覆盖不同记录布局。这里也不能用
双方输出要求可能不同的成绩，代替逐条验证某个完整内核。

搜索了带 863、864、860 等分数的题目关键词，也搜索公开源码和 VLIW
实现；没有取得可核验的低于 864 源码。精确分数搜索多次返回无关论文
行号，因此后续以原始仓库、日志和论文为准。这个结果不代表全网不存在。
Paradigm 页面本次抓取失败；Chrome 预检未连接，系统也未找到该应用，
因此仅使用公开网页读取，没有账户操作或现场榜单验证。

## 新实验：补全第一条算术指令

[上一轮](SUB863_HASH_RESEARCH_20261023.md) 覆盖 406 个模板，但没有覆盖
所有第一条指令。本轮新增八类：

```text
x*x+x       D<<x       D>>x       x<<x       x>>x
x<D         D<x        x==D
```

三个目标各枚举 14 种第二条指令，新增 **336 个检查全部 UNSAT**。
`hash_extended863_*_result.json` 保存 24 个批次的原始输出；每批两项
正控制均通过。加上此前批次，共 742 个检查、106 个正控制。

去掉固定右移 16 与任意固定右移族之间的重复，每个目标由 17 个第一条
指令族乘 14 个第二条 opcode 覆盖，共 **714 个规范化模板**。此前唯一
SMT 超时项仍保留 UNKNOWN，以独立的二次多项式反例证书排除。

## 覆盖性证明及严格范围

`prove_first_form_coverage863.py` 枚举全部 14 种算术 opcode 的输入形状：
13 种二元操作各四种变量/常量形状，加上 MAD 的八种形状，共 60 种。

- 54 种形状通过带任意符号常量的全 uint32 等价检查，归入那 17 个族。
- 其余六种是除法、余数或 cdiv 的除数为 x，在 x=0 必然异常，不能作为
  对全部 uint32 输入有效的直线程序。
- `x<<D` 归入系数为 `1<<D` 的仿射族；右移按 D=0、1..31、≥32 分域。
- 第一条不含变量时，它是预先可计算的常量；第一条含三个 x 的 MAD
  正是本轮补上的 `x*x+x`。
- 最后一条若只使用常量，输出也只能是常量，不等于这三个非恒定目标。
  否则最多需要两个不同字面常量，已被第二条的 x、第一条结果、K0、K1
  操作数池覆盖。只有一条指令的程序也可加恒等第一条归入这个范围。

审计器逐项要求全部 714 个模板有排除证据，缺文件或缺 opcode 会失败；
不会将缺失、超时或未找到候选自动算作排除。结果和来源文件摘要保存在
`hash_two_instruction_scope863.json`，形状证明在
`hash_first_form_coverage863.json`。

**可以得出的结论**：在只有一个 uint32 输入、任意预先给定常量、仅使用
所列 14 种算术操作的直线程序中，transport、middle-tail、head-xor 三个
孤立目标均至少需要三条指令。它们各自已有三条实现，因此此范围内恰为三条。

**不能得出的结论**：完整哈希或整个内核是最优。FLOW select、内存查表、
动态分支、多个输入及外部已存活的中间值均不在这个证明内。尤其 middle-tail
换元后，若保留换元前的 b 作为额外输入，已经属于另一个搜索问题。
后续应优先检查跨片段共享中间结果与路径约束，而不是再次搜索同样的两条
独立算术指令。

## 验证与复现

```bash
PYTHONPATH=/tmp/aopt-synthesis:. python3.12 -m codex.search_hash_transport863 \
  --objective transport --first constant-shr --seconds 8 --output /tmp/transport-shr.json
PYTHONPATH=/tmp/aopt-synthesis:. python3.12 -m codex.prove_first_form_coverage863 \
  --output /tmp/first-coverage.json
PYTHONPATH=/tmp/aopt-synthesis:. python3.12 -m codex.prove_first_form_coverage863 \
  --results-dir codex --output /tmp/two-instruction-scope.json
PYTHONPATH=/tmp/aopt-synthesis:. python3.12 -m unittest codex.test_hash_transport863
```

本轮 9 项研究测试全部通过。测试增加了全部 18 种入口形式与冻结模拟器的对照，以及第一条形状的
符号覆盖性检查。变长左移的模拟器对照限定移位量，避免 Python 创建巨大
中间整数；SMT 搜索使用截断到 uint32 后的位向量语义。
提交程序不变，官方测试和 270 例验证沿用此前证据；本轮只重新检查研究
代码、源码摘要和冻结文件无改动。
