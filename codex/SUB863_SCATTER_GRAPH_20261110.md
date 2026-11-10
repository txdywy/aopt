# 863 候选图：第二遍驻留节点 + 条件地址 STORE（2026-11-10）

**可执行最好成绩仍为 864 cycles，本轮没有得到 863 程序。** 未修改提交内核、
`problem.py` 或 `tests/`。本文记录第一张通过 863 全部必要条件的指令图，以及
它目前排不进去的原因。

## 做法

对选定的组，第 13–15 轮不再做 d3 记录 gather，也不算地址乘加（每组遍省 8 lane）：
d2/d3/d4 的编码节点字留在 scratch，用按奇偶位生成的“条件地址向量”把候选值覆盖
写入内存，再用 vload 读回。未选中的写入落到死单元。两种形式（`research_scatter863.py`）：

| 形式 | FLOW | LOAD | STORE | 说明 |
|---|---:|---:|---:|---|
| 原做法（每组遍） | 7 | 8 | 0 | 另有 1 条地址乘加 |
| chunk | 6 | 9 | 80 | 每个候选一次 8 字 vstore 写出整块 d2 子树（步长 7 暂存），再做两次小的缓冲区选择 |
| wide | 8 | 4 | 224 | 三位条件向量 + 标量 STORE |

| triple | 11 | 4 | 80 | 保留 d2 的 FLOW 选择；三位条件向量把选中的三字 d3 记录写出（步长 3） |

条件地址 STORE 本身在 880 版内核里已有二选一用法；这里新的是多个候选共用条件向量。

## 已验证的事实

配置：chunk 组 0,2,4,6,8,10,12,14；wide 组 3,9；其余参数同 864，`--sink 16`。

- 串行程序在冻结模拟器上 3 组随机输入通过（scratch 1149）。
- C = 51,606，LOAD 1688，FLOW 853，STORE 1618。
- `packet_budget_bound --target-horizon 863`：首部损失 52、尾部 116，算术周期下界 **863**；
  三路指令流窗口下界 LOAD 862、FLOW 863、STORE 813。
- 不加改造时导出的 864 模型 SHA256 与记录一致。

## 尚未解决

- `packet_mode_search` 4 个种子各 4000 次迭代只到 **877**。日历里 STORE 从第 250 周期到
  结尾全程占满，条件写入包被拖到末尾。关键路径也从 327 增到 570（所有包共用一个
  缓冲区，按源顺序串行）。
- 1088 次新增 STORE 太多。减少的方向：用 80 STORE 的三字记录形式代替 wide（需要
  步长 3 的指针组和 4 条 FLOW）；让最后完成的几组使用 chunk（它们的末次读回不占
  饱和区的 LOAD）；用重铺镜像代替 6 条重复异或。
- 之后仍需完整的精确排程。

## 追加：triple 形式与第二个配置

chunk 组 0,2,…,18（10 个）+ triple 组 3,9，`--sink 12`：串行验证通过，
C = 51,601，LOAD 1690，FLOW 853，STORE 1490。窗口下界 LOAD **864**（超 1 次，需少改一个
chunk 组或多一个 triple 组）、FLOW 863、STORE 779。启发式排程 4 个种子都停在 **874**，
关键路径 727：所有条件写入包共用暂存区和缓冲区，按源顺序串行，chunk 包内 32 次、
triple 包内 64 次 vstore 也各自串行（步长小于 8，后一元素要覆盖前一元素的尾字）。

下一步要先把这条串行链缩短（允许同周期按槽位顺序的两次写入，或给奇偶组各一套
暂存区），再上精确排程。

## 追加：同周期连写与第三个配置

`research_packet_870.graph` 现在允许两条条件写入共用一个周期（槽位按源顺序发射，后者
覆盖前者），只作用于 `scatter_store` 标签，原 864 模型哈希不变。关键路径回到 **327**。

chunk 组 0,2,…,16（9 个）+ triple 组 3,9，`--sink 12`：串行验证通过，C = 51,609，
LOAD 1689，FLOW 854，STORE 1410；窗口下界 LOAD 863、FLOW 863。启发式 4 个种子 4000 次
迭代到 **872**（同预算下原图是 867，而原图的真实最优是 864）。872 日历里第 350–650
周期 STORE 占满，同期算术空闲约 500 lane：锚点写入和条件写入争 STORE，把其他组的 d5
gather 拖住了。下一步要么再减 STORE，要么让改造组更早进入第二遍、把写入摊到第 150–300
周期，然后上精确排程。

公开榜单（paradigm.xyz，2026-11-10 抓取）第 1 名为 863（@ryan_kirkman，16 次提交），
第 2–6 名 864；未找到其做法的公开说明。

## 重现

```bash
python3 -m codex.research_scatter863 0,2,4,6,8,10,12,14 3,9 --prefix /tmp/sc10 \
  --small-stage 4 --stage-hash final --stage-uniform=-60,-40 --cheap-coeff --head-fill \
  --head-flow --alu-layout --tail-transposes 29,31 --tail-low --low-alu --bootstrap-flow 2 \
  --scalar-leaves --fold-raw-hash --sink 16 --literals 2 --serial
python3 -m codex.packet_modes export --ir /tmp/sc10.ir.pkl --prefix /tmp/sc10-modes
python3 -m codex.packet_budget_bound /tmp/sc10-modes.pkl --prefix 1 --suffix 3 --target-horizon 863
```
