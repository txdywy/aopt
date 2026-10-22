# 863 结构研究：当前验证最好成绩仍为 864

本轮没有得到 863 可执行程序。`perf_takehome.py`、864 独立副本和
865 备份均未改动。已验证 864 的源码、日历和既有研究已在提交
`4d1e7c5` 推送到 GitHub；本文件记录随后开展的结构研究。

## 当前图的 864 下界不依赖指定启动日历

`stage_constant4_864_unconstrained_bound.json` 使用导出的完整指令图，
允许原有标量/向量模式任意选择，没有固定 canonical startup：

- 算术工作量：51,665 lane。
- 第一个周期最多执行 8 lane，至少空闲 52；求解状态 OPTIMAL。
- 最后三个周期最多执行 64 lane，至少空闲 116；求解状态 OPTIMAL。
- 因而周期数至少 `ceil((51665 + 52 + 116) / 60) = 864`。
- LOAD 和 FLOW 的独立必要下界分别为 863 和 861。

这只证明**现有指令图在这些模式下无法排进 863**，不证明所有算法的
全局最优。在首尾界仍为 52/116 的新图中，863 要求算术量不超过
51,612，即至少削减 53 lane。继续增加相同图的调度随机种子不能解决它。

重现命令（模型由 864 结果说明中的构图/export 命令生成）：

```bash
PYTHONPATH=/tmp/aopt-solver:. python3.12 -m codex.packet_budget_bound \
  /tmp/reproduce864-modes.pkl --prefix 1 --suffix 3 \
  --seconds 10 --workers 2 --output /tmp/864-unconstrained-bound.json
```

## 32 组完整构图及必要条件检查

新增 `scan_stage863.py`，每组重新构图、展开可选标量/向量模式、重建
依赖，再检查前六周期、后三周期和访存/控制流下界。结果逐组落盘，
支持 `--resume`，没有用指令数量冒充执行成绩。

| 实验 | 组数 | 周期下界分布 | 结论 |
|---|---:|---|---|
| 小常量暂存 × 哈希常量暂存 | 24 | 864：13；865：9；866：2 | 全部排除 863 |
| 根节点标量异或 × FLOW/LOAD 交换 × 单臂转置 | 8 | 865：8 | 全部排除 863 |

暂存最多常量的配置将算术量降至 51,631，但前六周期的必要空闲增加到
84，LOAD 下界升至 866。根节点标量化的基本配置省下 7 lane，却使
前六周期的必要空闲由 52 增至 116，总下界为 865。

原始结果：`stage863_staging_scan.json`、`stage863_root_scan.json`。
这些下界没有固定某一个启动日历；限时未证优时只使用求解器提供的
保守界，不把未找到解等同于不可行。

```bash
PYTHONPATH=/tmp/aopt-solver:. python3.12 -m codex.scan_stage863 \
  --group staging --output /tmp/staging.json
PYTHONPATH=/tmp/aopt-solver:. python3.12 -m codex.scan_stage863 \
  --group root --output /tmp/root.json
```

根节点改写要求常量 3 保留原生定义，因此 root 扫描的 small-stage
只选择 `4` 和 `4,-2`；不将不兼容的 `3,4,...` 配置当成实验结果。

## 深层合并记录：纠正一个看似可行的成本估计

考察了让第 7 层记录同时携带第 9 层四个候选值的布局，利用 STORE
转置取代第 9 层逐项读取。初算似乎能连带省掉 256 lane 的地址计算。
但逐项追踪 SSA 后发现：**全部 32 个第 9 层地址向量仍是第 10 层
multiply_add 的输入**。取消第 9 层 gather 不会使这些地址变成死代码。
新的节点选择还需要额外的 FLOW 或 LOAD，因此不能据此声称节省了
256 lane，也没有把这个未成立的方案写入提交版。

新增 `audit_record_dependencies863.py` 对每组检查该依赖，输出保存在
`stage863_record_dependency_audit.json`。此结论只否定“删除第 9 层读取
就能删除地址”的估计；改变第 10 层布局仍可能产生新的方案。

```bash
python3.12 -m codex.audit_record_dependencies863 \
  --ir /tmp/reproduce864.ir.pkl --output /tmp/record-dependencies.json
```

其他尚未实现的布局也须把原始数据预加载指针、重排复制、记录写入、
候选值选择和下游地址全部计入成本。尤其不能将任意四个原始字拼接
为八字记录而忽略完整 vload/vstore 的 scratch 连续性。

## 公开资料复查的限制

本轮检索了 `performance takehome 863`、`perf_takehome.py 863` 等，
未找到可核验的 863 实现。成功打开了已有的
[Josu 研究日志 GitHub 页面](https://github.com/josusanmartin/josusanmartin.github.io/blob/master/vliw/chat.log)，
但页面抓取没有返回日志正文，raw URL 抓取失败。本轮未获得新的排行榜
证据；已有本地日志只能作为之前保存的研究资料，不据此声称当前全网
最佳仍为 864。

## 后续实验门槛

优先构造能够同时通过算术、LOAD、FLOW 和边界预算的新图，然后再
投入完整排程和冻结模拟器验证。现有下界与上述 32 个负面结果均不是
新程序的实测分数，也不意味着 863 在其他算法下不可能。
