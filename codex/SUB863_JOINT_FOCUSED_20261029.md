# 联合窗口定向搜索：新增 16,020 个连接仍未找到 863

**可执行最好仍为 864 cycles，本轮没有新的优化提交代码。** 本轮继续尝试把
第 10→11 轮边界的五条指令替换为四条。两批搜索共检查 16,029 个规范样本，
排除与前轮两个四指令清单重复的 9 个后，新增 **16,020 个连接**。所有本批
样本最终均被排除；这仍然是抽样，完整四指令语法没有关闭。

## 窗口与搜索重点

自然的五条计算为：

```text
e = 9*z+C4
q = e>>16
f = e XOR q
n = f XOR N
out = 4097*n+C0
```

搜索免费输入只有 `[z,u,v,b,prior,a,a>>19,a XOR C1]`，不提供 e。关系、
可达域和固定合法根沿用前轮联合窗口研究：`raw_root=0x12345678`，
`N=0xa76e1971`，所有预备常量可以任意依赖根。全部先按完整 32 位计算，
没有截断后右移，也没有把非 T 的早期输入当作 b 的 T 函数。

四条包含唯一一次 SHR16、1–3 条 MAD，其余为 XOR/AND/OR，允许任意规范
SSA 连线。第一批换一组连接；第二批集中检查 SHR 的数据祖先包含新 MAD
的结构，同时要求所有结果存活、最低位必要条件成立。这个优先级选择不是
关于其他结构不可行的证明。

## 第一批：7,104 个样本

复用 `search_joint_live_boundary863.py`，`seed=8642929`，每个顺序抽 64 个，
共 111 个顺序、7,104 个连接。每项最初 1 秒，加入 24 个八状态切片。

| 初查结果 | 数量 |
|---|---:|
| SMT UNSAT | 2,143 |
| 最低位依赖不成立 | 4,901 |
| SHR 形成真正的输入割点，像太小 | 56 |
| 超时 | 4 |

将最后 4 个单独以 60 秒预算重查，全部 UNSAT，实际整批约 12 秒。原始超时
记录保持不变，复查证据另存；没有把超时直接算作排除。

## 第二批：44 个重点顺序、8,925 个连接

新增 `search_focused_joint_boundary863.py`。选样时避开前轮主清单及第一批已
出现的连接，固定 `seed=8642930`。每个顺序最多选择 256 个，受 200,000 次
抽样上限约束，实际共 8,925 个，不是 44×256 的完整集合。

初查每项 1 秒：7,561 个 SMT UNSAT、1,274 个移位割点排除、90 个超时。

对 90 个超时项，新增 418 个不同的合法 b，来自两类逆构造：

- 令七个排列中间值分别等于 0、1、2、65535、65536、最高位、全 1 等边界值。
- 精确反演 Y，使目标覆盖 0..31、每个单比特值、每个仅一位为零的字等。
  若 `t=(Y-C0)/4097`，则 `folded=t XOR N`，
  `e=folded XOR (folded>>16)`，再反演 e 得 b。

同时加入 b 翻转 bit16 的配对。每一个额外 b 都构造了实际合法树和 30 位
初始输入，并通过未改动的冻结参考计算。只加入真实可达样本，不添加关于
未知常量或乘数的猜测约束。

每项 15 秒、两个进程复查，**90 个全部 SMT UNSAT**，整批约 85 秒。
无候选、无剩余超时。这里的 UNSAT 只对应各个已选连线；没有枚举所有连线。

两批原始样本的最终账本为 9,798 个 SMT UNSAT、4,901 个最低位排除、
1,330 个移位割点排除，共 16,029 个。与前轮 14,208 个主样本、4,200 个
完整仿射尾部子族的清单去重后，新增 16,020 个。

## 搜索器修复与验证

第二批首次执行在旧搜索器的 `Distinct` 调用处失败。首 MAD 若只读已有字，
不读未知常量，则它在具体样本上的值是 Python 整数；Z3 不接受完全由普通
整数组成的 `Distinct`。修复将这些值明确转换为 32 位 bit vector，语义与
先前的模 4 必要条件相同。失败批次没有采用任何结果，修复后完整重跑。

四项新测试检查选样存活/去重、移位数据祖先、纯具体 MAD 的模 4 单射与
非单射分支，以及目标逆构造。逆构造还保留一个自然四条正例：N=0、
首 MAD 为 `9*z+C4`、末乘数 4097，重新证明整个 32 位等价，防止新增样本
误删真实公式。

**97 项研究测试全部通过**。提交程序、864 独立副本、865 备份及冻结问题和
测试未改动。研究测试、合法输入回放和 SMT 结果均不作为新的计分成绩；
官方计分沿用已保存的 9 项测试、864 cycles 记录。

## 数据与复现

| 文件 | 内容 |
|---|---|
| `joint_live_boundary863_expanded_20261029.json.gz` | 第一批完整原始结果 |
| `joint_live_boundary863_expanded_refinement_20261029.json` | 4 个超时的复查 |
| `joint_live_boundary863_focused_20261029.json.gz` | 第二批完整原始结果 |
| `joint_live_boundary863_focused_edge_20261029.json` | 90 个超时的边界输入复查 |
| `joint_focused_edge_legal_20261029.json.gz` | 418 个合法输入及冻结回放 |
| `joint_focused863_validation.json` | 去重、逐项状态、来源和文件摘要 |

gzip 归档使用固定时间戳，并逐字节解压往返核对。完整 32 位语义、正例、
所有原始 UNKNOWN 和后续 UNSAT 分开记录。

```bash
python3.12 -m codex.search_joint_live_boundary863 --per-order 64 --seed 8642929 \
  --seconds 1 --dense-fibres --workers 2 --output /tmp/expanded.json
python3.12 -m codex.search_joint_live_boundary863 --refine-from /tmp/expanded.json \
  --seconds 60 --dense-fibres --workers 2 --output /tmp/expanded-refine.json
python3.12 -m codex.search_focused_joint_boundary863 --per-order 256 --seed 8642930 \
  --seconds 1 --workers 2 --exclude codex/joint_live_boundary863_scout.json.gz \
  --exclude codex/joint_live_boundary863_expanded_20261029.json.gz \
  --output /tmp/focused.json
python3.12 -m codex.search_focused_joint_boundary863 --refine-from /tmp/focused.json \
  --edge-targets --seconds 15 --workers 2 --output /tmp/focused-edge.json
python3.12 -m unittest codex.test_focused_joint_boundary863
```

完整四指令连线、其他移位量、多移位、除法/比较混合、更大窗口、额外存活
数据及不同内核算法仍是后续空间。本轮没有发现可以接入编译器的更低分公式。
