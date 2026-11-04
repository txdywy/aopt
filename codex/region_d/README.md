# 区域 D 工具与结果（见 `../SUB863_REGION_D_20261103.md`）

求解器（均对全部 32 位常量精确；UNSAT 即排除整个连线）

| 文件 | 作用 |
|---|---|
| `rd2.py` `rd3.py` | 需求驱动宽度模型，常量移位按带 [s_lo, s_hi]；rd3 共享单项式 |
| `rdg.py` | 通用精确模型：全部 SHR 操作数 32 位，移位量为自由 32 位字面量（一次覆盖所有 s） |
| `rdv.py` | 运行时移位量的精确模型 |
| `rda.py` | 纤维归约：移位值关于 z 仿射（三种情形，c 位） |
| `rda2.py` | rda + 按位路径 (x op k)>>s = (x>>s) op (k>>s)，含按参数拆分 |
| `rdvf.py` | 仿射运行时移位量：v2(A_b) ≤ 31−c 由 closure3 排除，其余纤维上恒定 |
| `rdq.py` | 平方路径 w²+w（导数奇，高位取遍），按 e 逐个查询 |
| `rdvq.py` | 移位量/值为 w²+x 的精确展开 |
| `rdvf2.py` | rdvf 推广：移位量为仿射根的按位组合 |
| `rdp2.py` | 一般多项式值（≤4 次）：逐类 v2(v'(z0)) 纤维模型 + 逃逸过滤，c=13 |
| `rdqk.py` | w²+x 的精确展开（按 e 拆分），任意 c 精确 |
| `rdm.py` | 多 SHR 同根：所有移位操作数由同一仿射值经移位/按位运算得到 |
| `xor_reduce.py` | 输入异或吸收：I1 = z^k 且 z 不再被读时化为 SHR 前移一位的连线 |
| `rdb.py` | 通用 g(w) 的 32 位位切片模型（慢，仅作交叉参考） |
| `closure3.py` | ≤3 条 T-op 在模 2^11 下不能实现目标 |
| `varshift3.py` | 定理 V 的逃逸查询 |

验证：`test_rda.py` `test_rda2.py` `test_rdvf.py` `test_rdvf2.py` `test_rdq.py` `test_rdvq.py` `test_rdp2.py` `test_rdp2_c13.py` `test_rdqk.py` `test_rdm.py` `test_xor_reduce.py`
（引理随机检验 + 正控例），`sound_rda.py` `sound_rda2.py`（已知 c=8 真解必须复现为 SAT）。

驱动（可续跑，按枚举下标去重）：`drive.py`（带）、`drive_h2.py`（常量移位：rdg 快路径 →
rda/rda2/rdq → 带）、`drive_v2.py`（运行时移位量：rdv 快路径 → rdvf/rdvq → rdv）、
`drive_g2.py`（k 型逃逸与多 SHR：rdg → rdvf）、`resid_*.py`（残余加预算/拆分重跑）、
`coverage2.py`（汇总，写 `coverage2.json`）。

复现：`cd results && gunzip -k *.jsonl.gz && cd .. && ./campaign5.sh & ./vchain5.sh & ./mchain.sh &`
（在仓库根目录 `/home/user/aopt` 下运行，脚本把它加入 `sys.path` 以读取 `codex/cegis_c1_absorb863.py` 的常量）。
