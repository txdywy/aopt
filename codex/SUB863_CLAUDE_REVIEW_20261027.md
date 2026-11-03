# GitHub 最新研究复核与除法方向补查（2026-10-27）

**可执行最好成绩仍为 864 cycles，本轮没有产生 863 程序。**

本轮按用户要求读取 GitHub 最新提交并继续研究。`git fetch origin` 确认
研究分支与本地均为 `866e2ceb17931c5c53a1ede280a4f416395d7c58`；远程当前
只有 `main` 和 `agent/extreme-kernel-1023` 两个分支。
`opus/README.md` 仍是旧的 1285-cycle 方案。

## 1. 采用的新研究结论及其边界

复核了 [最新研究提交](https://github.com/txdywy/aopt/commit/866e2ceb17931c5c53a1ede280a4f416395d7c58)
及其之前的 C1 显式连线 CEGIS、资源感知首部界、值级重复普查：

- `cegis_c1_absorb863.py` 关闭四条无移位 `{MAD,XOR,AND,OR}` C1 吸收模板。
  这个结果不能扩展为任意跨阶段、带右移或跨轮程序的十条哈希下界。
- 低算术量候选的首部 LOAD/FLOW/STORE 竞争确实需要计入。旧的弱首部界
  不能用来认定 C=51,612 的候选足够支持 863。
- `SUB863_SHADOW_PRICES_20261027.md` 的交换价格对已列出的原语菜单有用；
  它不排除新原语、新布局或更大的联合改写。

公开 [Josu VLIW 研究目录的提交](https://github.com/josusanmartin/josusanmartin.github.io/commit/20195bbf6e299197ff98e31a6640725c0fba28e1)
仍停在 10 月 4 日。本次读取未找到新的 sub-864 实现；这不是当前排行榜的断言。

## 2. 找到并修补一处证明缺口

`twoop32_closure863.py` 的操作码列表漏了冻结 ISA 中的 `cdiv`。其结果文件
还将 346 个除法/取模开头的超时，用“第一条不可逆、目标是双射”直接关闭。
这不足以构成证明，因为最后一条仍可读取原输入。

反例是两条真实 ISA 指令：

```text
t = x // 65536
y = x XOR t
```

第一条将 x=0 和 x=1 都映为 0，整体却是 `x XOR (x>>16)`，为 uint32
上的对合。新脚本自动恢复此程序，并用全域反例查询证明等价；另在冻结
模拟器上实际执行边界输入，未修改模拟器。

保留历史 JSON 的原始 UNKNOWN；给原脚本添加了指向本次审计的说明，
并修正 `SUB863_LOWER_BOUND_ARGUMENT_20261027.md` 的结论范围。

## 3. 实际继续搜索：152 个覆盖查询

新增 `audit_division_frontier863.py`，目标为：

```text
a XOR (a>>19) XOR 0xc761c23c
e XOR (e>>16) XOR n
```

每条指令有两个任意 uint32 私有常量，保留全部原输入旁路，允许任意操作数
选择。操作集合为冻结 ISA 的 13 个 ALU 操作及 MAD，共 14 种。新增覆盖为：

1. 第一条为 `//`、`%`、`cdiv`，第二条为任意操作。
2. 第二条为 `cdiv`，第一条为其余任意操作。

第一条的分母若是未经变换的 live 输入，在全零合法输入上会除零，故有效
程序的第一条除法只能采用非零常量分母。常量首条可以折叠到下一条的常量，
也被允许死首条的模型覆盖。第二条不施加这个限制，允许读取第一条结果
作为分母，逐样本要求所有实际执行的除法有效。

`cdiv` 必须先将两个操作数扩到 33 位，再做 `floor((a+b-1)/b)`，最后取低
32 位；提前截断加法会错误处理 `cdiv(0xffffffff,65536)=65536`。

| 目标 | 覆盖查询 | 直接 UNSAT | 原始 UNKNOWN |
|---|---:|---:|---:|
| g19C1 | 55 | 54 | 1 |
| g16n | 97 | 97 | 0 |
| 合计 | 152 | 151 | 1 |

这是有重叠的覆盖查询数，不是互斥函数数。UNSAT 在固定输入样本上已成立，
足以排除相应模板的任何全域正确实例；SAT 必须通过完整 32 位反例查询。
本轮未发现目标的 SAT 程序。原始结果见 `division_frontier863_results.json`。

### 唯一超时：`x%D → MAD` 的独立放宽证明

将任意末 MAD 放宽为 x 与 t=x%D 的一般二次多项式：

```text
P(x,t) = A*x*x + B*x*t + C*t*t + D*x + E*t + F  (mod 2**32)
```

这允许比真实 MAD 更多的自由度，也保留 x 旁路。

- 分母大于 4 时，在 x=0、1、2、4 上均有 t=x，所以 P 退化为一般单变量
  二次式。这四个输入上的目标值已不可能同时满足。
- 分母为 1、2、3、4 时，各用六个独立任意系数和显式样本求解，也全部
  UNSAT。样本、系数数量和结果均保存在 `division_frontier863_refinement.json`。

五个情况覆盖全部非零 uint32 分母，故该超时已由独立证据关闭。
结合原枚举的非除法 UNSAT，两个目标在所列**单车道、两条算术指令、
原输入集合和任意常量**语法内均已排除。没有将此提升为任意算法的 864 下界。

## 4. 验证与复现

- 冻结官方测试：9/9，全部 864 cycles。
- 既有 `codex.test_hash_transport863`：9/9。
- 新模型自动发现三个正向程序，均通过完整 uint32 等价验证，覆盖原输入
  旁路、33 位向上取整除法，以及双 live 输入操作数池。
- 864 提交与副本 SHA256 仍为
  `98d0f4ce1ecae152378e31ddae70d52f0ed4cb6ace5b027ce191d4027346a884`；
  865 备份仍为
  `58ce2efca049a18ef343ba82ceaeb72fd74f638bd7c8bbc23a14e6e796d72f44`。
- `tests/`、`problem.py` 与 `origin/main` 无差异。

```bash
PYTHONPATH=/tmp/aopt-synthesis:. /opt/homebrew/bin/python3.12 \
  -m codex.audit_division_frontier863 --seconds 5 --workers 4 \
  --output /tmp/division-scan.json
PYTHONPATH=/tmp/aopt-synthesis:. /opt/homebrew/bin/python3.12 \
  -m codex.audit_division_frontier863 --refine-only \
  --output /tmp/division-refinement.json
PYTHONPATH=/tmp/aopt-synthesis:. /opt/homebrew/bin/python3.12 \
  -m unittest codex.test_hash_transport863
/opt/homebrew/bin/python3.12 tests/submission_tests.py
```

依赖 Z3；本轮版本 5.1.0。研究脚本只生成模型与指令候选，不读取运行时
输入生成 Python 宿主结果，也不影响正式提交。
