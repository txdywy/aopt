# 除数区间与比较指令：继续收缩 863 的搜索范围

**可执行最好仍为 864 cycles，本轮没有得到 863 程序。** 提交内核及 864、865
独立版本没有改动。本轮补查九个真实已有字上的除法和比较，所有结论都限定
输入表示、根预备常量和指令结构，不是全内核最优证明。

目标与已有字沿用 `SUB863_EARLY_BOUNDARY_20261028.md`：

```text
V = [e,z,u,v,b,prior,a,a>>19,a XOR C1]
T(b) = 4097*(e XOR(e>>16) XOR N)+C0  (uint32)
raw_root=0x12345678, N=0xa76e1971
```

每个 b 都有已验证的合法 30 位树/初值构造。常量可任意依赖根，但不能依赖
被变化的叶子或 b。后者若也允许，是另一个搜索范围。

## 除法开头，后接两条 MAD／位运算

`audit_division_first_t863.py` 完整覆盖非零 uint32 常数除数，没有枚举四十多亿
个值，也没有依赖带符号除数的 SMT 超时。

在选定的八个合法状态上，九字低 16 位相同，目标低三位取遍 0..7。设第一条
商为 q，随后只用 MAD/XOR/AND/OR。低三位是 q 的一个 T 函数；既然输出取遍
八个值，q 也必须取遍八个值，并保持最低不同位的位置。这给出首批必要条件。

再利用全部初始 LSB 只依赖 b0、b19，而目标 LSB 在翻转 b16 时翻转。在每个
初始奇偶类内，后续运算只能把 q0 映成 `q0 XOR 常量`。对每个样本约束这个
关系，用商保持不变的**精确除数区间**筛选整个 [1, 0xffffffff]。

向上取整严格采用 `floor((n-1)/D)+1`（n>0），n=0 单独处理，没有截断进位。
取模方面，偶数 D 的余数奇偶不能传递 b16；奇数 D 的余数奇偶等于
`n0 XOR floor(n/D)0`，故可复用 floor 的必要区间。

| 范围 | 完整筛选后的除数 |
|---|---|
| 九来源 floor | 只有 e 的 D=65536 |
| 九来源 ceil | 无 |
| 九来源 REM（floor 区间加奇数条件） | 无 |

剩下的 `e//65536` 就是 `e>>16`，与后两条 T 运算一起属于上一轮已关闭的
三指令核心。初始字作除数分别有合法的零像，因此不能用于全域正确的程序；
常量被除数则没有新的数据位信息。

原始区间、细分端点、见证输入与完整 32 位奇偶事实均保存，未把筛选剩余值
当成公式解。区间法只检查必要条件；排除是可靠的，留下的值仍需后续证据。

## 加入比较指令

`audit_comparison_core863.py` 将比较结果放宽成每个输入可独立选择的 0/1。
固定右移 q 与布尔值 c 进入最后一条 MAD 时，低 16 位只有六种必要形式：

```text
A*q+c, A*c+q, q*c+B, q*c+q, q*c+c, q*q+c
```

九来源 ×33 个移位量全部不相容。XOR/OR 与布尔值组合保留 q 的 bit1 以上，
也全部不符；AND 的像最多为 {0,1}。变量量是布尔值时，八状态最多分三类；
量是初始字时均 ≥32；最后是移位时，高输出或奇输出迫使量为零。结合之前
的无比较核心，**九字、至多三条 MAD/位运算/动态双向移位/比较**的范围关闭。

`screen_division_boolean863.py` 进一步把 q 换成 floor/ceil 除法。商恒定区间
上先向量化检查低三位，再检查上述完整低 16 位形式，18 项全部关闭。它同时
覆盖比较在除法前、但商读取初始字与预备除数的顺序。布尔被除数像太小；
布尔除数必须始终为 1，才能避免除零。**REM 与比较混合仍保留为开放范围。**

## 除法结尾与保值链

`audit_final_division863.py` 使用一个与具体除数无关的限制：uint32 的 floor
或 ceil 商若 >2^31，除数只能为 1，商等于被除数。因此最后一条 DIV/CDIV 的
被除数必须已经在所有高输出样本上等于目标。上一轮的 297 个“先移位、再
MAD/位运算”的高输出证书，加上八状态分类与奇高输出对，排除两条无除法
前缀的所有相应顺序；除数可由前缀计算。

任意最后 REM 都无法返回 0xffffffff，而目标在一个明确合法输入上返回该值。
另有更一般的保值事实：REM 的结果若 >2^31，也只能等于其被除数；否则被减
掉的除数与余数之和已超过 uint32。左右移位在奇高输出上同样必须保持数据字。
因此纯 DIV/REM/CDIV/移位/比较的数据祖先链，即使量和除数动态，也不能改变
记录输入对的低两位。允许最开始有一条 MAD/位运算，结论仍成立。后续再次
混入数据的 MAD/位运算不在这个论证内。

## 两个补充范围

- `screen_division_xor_affine863.py` 对 `q=V DIV/REM/CDIV D; f=q XOR W;
  out=G*f+H` 完整检查 243 个来源组合。翻转 b31 时逆仿射输出的低 16 位
  必须恰翻转 0x8000，据此限制除数。REM 对大除数按两商恒定区间及精确线性
  同余筛选；两点单独为每个除数反推仿射参数。全部排除。
- `audit_bitmask_affine_tail863.py` 不限制 q 的生成方式，排除与九个已有字
  AND/OR 后再仿射输出的结构。由此将上一轮完整联合骨架从 XOR 扩展至
  XOR/AND/OR，合计 **12,600** 个规范连接全部排除。仍只是
  `t=MAD; q=t>>16; f=q BIT h; out=G*f+H` 这个四指令骨架。

另补查虚拟 e 在真实根上的逆仿射高半字条件，3.62 秒 UNSAT。加上此前八字
证据，`G*(h XOR q)+H, q<65536` 对九来源全部排除，完全忽略 q 的前缀。
这也适用于任意 uint32 被除数的 floor D≥65536、ceil D≥65537 或 REM
D≤65536，只要后续确实是该 XOR/仿射结构。

## 验证、外部进展和剩余方向

45 项研究回归测试通过；新增控制包括：区间与小范围除数穷举比较、超大 REM
除数、已植入的真实除法/异或/仿射程序、九来源逆变换、布尔六形式、向量化
筛选与标量穷举对照，以及冻结模拟器上的 floor/REM/ceil/比较指令。
向上取整覆盖了 `0xffffffff/2` 等进位边界。提交文件 SHA256 和冻结评测文件
未变，官方成绩沿用先前 9 项测试均为 864 的证据；本轮未重跑官方测试。

GitHub 上 Claude 又前进一次提交：
[dfff680 报告](https://github.com/txdywy/aopt/blob/dfff6802f769253b25cbd730804d5d282567896c/codex/SUB863_ROUND0_PINNED_20261028.md)。
报告中的首段除法 98 项已全部关闭，第 0 轮联合窗口还剩 40 个移位组合，
可执行成绩仍为 864。这里读取并固定了报告版本，没有独立重跑它的结果。

公开检索还找到一份独立的
[1166→864 时间线](https://jeffchen006.github.io/notes/864-cycles.html)：作者记录了
双线性布局、常量技巧与混合调度，最后用 GA 配合尾部精确求解达到 864；这是
作者的验证报告，未在本轮复跑其程序。其 HTML blob 为
`7e7eeddc87c57d6effa280132c4703429b742093`。另核对 Josu 的公开 chat.log
提交历史，最近一次文件提交仍为
[`9afc853`](https://github.com/josusanmartin/josusanmartin.github.io/commit/9afc8537a97a936bab0af2bf3746779924527df3)
（2026-10-04）。本轮未找到可复现的公开 sub-864 程序；这不等于证明不存在。

下一步保留：计算前缀后的中间除法及旁路、除法与移位或另一除法混合、REM
与比较混合、其他联合四指令拓扑、更多活跃字与状态编码。已有证据不能推断
这些方向必有解，也不能把局部排除升级为 864 的全局最优证明。

```bash
python -m codex.audit_division_first_t863 --output /tmp/first.json
python -m codex.audit_comparison_core863 --output /tmp/comparison.json
python -m codex.screen_division_boolean863 --output /tmp/div-bool.json
python -m codex.audit_final_division863 --output /tmp/final.json
python -m codex.screen_division_xor_affine863 --output /tmp/div-xor.json
python -m codex.audit_bitmask_affine_tail863 --output /tmp/masks.json
python -m codex.close_division_comparison863 --output /tmp/closed.json
python -m unittest codex.test_division_comparison863 codex.test_runtime_shift_core863 \
  codex.test_joint_live_boundary863 codex.test_early_boundary863 \
  codex.test_live_boundary863 codex.test_hash_transport863 codex.test_modular_polynomial
```
