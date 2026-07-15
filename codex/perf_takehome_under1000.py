"""A resource-balanced kernel for Anthropic's original performance take-home.

The scored workload is deliberately fixed (height 10, 16 rounds, 256 inputs).
This implementation specializes that shape at build time and emits only legal
instructions for the frozen single-core machine.  It does not modify the
simulator, the reference implementation, or the tests.

The main ideas are:

* fully unroll the two traversals (depths 0..10 and 0..4);
* keep the hash result internally XORed with the final hash constant;
* cache tree levels 0..4 and mirror path bits to make lookup cheap;
* gather deeper nodes in place and pre-transform levels 4..7 once;
* split independent work across VALU, scalar ALU, load, store, and flow slots;
* list-schedule a dependency DAG with the machine's end-of-cycle write rules.
"""

from __future__ import annotations

import base64
from collections import defaultdict
from dataclasses import dataclass, field
import heapq
import struct
from typing import Iterable
import zlib

from problem import DebugInfo, SCRATCH_SIZE, SLOT_LIMITS, VLEN


C0 = 0x7ED55D16
C1 = 0xC761C23C
C2 = 0x165667B1
C3 = 0xD3A2646C
C4 = 0xFD7046C5
C5 = 0xB55A4F09
C23 = (C2 + C3) & 0xFFFFFFFF
M23 = 33 << 9
C2_SHIFT9 = (C2 << 9) & 0xFFFFFFFF

OFFICIAL_SHAPE = (10, 2047, 256, 16)
N_GROUPS = 32
N_WORKSPACES = 9
N_SPILL_WORKSPACES = 9
# Experimental compiler-style workspace renaming.  The normal kernel assigns
# physical scratch before instruction scheduling, which introduces false
# dependencies between unrelated groups.  In SSA mode the short-lived mux
# registers are scheduled virtually and colored back onto this same physical
# pool afterwards.
SSA_WORKSPACES = False
# Rename the one-vector level-4 mux stack (and its first-traversal p3
# condition) before scheduling.  Reusing one physical vector while building
# the DAG creates artificial WAW/WAR chains between unrelated groups and even
# between the two traversals.  The existing post-schedule linear-scan colorer
# maps these short intervals back onto the same scratch register pool.
SSA_LEVEL4_WORKSPACES = False
SSA_ALL_WORKSPACES = False
SSA_FIRST_WORKSPACE_GROUPS: frozenset[int] = frozenset()
FLOW_SCALAR_CONSTANT_COUNT = 0
# Sparse resource-aware constant materialization.  The prefix knob above is
# useful for coarse sweeps, but insertion order puts the latency-critical C0
# before several independent constants.  An explicit index set lets the
# offline compiler move only profitable loads onto FLOW immediates without
# forcing that critical constant across engines.
FLOW_SCALAR_CONSTANT_SET: frozenset[int] = frozenset()
# Selected setup constants may use their own architecturally-zero scratch word
# as the add_imm base.  This removes the otherwise artificial dependency on
# the loaded scalar one and can pull an entire broadcast fanout forward.
FLOW_ZERO_BASE_CONSTANT_SET: frozenset[int] = frozenset()
FLOW_ONE_CONSTANT = False
# Selected setup immediates may be materialized with the load engine instead
# of FLOW ``add_imm``.  This is a late resource-balancing knob: setup loads
# finish before the steady-state gathers, while removing them from FLOW gives
# the nearly saturated selector schedule more freedom.
LOAD_IMMEDIATE_TAGS: frozenset[str] = frozenset()
ALU_DERIVED_POWER_COUNT = 0
ALU_DERIVED_DEPTH_START = 11
ALU_DERIVED_MISC_SET: frozenset[str] = frozenset(
    ("nineteen", "m4", "m0", "m2", "m23")
)
WORKSPACE_ASSIGNMENT = (
    0, 2, 3, 2, 4, 0, 1, 4, 3, 0, 1, 0, 1, 6, 4, 2,
    3, 2, 3, 2, 5, 4, 3, 0, 7, 6, 5, 1, 1, 7, 5, 5,
)
SECOND_WORKSPACE_STRIDE = 3
SECOND_WORKSPACE_FIXED = 8
SECOND_WORKSPACE_COUNT = 1
SECOND_WORKSPACE_ASSIGNMENT: tuple[int, ...] | None = None
INDEPENDENT_ROOT_CACHE = True
DIRECT_MIRROR_PATH = False
OVERLAP_DEEP_ADDRESS = False
OVERLAP_SHALLOW_ADDRESS = False
TAIL_EMISSION_MODE = "full_offset"
TAIL_EMISSION_STAGGER = 1
TAIL_EMISSION_COHORTS = 8
TAIL_GROUP_ORDER = tuple(range(N_GROUPS))
FULL_ROUND_OFFSETS = (
    0, 1, 13, 17, 8, 20, 0, 2, 1, 5, 6, 14, 20, 9, 17, 12,
    22, 21, 17, 6, 13, 22, 6, 10, 9, 15, 8, 14, 10, 13, 2, 22,
)
SCATTERED_OUTPUT_STORES = False
OUTPUT_POINTER_STREAMS = 2
FLOW_OUTPUT_POINTER_ADVANCE = False
FLOW_OUTPUT_ADVANCE_POSITIONS: frozenset[int] = frozenset()
OUTPUT_GROUP_ORDER = tuple(range(N_GROUPS))
PER_GROUP_OUTPUT_POINTERS = False
INDEPENDENT_TAIL_OUTPUTS = True
INDEPENDENT_TAIL_GROUP_COUNT = 8
PRESERVED_TAIL_OUTPUT_POINTERS = False
PREPROCESS_MAX_DEPTH = 7
REUSE_CACHED_LEVEL4_PREPROCESS = False
# Experimental memory-layout retiming.  Preprocess depths 4..8 into reversed
# heap order at addresses 2**depth..2**(depth+1)-1.  A path mirror then becomes
# an address after one base add and advances as ``2 * address + parity``.
# This trades setup-only scalar work for a large steady-state address-ALU cut.
REVERSED_RELOCATED_TREE = False
# The relocation copy frees each staging vector as soon as its source block is
# consumed.  Choosing which physical vector stages which block is therefore a
# register-release scheduling decision, even though it does not change the
# memory image.  Experimental modes prioritize the per-group hash temporary
# (needed in round 0) and mirror (needed in round 1) by the steady-state launch
# order instead of freeing mirrors 0..31 first by allocation address.
RELOCATION_STAGE_ORDER = "linear"
RELOCATION_STORE_STREAMS = 1
RELOCATION_LOAD_STREAMS = 2
VECTOR_TOP_C5_BLOCKS = 0
REUSE_TOP_RELOCATION_LEVEL4 = False
INDEPENDENT_TOP_P0 = False
INDEPENDENT_TOP_P1 = False
INDEPENDENT_RELOCATION_LOAD_POINTERS = False
INDEPENDENT_INPUT_POINTERS = False
DERIVE_TOP_P1_FROM_P0 = False
DERIVE_SETUP_SECOND_POINTERS = False
EARLY_FINAL_CACHE_SET: frozenset[int] = frozenset()
EARLY_FINAL_ADDRESS_SET: frozenset[int] = frozenset()
VECTOR_EARLY_FINAL_ADDRESS_SET: frozenset[int] = frozenset()
BRANCH_FINAL_GROUP = 31
BRANCH_FINAL_LANES: tuple[int, ...] = tuple(range(VLEN))
BRANCH_DISPATCH_PADDING = False
BRANCH_DEDICATED_DEAD_REGS = False
BRANCH_DEAD_CANDIDATE_GROUP = 0
BRANCH_DEAD_CONTROL_GROUP = 1
BRANCH_DIRECT_FULL_TABLE = False
PAIRED_BRANCH_FINAL = True
PAIRED_EARLY_XOR = False
PAIRED_FLOW_SELECT = True
DELAYED_PAIR_BRANCH_GROUPS: frozenset[int] = frozenset()
DELAYED_PAIR_TARGET_SENTINEL = 0xD4A00003
DELAYED_PAIR_TABLE_STRIDE = 1104
SAVED_SECOND_PATH_EXTRA_GROUPS: frozenset[int] = frozenset(
    (0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 16, 17, 18, 20, 21, 23, 25, 27, 28, 29)
)
PIPELINED_DEPTH3_GROUPS: frozenset[int] = frozenset()
PIPELINED_DEPTH3_WORKSPACE_OVERRIDES: dict[int, int] = {}
PAIRED_TARGET_SENTINEL = 0xB2A00003
DIRECT_BRANCH_LOOKUPS: dict[int, tuple[int, ...]] = {}
CHAINED_DIRECT_BRANCH_BASE = True
PAIRED_DIRECT_BRANCH_LOOKUPS: dict[int, tuple[tuple[int, int], ...]] = {}
FUSED_DIRECT_BRANCH_XOR = True
DIRECT_PREP_SENTINEL = 0xD1A00001
DIRECT_JUMP_SENTINEL = 0xD1A00002
DIRECT_TARGET_SENTINEL = 0xD1A00003
DIRECT_BRANCH_PRIORITY = 10_000
MADD_FIRST_DEPTH1 = False
MADD_FIRST_DEPTH1_SET: frozenset[int] = frozenset()
SCALAR_FIRST_DEPTH1_SET: frozenset[int] = frozenset()
SCALAR_SECOND_PATH_GROUPS: frozenset[int] = frozenset()
SCALAR_SECOND_PATH_DEPTH2_GROUPS: frozenset[int] = frozenset()
SCALAR_SECOND_PATH_DEPTH3_GROUPS: frozenset[int] = frozenset()
SCALAR_LEVEL4_CONDITION_GROUPS: frozenset[int] = frozenset((0,))
SCALAR_FINAL_C5_SET: frozenset[int] = frozenset((18, 20, 22, 24, 25, 30))
SCALAR_FINAL_JOIN_SET: frozenset[int] = frozenset((21, 22))
SCALAR_FINAL_HASH4_SET: frozenset[int] = frozenset()
SCALAR_FINAL_SHIFT_SET: frozenset[int] = frozenset((17, 18, 20, 22, 23, 26, 29))
SCALAR_FINAL_HASH23_JOIN_SET: frozenset[int] = frozenset((0, 1, 17, 31))
SCALAR_HASH1_JOIN_SET: frozenset[tuple[int, int]] = frozenset(((18, 15),))
SCALAR_HASH23_JOIN_SET: frozenset[tuple[int, int]] = frozenset()
SCALAR_HASH5_JOIN_SET: frozenset[tuple[int, int]] = frozenset()
# A parity extraction is semantically either eight independent scalar ANDs
# or one vector AND.  The scalar form has better aggregate throughput on the
# 12-wide ALU, while the vector form completes every lane atomically and can
# shorten a following whole-vector mux/address dependency.  Keep the default
# empty in generic builds; this scored specialization opts in the seven
# whole-vector parity cuts selected by exact resource-aware scheduling.
VECTOR_PARITY_SET: frozenset[tuple[int, int]] = frozenset(
    ((31, 11), (31, 12), (31, 13), (31, 14), (16, 14), (21, 13), (21, 14))
)
FINAL_CACHE_SET = frozenset((0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 17, 23, 28, 29))
# Final cached depth-4 lookups normally spend seven scarce FLOW slots on
# vselect.  Selected groups instead use a short-lived, post-schedule-colored
# vector scratch and lower each select to ``diff`` + ``multiply_add``.  This
# is deliberately opt-in: it trades 14 VALU instructions for seven FLOW
# instructions and is useful when a cached group is added to relieve LOAD.
VALU_FINAL_CACHE_SET: frozenset[int] = frozenset()
VALU_FINAL_CACHE_COUNTS: dict[int, int] = {}
SCALAR_VALU_FINAL_DIFF_SET: frozenset[tuple[int, int]] = frozenset()
FIRST_CACHE_SET = frozenset(
    (3, 4, 5, 7, 9, 11, 12, 13, 15, 16, 18, 19, 20, 24, 29, 31)
)
HASH_SCALAR_MOD = 4
HASH_SCALAR_STAGE = -1  # scalarize the independent stage-1 constant XOR
_BASE_SCALAR = {(group, (1 - group) % 4) for group in range(21)}
_SCALAR_CANDIDATES = [
    (group, rnd)
    for group in range(N_GROUPS)
    for rnd in range(16)
    if (group + rnd) % 4 and (group, rnd) not in _BASE_SCALAR
]
_SCALAR_CANDIDATES.sort(
    key=lambda pair: (pair[1], FULL_ROUND_OFFSETS[pair[0]]), reverse=True
)
_SELECTED_SCALAR_EXTRA = {
    (16, 15), (31, 15), (12, 15), (3, 15), (18, 15), (11, 15),
    (27, 15), (2, 15), (14, 15), (20, 15), (15, 15), (23, 15),
    (28, 15), (24, 15), (4, 15), (26, 15), (10, 15), (19, 15),
    (22, 15), (7, 15), (30, 15), (8, 15), (0, 15), (6, 15),
    (16, 14), (21, 14), (31, 14), (17, 14), (5, 14), (12, 14),
    (3, 14), (25, 14), (11, 14), (27, 14), (20, 14), (29, 14),
    (15, 14), (23, 14), (28, 14), (13, 14), (24, 14), (4, 14),
    (19, 14), (9, 14), (7, 14), (1, 14), (8, 14), (0, 14),
    (16, 13), (21, 13), (17, 13), (5, 13),
}
HASH_SCALAR_EXTRA = frozenset(
    (_BASE_SCALAR | set(_SCALAR_CANDIDATES[:43]))
    - {(27, 15), (28, 15), (29, 15), (30, 15), (31, 15)}
    - {
        (12, 15), (19, 15), (26, 15), (11, 15),
        (17, 14), (8, 15), (12, 14), (10, 15), (7, 15), (6, 15),
        (4, 15), (23, 14), (20, 14), (19, 14), (15, 14), (13, 14),
        (0, 15),
    }
)
HYBRID_MADD_PAIRS = 8
HYBRID_MADD_OVERRIDES: dict[tuple[int, int], int] = {}
SCHEDULE_POLICIES = (4,)
BACKWARD_POLICIES = ()
SCHEDULE_NOISE_SEED: int | None = None
SCHEDULE_NOISE_AMPLITUDE = 1
# Optional offline compiler oracle: a feasible CP-SAT schedule can be distilled
# into list-scheduler priorities without embedding the schedule itself.  Search
# tools set these arrays before construction; production defaults remain purely
# deterministic and self-contained.
SCHEDULE_EXTERNAL_SCORES: list[int] | None = None
SCHEDULE_EXTERNAL_TIE_SCORES: list[int] | None = None
SCHEDULE_EXTERNAL_HEIGHT_WEIGHT = 2
_EXACT_SCHEDULE_990_B85 = (
    "c-noO3EWTB_W#d0n+(mFLJ<w5QmH74G>{}|Qi+P2iZse?j*28hl!OqOB148kgfc~u6ro8PZs|7Ymg_eC*SPoETfe{e>-#+0^PIEy+G~B*+WVaEdA?azWL4pn"
    "WqIb{RU+F9*e}~ZJ0L5UmCq_<Rp93(=ywHt#db+L_H=laq@R|6*Iv*r1>kQ>_#^RE3Vu>DD=n1E%6Q5GdlUNrIA%Gpr5aEjr~#BG3o4K`2Lh#`w=!}T{%mh}"
    "?UU^b=N_0H1ZVH(kFA(h$_|F_2h-2XXO&?dRpEP8I0AonNOmZUQz<(Pj=<m5$ZBSX!}r7C2>jg<*^${%@ck&V=CG_*c63$;zSof>kI9bBj)U*V$&q!l<Fgas"
    "`w4JlZ8-MCtX_5!d_M`!z++F&PRZ)S_xf-K9@`*0H9HNyp9W{(u?@3E;L&>UI*E>LlAW0~C0i?!RW;ysI7J72S&}?amR_a73#GveWxx;kDhu;3OMWR2uZr-K"
    "O7z+f`Z)k*cOc9PUkAa=4g%}Tf&CTWRS~SO2-a2td+~KJSc|X9U~yHj=Mb=HuY^)Qt^tiJ$7TP7vI!h}JaRvY9L9?&FL3NF;h1vf=n4{Bj%ZAo1XZ2|?bz4Y"
    "5yP>t;7B@FRZ&#+grM(z%uHi&Je@hu!cx`u9fPVojMRvFWEIv*2tF(?xrA1;hbkMZtNE;}Zu&hu0mn$;F>6zHpw44-evEpxoUw&1sVw%OMQowP9#h#^t{y*("
    "YK7&{gldpcn12l<?>O6HYMWVbj>ce~a#o)t*^GU~oCdaNEMjW;{?ido@Ej+j)bk_6G>ArGa3(mHGqT21Gh&*;%nwUo#l)b^svxv7rY8NIy=OV~|6@T@&p8R_"
    "X3c=+*?B;V?EI`{c0tw(xX{orp^?Pv3>u|z)NUfRGUcr7Y=Fn((JU>%&sGcEmj!!0OE_D=-n2ioooOjt;E%WGo#W4H7PXs8FD;~p3&;npvx^eifMVF1vlF<l"
    "n9whKHim6F$IKxnSfaMH$u0&iF<c5i$FG-VmnXE%t{|>7T$Q!UuFl$L9kOe(YqRST&QEA5Hme?NyS3QUM(kk=bj-k&bcUt1c(skp#<sO5?yD`G&o;-jr=u-="
    "rk;Cw)YhK3FYAGr>k?1%nU?>n7xZ37pcC*98rRZR{KDs66}7de6zrK})RFp(xn9=Yuo70Gl_};BskU5`>n3!Rp0Ce3XE$UwW;Y34gs$1mK)37`;MVN6?Dp)A"
    "?9PPlGzUvNi5AbHL)5;;t4=+oW5?V?M_aB*{Jf{nua~}fCThK<8y#)wB%bASJ4fvsBA?u3ywjEX?j~M|xt-2W>70Q5be5jFc<XF`w+nZOm+u04WdF?W&hE+X"
    "1@1H4@B6*MaHCj{k;Haa1CGYyg}8Q4xC`n}%Y;ygG($M&Ec+(#jwD`bPf0k;>>1+q?<hEv&pIQ4cYA@OeeZSsx*b**=B|W)l1I^}>{Z)l&prS=m^}pa%z9-H"
    "XOCo$CiM3C67ohfhjYia)#Em@J;VyjJ#@6?R#69^dxud-su5KQ+8^@(9c{T&BG2dch}zoIJ;oakk|$z%k(Vrwh@W}`eE{2j4CtFZPV_T8k@W`#WCOE7*^}Ae"
    "gdt>+<sq@87sch_sLkW^h&rB*U>QKQ`3h5smAJNZolLW<CFOzTB3X$r4|pxS%2zMtd|E|yXBAdai(Lb+FqUa8y}+-TZDbd4IsB%Bu@!4hw}d_kPXSK@LxE=i"
    "+a3lyn+*p>WFvv+vQgRSY)tli!dS``%YC9d&YTCMHpa}hdrDTNsCxPstpxhf`4*0GU#}_KeuCmYU=L7dth6EG2g^`%PlT@&v9`q2m8%84Lbx?Sm8d-7HIAbX"
    "jM`|6Z9gfN4e>F0#z(|5f}&{QXkolCVxyz>7>eLnU|jYBFg}}*P0S``lM`MfJ1ndRRm6y>t@;>cMi@uE$4r3LD6BSGt1Mh?EVc%tR@9fFqI3eyhcoZ<`eTK~"
    "XocYF5GoJHrK4ngrv%KVc=i0dT~S;`x%%R0m}z|lROFxpu60lO2*(^M>mjd*N08!9hHF2*QsV0*tmC?py$Fm;m<qfEOarC^wmk!wnZ3LRv$EO1oNR9PO7?0t"
    "FJV6U%n~%qs<mxYrERO8pYfU+;T2>VO~+WqNCjdAkBi!<XWJex>Q;Tfm`x!+OiP#!s_B)03hAA|D~YZnLlZ_O1Pj;(+nxm2wpusE$7y;Lm07}UI>y2=!YE;!"
    "UWwYTQrzYP3$oXMh1sI)^=xsrBw;DpVPQ?o^kefF79JtyRZ+$Q87t;>@?J{c1gzP8B_sR!HF}_E%kqEK#u0{*aBisYLuFO7;5k|w6;;?csgE%eWw&{Vr*3r5"
    "=<n`Iofgp$GBwmQRN9aPJ1fnNV4d0*vSzfe>Xvc7zH<hk!a~N+ln5UZaj3tqW~mq5VGDPFm?@-L1v5(3gWAywvJiOPupnWvD03<B24LH70?V@Hz=~`ouqt~i"
    "Tb-@R-cERjykiO3#rb91SQTwsGjF=FAC*5N;Z?8n`DD3mW1iXeLdm+<{p{ZGbFr-OY7E-dRXk`?BRV}nvt9`!^VqhHeztA3Dg~`vCLUkqqoAx2@4o}A1=eNn"
    "Calli12$yu10Q4?vrXBD+2({TWP#;PKQ4~UW5;0RSNqX<<n;;fk>@NcrKh*#eB0)^yd%A>6WiHx%Lk+u1r@W2X7!=WWlPlF3Vf7(41AJ(3VfDr%RbMxXI~_I"
    "3A=l}GQhhJxKHQnJl-Yowc)+VzWiBQUo4XnaJ@mbPY_#Q^!Yhm^!l2C^<g!-ocyxJWZ7E34k=#){{p@Nz6ET12k>3?Z{Yjv2jIu-r)+2TbM_zL7sIdqy^-2N"
    "eZRBOJJ?NWowKy_SHyZ1+=ILnc78pfvd&C+lV->Yvv5^enWzM3yq<LM0Y%zI`-p^wd)xund*FJ1S1rf9!@9OMVS~>V%$AKZmPK{-5$Wm^(NxUmz6SjNb^uEg"
    "Rwk^Id|oeEY};t3ZEJpS^xk0`bzjLfG*nE5cL{M-T%E9nB4J6hRqKppYr=NPx-WsR0M@L96&r)<{Z7=2+Kt&sbGK~uk-^w-bYd_%pL<)_j*SVQ%6Z%5Xxrv_"
    "d@1LCEq1e=mVeVMEj#>da6aEh?H~O7Fgt&y?EE#`mHjtiH>^p#r<iK6Yv#*7+H*;HmQ4w}WQ5(o{{Y+m4fs9#V-Nn!{sOW*&x_pUC4iEKz4BdI4EOW9ocm|g"
    ")}FX8M}6&;m&!{gl*yO-YW|6^9sIt-fZp6mwG8bqVS3qHm__+G->=JxX5$-vSFnPVt*a`s4fOP>SK>~H&98~~g1`RE3Yy2>3H#(LNLk32!nbs8Ntj`oyllR2"
    "zF)q7UeQnq*q+6(R#Yuo%`Lx(eQe$TjAenp%w9q(Ft7G8EB$(}8E;ujnP&OK&oQ9cvt4rTE2#j?y8i>LcbJX%9jpBYPylAOYTz^I?I+3>B&=ktIn^EOjc1&J"
    "&PqUJ0;NbNq2?<aWu4jVE9u8xdCI4LY=!l#iq`zFl$23f)n!Q)-()fAs{NzH+46nD&t^@>*xNSiboU+v&HgENtL<$6|FRfo@*QVk+w8|Odr(%C&&oakI1o5U"
    "D3{Ok*QvN3FEcxFW|i3;Dq&BHaoUow)jxr8l<Z{A@b6peKA@S;Kh<EVJZAsITH7uMl+P;woSz5hmGdfj)%=izLt*wYe5RgTF=}g1DcEyW>aAK{JwGg=hWD)P"
    "i~BoREUO}CAC=ciSPxkit_yhOv~bRT<yYq&Qh6*pvxD-3Ohn6xy*MV0S}kf<r#aLBYUYOnN90G6UA2Lu4aelCfb}@{>cU~kd3@F}`LX%&`3VVGLMfkdWhf4)"
    "$QbluiYnkBirPAVJr&o9bjS5ERVLjnDBF`gIcNCylHXxht}Ab-V5R9PHhu;RpNG<?C(fsLE%05A8@!4!PFOX<446t@4Xg*-R&^a}qEnMR98)X*Y>%ra&7!aq"
    "(!CJxXe>Wv`{ptGNtPYxeO!Sov;0gI%!0M-cf+5)dg+cN>_xQ3yM(<(0sBhaInox*bk3@ZJ~yivs?u*Qu}*77ZB_e`l831GI;7v2y5#4eo9#Xub%n*X6YKlG"
    "WVQRn?+yP4{x~u}Qbu8ph8{6PP6AHOPt8xuPtVUtXqx|)sc#OZyyJRSEkS2j6Z%K%1IGZG-%2f$*LaL=A44-g4nXVBz7z9$VBaYTr{=Z&6)8V;)dVuYJ};zC"
    "O`T8g=HWYkTHTHl)ceXY;&mRqZq!y=&>HsGDS3VJ8q-iXgXU95W^s~OUf<9_jz|AC1`nT=x5&>=xG>jUa7u3e@mgDos#hslNl>4)XeO4EjSUTihIu0ZZ8$4G"
    "J8vd57h3q)D?IBkAKS+MPL+9KuTAnZq0e)CZ=u)vawHztltyOG5zZAW&<icW53Tby`NavB<@|gM-}^)^q<2{LnF@S|zejdr9#bK~uB^Nc9_E#xwe1UYMaLEF"
    "M{|w2)9kW2<8)75<S`}lu+t3ts6zIFC218Z>-R$kkY$$2aDGrB>kxf>CVA{!@>T2nqWqEsT{Sfu?fLZp+dj#tmGyBtjnOQjdHx@mMG3f`3w5=$+1HizE9w3T"
    "T+z|<SVd|i)Pz}96%P5EmFGyR5}b{edNBGv;=7s&M@pr`xgSTzvW6`UXe5g&K&vHJD~r}D&XLe>1H-B0Z43YOR9`O|frn!-8ykbyn^JC|uUqC9BwUo&$zoXd"
    "UF6;U2kGA0sX5a~2>HceHl7Exkm_UGEoszN0O#JNka2DEEAp!o+QV@t_=;#bIX`s|PLuiJ%y=xzX$eh4-|Pvj6Xzy06PwY}3&~QnGp4QdU<oznY*T%(N9?T?"
    ")f~(vrVY?ILF?7IzFM7^p#8M*cA@{S$gdLG0Uc-@OItbWN&t_tZ9D@#-X6}o2DmoAF7KFk%5Ma2GIa4eX(i}v+DJ6pn2L%ug1vS*jeRAJbZ!2RgwDP<h0oM;"
    "FOS-7r7yKT%$ilxf$R@`GdJb>+(td23#hqg^sLF_*`;~R<q7N?^bYzb&`J15e!WLWzJa{hCGVQwoOjE682$;|Eg4~9kFx*Nb0I63u+vhGk$TG&wxO((XL_9@"
    "XH$)|gewuIT!QY^6}{ThYti!lx({n=69IjPae6CXoo`4uuqX$VFDeujfhUV9vg5PY07oWpE#%dtzN{X)i`2Yn0v^WbhiLp=R6=xapvc@rk-5WgCy>^wmQpEO"
    "QRc<8_G{y%qEA+O`-JOgX0@T79YNQTT<LL*IZi4#M(%Xd7%~%h8F)Ky<!&_G4)k|}-B>ryO>ooPbT`x0m+BWH9wL9%9&mkW=GXO@OQ5gDvco!8>Q4)bCelGz"
    "ov#7Q-pM(RLy9TI_r>o(C0EHEB2;q?^O#1mDrwx#^fu`V!h6J)vKF-=ueM2O=X1S1*}^%gd3l}GE{%I<iSjK8x0>_9`5pJ5lhhGbOE{b9kZ^<KJ6FptQM;?u"
    "fNpgDZTan>wC?#``GbasfSxk<i)r?u+6B^zjh5e#(3L8@rIXA>s}oylVLRDUMJ>Laa^t>)`@t5pC;0yFW5C|Kjb<IVCV|J(5!p*Iw@Z$8C(reu3feQFm+#-^"
    "G@o%-RDnYkz6WL!^MJ$y`+GR=Ej%Xl^=qQ~iTlBx?#b^3Y@0px0QLEh@ADDpg?U`)m*d}Q=?aBUkvsw`6Fw0$z&smUFORu7fk)Hv9*o*LI*<QI{wU0;56tKB"
    "gnkeKJx^h8sE>Oky*!rp&xZ(43E@nJkA0Lp*$1#~_9~D41kgVpkPplU`M#8)po=bEmln>-+a&UwTfGv_qpEj7Qpd4+>rM#kKu1t4j&OIvJy4;}gR?J&%E&#O"
    "4;2)5Caq<MfLVE6K0N^|rLKSP10MhtiXFiBMQK;Y?d>W!*601=XZDHZF|U@sMmPP)TkNZ+fuV+H%pQcfRWeF*!?y2C&<s-@D?!a@r$$ieBOc()@9fvu7|rrq"
    "d?mcY^nb79OfAQp0GtTi2iy;ga=l$2_n3P~814GG;cl24Lo1@?1vk-6a+BSQZi<@<yyWaIQg@Q=60Vk21#4IrpH-@|?!sNdLy{Go4G)XLExmmNA2ansm5#By"
    "lYGFn$a1&w$DVbhCye$)!ovxiNjL^7WMI@D1U#7!-UH5G)Y7p0+59<Rlwq`A`;@y9P*rICZQ_-le*Ob##F)WSF<8CN(3r!85yEriWy@od509H1h|xLf7*9dZ"
    "tn<4?#az2AT8%V9*lVy;y`#R>vj+K5>|883uN0i;voyz%Fh7lpG6r}a7z>O8Z2JXZd_Dn~m`?&G=P%|{@~Qbt2`~HJ1`1DkKeI))J<NNaEgD4@@Mx9^5b+@i"
    "K`nUp7;m3tpaDHL%zJ?2Jd%`VL9LDUei&zF&GY8DkD#BR2V&SC)2RN=puP00{8eF|Ff4(4M1OLBwmlUHz0HK)X6JMAx%q3tLSYdo1GS^73u+Fk9=Qk4Np6ji"
    "xyHO8`Hu?Kn(?$>U1kb3Tupb0aJg`k>jFIOo^#JbH5{KX!9APDu(pDq$0t5lf6O2+pbzHe^Ai^22V?=fhxCN}WdlCD`;<JnstQ;7C5#kBj1nE#HlAzS&wH<_"
    "^%IS)96Mw-<n;jZU0+fDKvDiPJ~r6vHsMF1s=JV`yY4XbhbyqzZmyf>UI$jXTl}gSR)`+5L)D6Nzn`7f>;9Ar7Of+glUVu2Naow@A4tB@oZxv{c*dyyX{1-w"
    "{#@V{!>higC{umTu+H(U7*zZmI2Utg0cig9d~v?aupC&CACn;`c}8X9{4>b>4E~Hv?+NQ2M2)60L$d5;vBI`zOXj7#BDp=EvT>>54d6|&D}ajBC#carv<}rX"
    "$)I@=@^rM12>O1aR1EbSG9n@D$5?CF52|vM_<O3HhjDtv*QijdFr#9y$}9rkFQIv@NLcCfLQ&;RFqtte0X=K=fnL^HIiGwQ!#1D-)Q*+;s)V<wZdm3><Zy%q"
    "q@jfg*k=lku!Q2cTt<1zup0WEMb^$sm@nSKKJfhI<ein~%+<a&sQR^*FA__5Y>VoN6~BTkTA%QNRKmq_JYE6Btj*T}@6yp5=$c?-zB%C|;{oPPKkjmhwQa8?"
    "OL$&u<o?WB=;LGI6JZ-=mt}kcW*bHds|NZ}S2-LT`;5U^;5^;~a3&k`O@0PjgpcycP%B>qCb~Iprkn28xHsKuw+wdWtK7@(4fi(G(6^zz41PKbzQ5zv0_)tn"
    "z<T$d`vBNLU*DJ3MrwT^UpE2xvrW|A=oaNMD~)n6?zXMo;;5|mv)GdG5mnO}zB+MLn!N|Aaa7<!Zvks(DS6hi*z22>W!o5U+g8o3@U!7Nak_WlYGirQ?1pqL"
    "m@Sn&WFuEs%WPRCG)rF>D-^cm&4jQAVDKpP@-na!^}d?4@~-H8Q^JR$nXLfo_49mt{x9Gg!?zSk%Q~+>)*xSBTUd{r0o(I0sF&{zKL9^MUVNT_$9|E2NyqP?"
    "V}8kh1$G&hC$N@OO>3hzTKb;MeGAR=<NQ+?;j4si$rmx3<%~~(&%may$fh0ncZN_W6jc&yhpW}<gf)9$?^A39Y<m;s#b>}aumbHs8-DP6CFNJ~gGIf-l>+<Q"
    "3ON<?nfKtk!Y1*C){~EXB?}b|z4o>E<H!6b7=3rbZ@xd(%2pX&YvZRhD@*A2yZqnK>rUwRe+j>VwI7PQum@C4%opURn6F9K+Uw84e}G?rUEb3g{ZGZZZK6_E"
    "riFWq!T$eE{qM^EOMO_jdwqW?^Njfyo&7zWk3IYseE55wJbF+RMaUKA2hqY$G_##DD+}jP4BE#d`~hQ>Oej^T%^I1nWxiYyzKz<f0?upp>TZbB?`UlF&lTJ9"
    "7|kc1_xDko=k?Pb?36j-7=Ps4Z=gi6imDRllZCS>Ms<#w+~y;znz2=c^8%0lSTf*q@CeUeLCbLm$7_)8RQP<0R?ZEw55a!7CVWh1ZlR3%IN=kC46A@^>kkRP"
    "kTvQ{#OfUtVB5P$513ob@4_GXUpYShQUWwly4W{izaqqh(R2R*{si#s7#2kdQd#MuOtBC2bAbFj#E>aj>;=cflqt#rd&_zFj%Fppjyz_U*ZFR$9k$K3vZa6d"
    "dG2Lq$LtTUUwp<`b{kIxkMS(`f*dOC`%tz*)VMzB$`f)wtSljNoC7%aZ)B+@D|AKFS^Xvc&c(~N&3-LOJ}m{H4{>hlxr2qu1zY#E*#)S+IU89mICJd1OhVa0"
    "?fX+GMV=@vzvbA)>`Ru!98eqx93<~tR}~Hs4lVx7WA>JQHFDTP4EDzT@q4J1V6K1VF{(cH(_T@V=fSg3{o#A-2OBB_ReZ0(|J*NYg3n}I0woMM;{Gy&1I2FJ"
    "*7x&MDoP~LhN?vkp{8)Si4mh7Jrt-0R4)!os3B(_0UTKzRn#hK7e^-?N8YiN^Iqm>4s2U3WE-o}=s0rCgd@Dq6;&ypTP131Puy2^v0UvA994WRdrQo+A5G;_"
    "c<uU4suxDCl=qjiPZ4$y$}fI*@|&rmp_kId-U<5@dcWpmVNy1gp6j;o6NGol-7Q@mbnk{Mz&9{9?%J`&e<LgK53=(Amu6GD2w=o6bT>zL*17{vxY2Hmd!F92"
    "c?qr-aW(i;D&(J(86|uqY%LD63AwzV<Zn5OkEIG}*mAgu(oukI*YbL+Qyc>v3tFpN9ABIW)H9q^oSel>%jTHBw`2I-9=$K9*DgxMA_lcqB|)>L8tAJISi>AA"
    "94v}M+tp%AIFhMLb2)+La!OIZXjGh@(1fzxQeASiMwFQr^&H2#4vic83cTejEY^}LrmlwiT5S(BtFX_y)Q_bBXnZeFL`kV~tc}A3Ro^$oj^cm9Z$b%I((UE;"
    "a|gI;u7>-E;d<a^r}?ea^0UMF)rs1{&*)q9ZauQ?G($t+4ASiWq*q-H4)NA1Hx;)R4;mf<dKZrsj~7oCgNq@><YJ!in(!4x2+``xR-E89i{6jH*w+UePX$f`"
    "%NiGFQYEvn_c^vW!@AH@!-Q}w!@fAyYt6Qg7v-Jk6>$n#&<JP(oGCOlYeHC04i}vtNs%~C)WCCL>YIwo*duVhw%vfn=GZi%JYX-JYiI^E_uf>Gs;y0dvw*XU"
    "a}xB-^MDq``9;g(0&kPnm<HbatPa~&dr}(9*iC8FW(m!yUrQs&Z$7ti)W)7{`%LLq=X0*ipaso<X-P8)yCvQo=x(CCfHecx$%7<*hlw?{DTj|wV4rfFaQ0|T"
    "^Mvy#Q!4wF)}nQZ>rgGo$-2~|<rG*Q${Tdwa<E^?4}qTRc=al^;Wu^QY|M20|1qM3<9!xk495Utg%^NsuAdtKul}&U;p+)#Kjp@`rS1*)rdtlIaP^Crv%S|@"
    "JGR~2#DOEws<^OdU9>S=3|u04W(mqY!DzUiRDV=j(*({83+u9xtPZS_R%8Ry8o0=3gK`OQDR3EZIbhpufh&qDfvbvkz|}?jqC;^_ac#nNK1-D5KC4=YA}>e?"
    "c3o&Jya;GRK4L2rwnjgrQa|$<S8{#NN}sF8Rs24Bb^zbKyQYXy`?TISN;uu;Guz+XXB1j|NrHOn3i1;Areo0wxE|;%+)z9Ux_Z>%*WU1Y+`Zyv!)vS?4?C*4"
    "?gcmB&4Jfs;05}=z`f=c1Ns$N0;*gJs$45~*BcixEycqcZ}y~xwHKq2N8fUUE$xdDuCE*JMz~?_J+R<Cx5{mBUl;#^mHM0FTUf2XEB+09U;I$~So{QR02?;|"
    "@6)~iluhnKV6)rewgMkHt3u8z%%tX|nN~0Zw685$7EpBuQ0HeSXij3>T6u*lmql5OwD!tlFWI(@v*yE9lnXH(Bm*1L>U5@AC7b&BVU@rbVegkGv?aYM=!14L"
    "N_$dPhp2rGW#M%IEAAhpxEqQa6K<lpSy%~|(wNvk_7k+qv`Dx{a-PTQNaMv^FX!`^U5c*7%|*9_TOvOrountt7u!ZVY+HNgzHg@fZY^$uT^MtRa3|%0g`=i*"
    "?5e1(RV^iWxTCRA=}a}bv8l5dU9EW`hUfdd47tfZU~Q`<%GJ;d@48Mg@Ggk=8;#9~trCL1ncy|+Hw*fZy?7gW@Xn%paaY3KU=`+4*j=1&YNFP0oDa?g{n^2Q"
    "QO5k?xmwu&tY-8vM!Bo8kuB?#aI3eLBW|OAr?<aH@lW9H;-2E(;=bbkgonJpwW@UWnzA0T=m_0ugdWs8f6{;W^fmlsktTV}qy8xpe5e1GB7igDxLB;VaZH@U"
    "eFn7uri4d;M}gizAHcRB1Ny=>$>T-8;t8OCF`yV&41#Nvn87~BHP7$#v&H%HjAPKd>fZ<G3aDq%E8*b+J=4T6*MCCcf@~puw!?<3P7%|NEU<KpY8?8l8+ow1"
    "fL7iE740fg*44gJDV)EdigW|-_AsDc?v2hswf$3OeUHr5wmBLP>;Yd1_N2LUTr9mQL#`p+>#Vxa8T|x|62{|MU^p-W7zw;sOaWdl7Ql7n*TwFH+yw<O_xZ>?"
    "AlIio6VSqky>|L|rLbC<{-%xx<=*D2t<Ae0<iq>K1D2i?D+}wguOADQ8G|}irQ-GUQ^nK8(1d4-W?3`y7vRl7R|`N<{CO6bd-#q!zCUm8=5#M&o{-T8$e6Z0"
    "2(WFnCIzi~nk*Uyanp5FtM5bN{a)g8+ve=;EfL@t^fMMI13@#m4t0h6<S1V&G3jP%)SWb&F@_56V0Wd^LBJ=r`@r6AyqieRWb2L*cLuu8v0Nd09Nm3%kgI^}"
    "B(r#yT@5!=o>+qS1{;rA(0eJvBo8nLo-0NPqlGbG0otK;;Z{?xbzKOnf?W?nwxU|Y+NZ}nNPe^ofec)n@U`rOe<(`0(ym^@8P3KT8RqB1d7b8XnBgchSB&EG"
    "aLqanu35(?Oz`zc(fRc>HK@0MD*|R6*E`GIrWajn*gMz@_R&|+%Fv569V$ks9J<cvN;B9*m<d_KydX21089jIdlE1ie4jF<m<qg9Oe>}rGm4qQ%fc+mD+~Mg"
    "Ij<k|X$*RJoVR2mjeC_}1N1&$FY#y}pR4_;I<Q)q`{=lb&AJ>S@R+ZC%HvXL*<XVV*ynJFgGNy+!yuXoqFG+b9YmTt2{;+351j7Ka_6{nU2E44u0DGMkGlc%"
    "9q)P!b$yDMr+gI~YI4t_N@YKeHa-iCFCL|<-M(~3Wdw{_IOp&_F5by4<?t_T%DA#_Z?})z7ue73?+$PW0;An?ZX__3{)+NtcrkO~ig=-01mN}X>u@Eygj$Q~"
    "nYA_WT18*+``hjvr}bK41x*&!Od-|8poUUrh@NH@vx_;!+=N#s3c51Is0TUQ(Pu&3$|SE9WvbXUO{@u;4%(h2*32ObUM=QHepqmxQxmvX>;cc2FDwv-C)irV"
    "F%Gs*6b2=Xk$f5}im+`~#RSPIw$MglnlPPewMF-y=}xVORPg&nxsOrAb#+mOM9~=Tm1`OAEmdat$XkC~(D&^9dE~{{iiMKZfrkxPeK60_ngL)BM*C^WsF9LM"
    "G0&6VV%YcWyBT5!Tf)|~b&Rg{FSsY&U^m1K0){~zj)(jj39r#^EPNm3CcwI%GMSzUog(k^O>>irnAu)^JhE-S>b13i-tAaeEGk}yHF8P98&X9q>IGI2&o$)X"
    "42j7giX*GjGD>_lR@O*V*~<yfNX6$PCixjn5m1G*5^Sc0Y*7}HH!N=g%Yfy;3c$8k0;`I*fYrqs;O*j_Vr{XmcsF6aS>L9ZYQ;4HRXN*4UHv{>R>5HTY>5>|"
    "hogcKS&~?^in4PO<)`N5Op4QH!w;~&94hGF0oHL1VTEY|YfNi*5xq;S;9mhgMjD$5U&n#^hEY|Y=w1c&O_sk5#OID*1Fg+~O8>H(1=aiww+t%!GPe?1%jvbs"
    "y#-*6e}|qx4nARC^;yXALDj{e233ViL3<dXRT2qhHT7dr-|%l&Q@mm@dK$m=#e2nu;(dw~Mo2ZY5JujbP~O#WEd~ASpi5m>$kie4X;_m-fp<pJi@|r@=NB;x"
    "{5y&PoT=v9(qwk4;WuxR7FQ&&CUG{bMx4cmfQ@~q&!J9ZKJv<l{#U;*5^uA|XCy54QNTF75w$UMY<rovR^zZ*awcZ2UypRB_*!C%Z7&w@EcK%-_aj)|qBAV^"
    "Ts#jgSx<4t$TRa2)`)7}5v|zvI#SMhQ4V_jgJNT`iF|L_OxbGLBu8Onwg6j;j}ksEJ^?;0J_EKDpBLMUFN!Y{z9MZd@^iAVZ(_8Lt)xh-71nwEbIsA!kmG1c"
    "BZ|>npYWd7j>anV&z`nz^&0za6M19{%|2!u9c}s4`v$$k{)s{V;60Xq(K{XCeGbd_Fso$=%c*X}tn$i5f3dxRP4FA^Xjl!xZ#K!iws^FL@GL(e`@fWTV*=k6"
    "_?Onq0Q&k8tn9Ph3aF=ZU?060D(*sPt%ADx8tkdDy1q^O=6BpW;B5l`$^-w-V}tw9Z2|Co8*bsdH*1QRt)jz^BlTIVufG)Uevq)y%y=vHglELy-;v`kWirfZ"
    "I?QAitSWe|FdcR)GhkM;p*<UZI)m1gx$YHMb6%x))Lw&?1>ezK0W*m+(^m2gQUkY|IcyWQ!_RPzIs<&)D~8_%+gbb!{HOS(_;0b>KhdK6K^dWWu-f}jt$0tW"
    "f!gr7pdR>w^1`y+`<0bz+g}5=t(s0jjei%tmuJ~YmRi_P?|Vhtw&vqz(Ja@b;K$(4_r1<TU1J?>Gqr1n@SX5)lM_ss8PCnOcSP?~^6Yj}e3)N^U!{-0&vF$1"
    "_Sa$;#4~2MJgbDCwev^uXYp5&(RFSBuXwTQYy^L4jo(Ze5`$xhUiF@?lPhO@{t2(>8vths=eP|;;QhY=<AL7UNJ>KOvYyc==!Y2eN{Y_^SDNW>34eKQDm%Q6"
    "(A(cf?KD4rg_-_FenZa%{u0~R9vpv{cz$=}Z_6KYX6EoCO!<DpJ_-A}%~CTtTfQ>byx1Xm@T;&(_|5B2GvzlQc^==k{~~L0v4*3FS}5z;o7l(cd+9;t>@N#@"
    "WGDHI$FQ(JEvi}e+-}ikOy)4RewO&g?26R=hw*Koa3NxU(MR{&e<c8Xu=yasxpJ-wP|j5)s=4a!FxUsj)O3dfN4O(_qu~0YwmTZGF=CE!Uy8lE#6FC`{%(J|"
    "hs1Z04ulcQx$+4WTt%RgJJ?kQOAbk>=6*I60r|_!80TTz_|LXWNObw`R%w5COKYz3gI6|d+qOg0VmOAoY2Em<DB%KJU4Ao_Cnk4ALV6c&Zv$t@t|BJPj^}6F"
    "rJ{Rzd^c}DiG$@pI>u5?;#0vyC8jdzC8jDx>QGS+KK*GwhwVw%7p^gJKhO`ZJzsJo+;F(coC$k@L9hdw4m*RnZa(pve4h_jm@8=)^fp~ven7Lbl=AT`WAZj+"
    "Z@7Pnvu4#*5Go3ls25A{HlDj*)ZX9U3q44@RKfeFvZ0DUTB+*&gC1fZ#bAb|V7?vU^X>Pjg6>7LJW#-RRg(NUG(q#ZraQx3=sE(O0DVrkx6Cb2$;=2NYuh{z"
    "?N$5bs$i)tv9Xkq$m}CpL{FAWC{IVA??S9X%qs1HT2@uGS53TIU92lBJ!x&RM}?zp6w+$PzK=QD)lJa*`lkT66R>UE6`bl$+k=L#5pX(0rLk+`&P-@3u}2Gz"
    "qB}YmHN1aw7jQ3t(fg;spZ4^C=Ru*T>jgXvR~m!dQ|@U}vL)4Yd6OGTDHB1&R&!iGY#U=}+i`6_SYmakk5GsdXGaZDeenAU2E3!y*wBRh))aPIXVW_=&D{Aw"
    "OTz_ZiKV*4yM|;O=a_{(9#c!=j%(bp?)ZcgOvWDWl~aq3#^aA8b)1qAss-oO;XdElf;9d({?XEBT^RKwqcNp{tl@!L-gaKEZCkBNsVlQOk$P{CaB6b?v2+F>"
    "&BvqK8zuCiPa%8OJ?{F$8ZgWaaf4w+7~{seQSj4e={ZLH+?{cBAAA&j{?1GAnoOU)GZC&9arIatPygbRip${Y_dUAq<>wYR$oCKFO8Dpmo)LdG8om=}+b0=c"
    "YPMo_o*@Lig`Lka3CAX$t0xt~QXkI8bB~oH^|+WbV4S82&E0vfmAe}<x-YDyc*k}$`0H8l)No)j^oc#420e3s$I5(8_PW6N^oL{n(UY$D%;<A)O+J%8ljjw9"
    "Df8iK9Iwau9X-nuSc}g};2gUEY-<H;%Y{+9HQjk`16(ZkQ7?0sySDBMLEj0vN>pSyj;g9g4CjPVi8;+@n-cWM=i4^M#kRF)jgv<GbRoq4VnOc*4+PH*1fBMU"
    "7!0S%Iu;@^2|PN1{EF|;Fw=o4;NvNDr{fj+^rI!TuQ@THfy|HBg@&{)#GD}tJTt0zG0n_6(@<!XoXzJl=K#$~Tk6wrRwM5PJkw@^R$c9hD{phxBB5o}8`8wv"
    "-;~tD7Fe*SW@OQM(8KwpF4dFL+RPx#<Xo|$xjFL!a}H`TtgV_=I#LT-Ic&S7A3G-K`eM>FbD3~C`LGfAKdkF#Q-7A0(9gvQ+{cyfYIiNYGdP&u9b^W(p`avu"
    "ALd1R?|n9WT?xD5H{t7C_`U#iwF>sgtAIuD9iK;EC!YkvPkV2cJA>^Kt^uwEt^;sqXxqF?#GQO+cLQ)EeLh-OcXL8FnqyNRJq!CcM*Vw%_on3{Qi(;Cdam?u"
    "+ZaLHj{DaUE~4Kl7-1estCNjni0I|MT49Z92a##vm32{q?FG-|a~SpY)nt+75`Tsr%@TUK%Jg?N(B5^R*j<;<(b&Ya^|QFroNsAQPo1d}I{LU<I{P_Z=07h_"
    "X=`k4=WSy;3Y}crgm!)=c6^KKkpG^rvuH4;3+xMq(UU3Tp!F=gmbm4R1xwvqP-&lYi=nj|ezp?6u7oPRgzl%WHM~uC)<d?u?>4zx5^e+ZGwtpKco%NzLHqE#"
    "-95m)vKxOe;UVxL`tMTl*_BZ=+6%#V0nN%z;)$3We5{p=O-%~5sjV5cLqhw64#~Kk!S`Jf^vG-f#<m*-t{F_I80gbm#cTHp`U$l?sq$F<;p<SSL>hgjYt-*;"
    "6sbECx_g_H8>Oc%Vk7ntgT3AAW6yEFo_x{SRK#v{6d%dl5slkjuG@Ohd~p}|C_UrZTb}gfPw4HMaI@HROJs$`YO#l?@7{#_z@IUEMfk9L1n3Qx;Mg&JVV9oL"
    "kDeFn5Br^gu;Y0$VKDfjD|zB3vG^wO#;wNI?vx#td+ExRKdBhu_@jL|3a)@pB`Z`%qqLx}n|%aAByKiQyu~XcMRU5l&tc2m<SF$QG6_5}1$;IDJU5-*=^6z!"
    "XC&A=5Wc?ztMjZqm`(3;&4XIR&r5`w5#Z78j=U1MU#{wde;$d>R%I!%fA5!btpbpK6xU%8@4oarAnsN18lgY@d@}qTcN~*pPdgG`{1nLSJ(xo~Tb$hjxOP|%"
    "m3ITw-z9DZR9n10dmX;6kSn$O6ZB5SLt<Mm@9p6AM=4iSm8ilY3D1B=hSJxk=^2e-bUkUoYfGHPbHFHhY9mDa?u2_JZ|*mK>y_|uROe$J^f6-ld(*Sr3TN2k"
    "?opYUzWSKV>t`};AUz?*1QjZeN)1up3^2!s-0Yt)fYcf@NRAysenfvhLtlr|*RcuX=$-5F^ts{sso_)obq}|Kg?&u)V$hb9fnvSd5;K(kB6uMEMeqPqC1Rc?"
    "yJMcA>t<BLNYX*fXus<Y*zvVz_6%!^6=fMk&j~&4kFyL*w8%<~c^;19BZd?B{W-CZe~-TBzYDNTqUQl)rqjFZGkxaHp{vk&kgNQz`>S-d&$bO0-vkSX)3d&6"
    "=kp0;A^I~(12bTpN$`C#j4}tnJ{Qunz!TxPF>>^D_&rvL=fN&i&1;|+e9s?$g9=&*-xtx<@#}C^gYa|2%jEC8SJJi3TavqD$O|zq$Y(T9Og_cl@{(wGoM?BV"
    "f2Puc=S`C*ET{U<R8e2DCu8P9FH_8!eD+JpnIQ`E=$=bpK1GJ_yjbuqOv)nq1m(s4PFTz`iTw<DMq5XW(fC{D(R(g>=jZhV{yS&28Y?{>GoRFi@#epUUI@RL"
    "N6)Y<lD~*X8<%>}K8)yF^fWl`$T7NW<hhpcgv&ZuEf!Ee3+Wk{Fy9#NFNU8XSxIWaXJ=yGhO-vXY~S?9T9*5B75)iZ$1JD+uO(ZV<#Y}|lf$T9Evv<XcjS{="
    "*GDXY<5!Beme6eR_!acD&oZ)O4PC3?70Yt+(Yt$~S1=pt^IbnAfAU!G&}%Jx<&oFHYb_id!&g5sAIRT^$KZX|&GNiaKt~UMOCH02SswF|tDaq*4K6CW^+hwN"
    "niHTJO@n+|0<Wd?*|zwa2Od~Knps1y)ufSkVZ4p>+6*J&YYW}`{ZM+|>OO|A8T@+}ev98X%Kx{({PFvTaK!%u)Ur)y"
)


_EXACT_SCHEDULE_987_B85 = (
    "c-p00Ym6k<RlfJssoS?}YPx%RdU|?#XLo03XJ^OjeU8_UUE90s=Q>_Hw!^L;c{p|~@{kEqcmy0n0x`naAqt7HEaO;_A$AB73JIVf"
    "1rxvtQKW!^6rn^w<e(5DAtHV_N<jSJoOAE3s;;i;?uA*8Z}(L7t@}8S?|kRnR=9mE0*nl*4uAnj4uOsU6hH(2PXDdO23@rpRm^`^"
    "put4SD5ZwW^hT+aE<_qs7wM;A)I2KPG!QQWa4)u*G^Af?6M*^F+<4mpSm?HkEiU#J0BA@T)%F1H(MBZ*>T=X3HvAj$uWi^kol6^?"
    "c4xjv_YRf^{JX@z8+$i)uI*mixxQz8H~((!-`aEUV>CDUp;_dwKxXBBn#DiM6`DF<)G<u^?4In;;64JO3wf{ePv*UczwEE;9R1r="
    "!Gw?RslTK~KUh8Vt9^@wkGpFT6nfDzk-8;2JU%Ia;`{Nz!4a&wj1aL<Lc@%8P(6OQ9awH@gfs)x8a2;l<g+Bdor)*X8-7-=evJFV"
    "75^^^uWAQ#sEacWd#S$f@~wC5n$RKg-b5I+J5yJDp7h=cmFDd(4a{iXaXd8pxWW`{On)P*1EGcg&ulMSubl|bl|E`Oz36>;QL_@#"
    "tJUll5qQAFzkR2dSW<c=5?C>^|HIN~O>0e}8QP0lnwrTpvMGpBBgI4&8P)*N)d>=CdFfevfwhOk!Y?|CE>TyHX&aNJjg`&&G0Iud"
    "cSrnaQ0s0}XjXmS@ul1+n>%CtXSPJ#lsXqjEPG}ZLvP#u%H3hD$g;BvBS|w2DcGW53CRNC2^lYBGvIO5?7*U!T}?Z{Vi}Qm`*zkH"
    "Fr&7@-4VG)UxY<a&Yar9CrtJ3d`R0$lc?R_PGrkaSQstFkN*;x$f-zBaCt{QnI1Tw?9CM)Xt()l>YsG0Z~L_LB&LS;i8_r;#RzY&"
    "k=1+|fEOLK*RK4NEB-}~hTz1=wa6Y*gBt(DGk`JO62pz7Ym4xtX33=YgWN*~agf0|Rj(9_wjY0I21Dd;ZB4)RX`Kja0bV>_0F-N5"
    "6*f^sH8St$1CUi<<&7*@r5c1~g7$Tr&K#}u$Woig!JXptdG~s>U@bMTZrai~Tu>@Dy~Tc}n;2A8iLCy+_P}Qy!MoJI*6cJ_-hS?l"
    "zTfW5!hD~9Vf>xi%R*?YoKBFgn3g~rjpcUyBlm6BIlGqEfFMnsS~<)*1-}gorO8eh^5vylXYl5~94xYLb;%^NL#bB!o4nt)ypi-@"
    "He&5}(uO<#k%m0qA|E|8VHkc#jaQ-H)xn8DtqWKZ;hg9t9uKbusVJxJwtG(W!i7K3z(~8N76;kKDRMC6nX>UFB#{zf__kuB-&ESh"
    "R78sxp6rBDi$kzv;wTV-yK?iPxkpuMD`s(}@}^UdY%<E~MzJ@`!E|&4;^w0_tx=qiIsYo5s1qPN!@a-YU%YL)m3fSe0)ue%OVjWD"
    "G+6*LhvZ64hm8@EYx$`NR(HbcN-{xM`;{KZ2SRwv>CG$ycrIdZOKZ9jt|ea&G1zBV8|Z0~>e7`|<njIf+Ag<k@{dfGW`BZ_8SVYA"
    "fZ{*$?P}c^@qh{vD>r<J-Zbcfx+aL>413UKpcV@{4ob&RSV!huuwFC{9)@+!MMG;&EBYPKtptE*b${MW#oue;1+slchDdInaA`>q"
    "X0SNdmOheTW)qa5g`;PjuD9=XRyvt}>$8kMP&twCN{u;0D&XlEoyyMHpZ$5wj+Eq8&`B1Zg4l_qN&nOl_Qdm~y-^uKvY4c+Q+Men"
    "6Cdo_aE&}M%b<ebmj*?+qoA^z-Qg8tXLRCS3D~rrS25fCxWh;+a@~a?i%377aTa2l#E^+w5U+wxEp^q>at|EBkaZp?vvb$ZH3;Cz"
    "K}OKWN^?mMi*A!-)4H)9Y~)TYTL|>1=DO0B6`W~#lVyc8wr9l}7|`2`+N-psb>c|GI&i5*BY32M^WQuJptfkn?%I#tk}3UjvnxRe"
    "_j|x`1@gyZF}NPgsWFWD$2!j57LZIB@zQoCZDi7-Di;{m<?YN|(AZ39i?b9HhhNYIH7f>X?&oF?SGzI)Jw}s*fP|5!*HbcKCpah="
    ">xP3aQmlpxDg(J-z#|g+{ZnM`*l4KVWTC@BkFLC?Nr)**%=G?&5*Dul)^Xb69@xf$U>Ji;Oi<{d)Kx{b4h#oxNc3pxq)F2!$g#!r"
    "%ajl4zHx3|;~f+1Xr0L$%;3v{A|j7Ni~|AiPNs-ZRzKcG=0Xt84VL=y?<5>l!xhDNi51LZh`r}$2c|FMmfF9ye@nfy|KGaP?k;q7"
    "d(FZw8o`D$63S!_i|2z*n1p#4>+Q?>xQBxBq1aU1vqzkGh7{C@1sah?6oNyS95n<t+^23R;?o$~%JCF9tP#|Pf{E$nD=efWZR!`9"
    "cCdy_($_)<#@XVq1p-N3JaxL=fWiC#b9+X^L=KLfbybTy){(`SCSctLK?)KElR5f59iGq?fhB)H(*gzrx;49XIdloY(S2LuGGw86"
    "POR>w8xOiy8gdN?mmVzAJ2Ls|@O&2WZ;attZ{#X==JTEvQolx{wG!vTU&3K^<fu9V=RA+}TYCB$+{ciP=Z310IIbKrssd$QU@TIb"
    "G>e}&;9X;{1=a8(@R@yx_(_}@$r#Y_Fy8m}lBU3$WLyGpX~%E=!(bFXIK#w4M6jhf?V=@O(+;eSxjzTB|HDfwyrj`%Eb|&&TIkaH"
    "gpK|<`5#`h$cDBSWj)gMuhtULdRlLe*jSy;g_{!R<wKxH*3KO9njSfL8#*~)DNs;{jq8D*^|W3-0=Mnh1<plREW%A~fUQXF*&vgW"
    "jyf`aBs+{qh0Pz%u%;13me$##F4%8Gt&Kez){otWiY(%=OLF?ptmB$NmXWm;3Nm+`Oy^<r?t~Lf{PQccB@e!h{_7nCF%pS+((zfO"
    "_+N-h+I9CC$K5!4AzeNdWMXp$*}cabEWJgeY_&-uTaMuwODnC9ECyC}_HY`Eo)e&oTf)+A%^qZ&J}Pf=Mla`7==ZF0R^i|blMPv}"
    "HJU?YbYw&8jE5fo#99Ahd%7fiDsJe6Y<=UCf(;n5;1Brfi@YEI5;kSq<2oJiT?O@_lLp5T$_`B;-kzYa8M((~c3J@mme^=+3ns!v"
    "3kolgjZ3Hp5t7`R+Xk41B(7YqBl~4qYA&6lhyc%$E-`Rt`9^qi$8#Ai7<#I&3vi<wPuwzQ?OSl~l$;rR8_w(DTY?%_@n=w#Wy*+@"
    "Y4Ou<fa=MN{w${E`XeBm`SuYo2g1Fo^;ui^d?AF&(d;`s*x2wR!_5Uyopyw)NN85L!!_sqp4Xh|rr5oiu$JT4l;zEilJHLtOCODx"
    "h7j}T!;XZNleYs7`dkQePg!I&mM03&N^CKFQ0%wQt!LRLsL`0~%IazZ!;6J*gv~{u0<Hnh>PsKeVm*OBgL%zikC1=D3~hOnbcMgN"
    "Dha<x*$JBevt-qV{A#hcyqeJk;c_c;jE2^m476|Uct;OCFfMy8Ho~!?pE_}kOOqT{`!+d~drSIHT6MC>BE_S97gb~O_mSrUk1Ez&"
    "whpcxWs$`zzhWG57X3gQL5TG8-ADo;69)r(#dKCWjPw}6mUcJb>EG5bg)6tQ(Lppu;d6+Eq5|i;Q*dd@KCLY57_MwsW$j4C!58&A"
    "r&1jcmi)`4dp}}Mt5esW0`=!QFkSF!xcckC#9lwMdOaO9{Yyef+tjV3_B3xm>Ta{hKl&%+8O%H+2?2?jn0NQEWOpIH?8^TW)HbF?"
    "TggHd1>^!!2<`Y{+~8k0>tt2Y>QonTz=w1ivhPKzAfVCB%&WvQq3S%KECN1qjbDEuEl0B*jwiw}msN_jA{Wt1%B*58%PMD?T;_pL"
    "oed(HOc050M9=Gw>*vHeehoKwtlH%icCJ&q)XHh!)PV(!FWYR6gn=<;TklgI<GNmWEJi9L$H=#fdsfME$vmzL8J742nu(p!B0JqV"
    ";psmZal!%2p^I({TN0RzknD;INv|Mk{fT<@*0SkF7u1(B%0g{_f?qSzXdL&(F*OH-tx@X!lC`d<2Jj*F1uE7;FgOsuRjv@q^Tx(Y"
    "U3%JQN>3Kc0Y(ODAQppXZ%Ap`dy7dT0E;OMSt;>Dp@?Y1i0zDQWx0(@#9i$#tcAr%>KF1e`i(`vfw`io0F|8!09bLvJV6?YtyQG$"
    "@p&Fwr#)arnWN|^^&k1LRuudaa=%8!JX-40x8nl{6FU`2WlL<QW9F+f&sJysg@R?b6$tvnWNIkXMDQ)vcK)mSjNd(>w-aB1m5$za"
    "pv*{S!@i&k^+}^eE>8Ws)%hRyb7h#CgBVUuA_X`xOTir<4Nxd8W-jI4$6-5U<;X+Bu;F&JgPz#ef)9g7xN5pjJ=@esx5sncU_L_K"
    "hXIp%$$y<bpiQ1LN3ZA13ptY9T?j1s@d*iL$1~$u^vT*-BI%-`k>4Y}s~e@rMMd?Z<g69#&K;#mOZR@5e4oYiX|{TdhTi}bg%B1)"
    "-&O!9m)20D*>TaD@Su~(F*TDwt~FDo{3NHz6i&{lOx-(loPSQ>GI~?OytuYPHQ!@H#S#rxLsUW0Y+^QJ$Bue#k32tf4P;M#)bRC0"
    "-`4ml3@@cvc4gQnF(LVU5;)dxchn^XkJyLwRsB2qSxuf#J*(Ag`Um<{rWUrvQ0BRimiPX{WKm6#<e~q9hq<2r!fx8P6;t`IhD!>m"
    "?`e1~WL5gBEUlgMcLpNLl}?fLb%nk`k^*$uM)&hoeLf(EkRU6ixihG_qZEGoJVd3&J#_PE)nkZ-#*z8EI431PS?vzrv4&Od^XPP4"
    "ze&LX*Wc^KkRIB$>rC0O%;@{S@2p+N=g;uVab|GE;KyC#y=ssz3s*C3uY|}(Ct<7vY(qi~!c|%dS0WqQs@m3I;=NDrQZEj^VL-5l"
    "wUce*+lp>NTKFJ8NW~NlNJSn=nr(UN{?Ut$Yp>0okAyN}$H^Dkig7%V<v)Z$)9#fC)LWg$;Q0?6fyB~oGWsB7STAHB4E2ZDb(%pM"
    "S1h3E{me4S%d#4rv_9xn@SQNIE#^L1YeG9dfv2<$Su|Uq9uQkL$l)pmoSJ1jb8^k&uJ=wcV5Bw=)pKGlWXtM$vJ0<-5q(#~kA3xz"
    "zAY8B`!SaZ)}#wz2PFUPemxi+BGd;K7;mc!O5J_o+k`Lno5q2?q-^^z+T6RzI1V|%H+<QY2%&7W^>Xx0GBsOVBK`V>9;#Nm)n4dX"
    "uuZ!vTyD2b*jgaFgRR;_4i(yGI3Tf&xYfnLpagJmSMa~Agd!{D0uZ<)XW_J*43HAS2uDQFg-d7oC`3J|$t>M*Vf!U3BQ4hF4E-fS"
    "l&LShdP~_{PVk2}2-HEnEhRF-$c?m>MHgNKAE>Cfgv#9XWr1OGz?%y#Ie9s@ly@bQoOiEatq2q3)rb{?d~_H}cLWlm<Ml14JA}-p"
    "<?3l>%zdg6y5fmIvb)_Av*%-9S7zChVo)I1@~I|uwA=^vHI4t{qGAPiTj-jD@F!x>F;?v?mnMZjrT}9?(06{yXefev*e6C0%&_%8"
    "47&a|8p4i!F3$f(6XEbSMJ0K6gifLxNkf_U5R}0UA8)lBy6hTQRX%C=P$tA2@w4{2wqAeVkwU*9BGPS|F@De@?G4FWd^WSSE{RsK"
    "jpK7%L6y!?u*keDS^9>JIkAMHe6tW)ADvJ~6S9%tu^GqYM2dFlKO=A4o{Vy8pTjsj&Nb*Pt{FoLT;r$aQs3_<^z7i`o(`ymp0lw%"
    "IDS8E4FC54!i`5&57)Qhi@2-I;~EYorADC;SCNc|BA}x{9$y%qdid|L!w1%;GI;dXG0Av+*ddbRgHns!^_DjI5Y(qq>fpVWEpc%S"
    "gwk79<C`#B+SHra#3_1<3r`4&KSFa)BcW~LG_T^&uG}Gvb8Y6ays5Nfcz|t@k{TIY6U+86j8%;o4^xgvPm(0D<w9}`<P&vdmcSl%"
    "vkY$b4EB9oC#(z(LK)GnZv~Ot#RVrSp2&S);nTc-5iP4YI;ZYjlQRU`B>hdnAE_h|Rl~v)!3vz@;vHS08lVzQUs+WT6t3<Z;V2z|"
    "mu?)uV}0D80d`xeaKa+JS~v}pi|xaexk)IUTM&xt(IR$1K4p|;QrK1(+3m1*?g$8MggNjBNw3mi43Lv33|`=)Tr$A7XE9a*8HI}c"
    "s!UV1iXV&G21&2+{redIc1&-@rCA0gkyO1GaQ6JkX*brmDt3)mMowatF28$Nm&Yp$z^SR5koO^}KUr{#64eZw&0e~Dt?`tr!mp|;"
    "V~MyVPb^oTi+VL`>bvgMX}hz&w9;v$3!Sw^mT9%iaknmG#j>-$&|aHcg3W&owGCU^0hmc9F2J*Ok^n|8Heq!pZRt^*j^gY;tjh~l"
    "I)+(T1tq~923V9=KnXlLv_5HLcDTU3SR)h8<d{2Z?AnAa3H6otl1J5i@X@yw0ZdB@&q5uFNeEU+?M1Az*zC<1bJbadzO9Jk4vnGw"
    "`enbbNK}~Gw8ph(ruwLlh7rEc7g0lK$MO~sUqt2nR^p0OFa#@FYyzhonu%gxoqk21vSrDrDdl_*xy!|VW;PO^)B8HWd4J0-(zR5;"
    "v`vD~hc!6NOepOfgGfP+`Tp$+c#2g{v0C5e!|$)?1hN2k=nmcTT3;m}ym??!^Rfm{Tv&#Kq(vq}5l6O1pj%Zr-37^hj4hwll8^_y"
    "vcw#W=G7)S<RI#OiA9~A(5mtfMyG=&Zuc8HFw`IcH_tS%f<q%8*Wg=QZ2~|ZI9)2<H3W~zda+A-VG0sdA@ql8Wk~<lKdgfkRHPAS"
    "I=&m~S%&f{F4}di9j)7CGqO~!%9~Juwrk(<P!&|j*7R61HQ1~Ubq4oyS<LvKEk>G8fxYcOC~BjAn#QA(rWAZ*LZ;N_{M~oNax&W!"
    "VP4eg&(7-IBeJ5)MWN%>u-RYkQX%F@=;o%)QHdHOU9PK%2bqe)2n*d`@G<x<&89D6Mp)W<_PPbV>~uYjlCw#OCEt}*EJ>jfn9$a{"
    "s4S%=XQBPUFI5B`?>%uNs<Y~r^X<(oygk14lBKnS>YTd1Qy8lB+<hVPE4(r5Nkxu_wNx*d&{h=m`qkc;S_Fp6yYY(U<9w8N-l|5|"
    "D#s;5caB!gHdOMdTpam?1jq5N9Z7Xb=#9tF>szZ}iCk=yzJGZ&VX_(n*3DA2kZ;bnv`cGtvjkEiq7aU=k!4;;=akXc)uwFfOh1LA"
    "?|Ve4)lZUN+TPbpR`-xmU)ZrKsZMc#p0iW%)CglaQn3{Hi_J*t?~N+8Rn~C@xGs4hRg`jv<ui1^z5#kXdFOyXke!WU3^Y>~vMtA}"
    "K$|hX>++eC${Ai5N(P(#hHDMf6x(LUd`oAy=^^(^!YfkuYjTE-CiRk7(`>3?5*Co)wzVcoRGdUj6DQC}^jznwITJ;gfvGv)+EQwn"
    "_Vq+wplTiQ&&Op+!sv0jb?2TP=H%|peCSvhEP^{)N7%$X<5S@BMJLqJt@2e6n<O{Aj49#pSBH4w)4Jw7DvXcuQLJSmL;cWz3(>hy"
    "zSLIuiz{eHBJr(U&W;-!YVStF#7wNU2y+-?e^eNolvhuz(yaPI4tGQ?+u9k<DeMimYFBTGHS8-Z)@4=S^S}$s>GtgGu#xnb32m*|"
    "s|UOA25)vmZ6%HrZ()tMIT6P~xA+9b`R4NKe4piMhqRegvBfEY(Fqp#2`x(L4&x(PCNfK+&<68@lE5DK<nS~5D3pU`=-XUZ7qUB*"
    "9G|u0LUTiVFkvIr#UEkbTX?2&`c>Qg9q!>UuIY>d!h92q!eZ6-)G{u@xOKRQ-KDj~Hnpp}*BFV9jI))hW$4F?SG|(wqNanv$Mg;Y"
    ";md32>UZMAqNK3cY0x&=?9k+~%&eQSx-sv<Mjw(BA*bgTg@Y?b7%tUR%tO%r_RC+sfg@$1qeWTaq#GY}Qm864=nmMAo8@Xqr3>!O"
    "m+}nrtZWQp#(VU+NhUlHjtxzdn{%TTc&4zD?EJJtA@qAweo)Sf44)^H#cld52IMD8{gDSg!Nh)JKf!i(L?6^A&YW#6oNwwA*WWN)"
    "x;az3@Fgbx*d!E}dVInw)9UrEw)06-wba`-uO6<%XY|gkODP~c5U1G$Gs;(G-!8;wpakvM>>z+)ju06&IwDQJr6?Sj4_}|-&|<|N"
    "zm@C1{a3X3gZ=pmw;No%2D*}zX^c7IaAl=_`*GT4_s2haVPQX<T<9Aw<!+uzk1>ghjU_v1Bv3k2n(NV=|L%|pqPC~m;_^Q})itg&"
    "^Apsm;_IIP&K3k0>tjBJJ1mT3yMspzQ&6qTcdpexy}SF+d~*EI(W)^x<F~uYDR;n3Qa`v1FNZ58g{siz@)ciP+^M@$4m7gU@;P}0"
    "H;{KKJsOGzjcdO&j(#zT>$3qq=9&wIqx0xrEQ~4IewX({(}Gwhb7{N9#Rrn88C`$a+12x^v%s~Q4V`N3s~2H>r@T7MhrKCW{$cAM"
    "^!MT)!<Y3p^|zy?{?hRHWwa?rj_y0Ke`~S7)L$Gd5BPiK#NY^I;KSyhZL2?O&%e-CpKC8Z-&WsGe;+>5Rxh<%{|kCv$dd"
)

def _decode_exact_schedule() -> list[int]:
    raw = zlib.decompress(base64.b85decode(_EXACT_SCHEDULE_987_B85))
    result: list[int] = []
    encoded_delta = 0
    shift = 0
    previous = 0
    for byte in raw:
        encoded_delta |= (byte & 0x7F) << shift
        if byte & 0x80:
            shift += 7
            continue
        delta = (encoded_delta >> 1) ^ -(encoded_delta & 1)
        previous += delta
        result.append(previous)
        encoded_delta = 0
        shift = 0
    if shift or len(result) != 20_128:
        raise AssertionError("corrupt embedded exact schedule")
    return result


SCHEDULE_EXACT_CYCLES: list[int] | None = _decode_exact_schedule()
ENGINE_LOAD_MULTIPLIERS: dict[str, int] = {}
ENGINE_HEIGHT_MULTIPLIERS: dict[str, int] = {}
GROUP_PRIORITY_OFFSETS = (
    0, 0, -8, -1,
    -63, -74, -67, -69,
    -170, -150, -167, -162,
    -200, -176, -207, -221,
    -290, -302, -296, -304,
    -351, -336, -345, -325,
    -378, -362, -364, -387,
    -409, -410, -398, -407,
)
ROUND_PRIORITY_OFFSETS = (0, 0, 0, -3, 0, 0, 0, 0, 0, 0, 3, -6, 0, 0, 0, 0)
TAG_PRIORITY_OFFSETS: dict[str, int] = {}
_FINAL_HASH_TAIL_TAGS = (
    "hash_23_add",
    "hash_23_shift",
    "hash_23_join",
    "hash_4",
    "hash_5_shift",
    "hash_5_const",
    "hash_5_join",
)
_FINAL_HASH_FULL_TAGS = (
    "paired_branch_select",
    "paired_branch_node_xor",
    "hash_0",
    "hash_1_shift_vector",
    "hash_1_const",
    "hash_1_join",
) + _FINAL_HASH_TAIL_TAGS
OP_PRIORITY_OFFSETS: dict[tuple[str, int, int], int] = {
    **{
        (tag, group, 15): 20_000
        for tag in _FINAL_HASH_TAIL_TAGS
        for group in (27, 28, 29, 30)
    },
    **{(tag, 31, 15): 1_000 for tag in _FINAL_HASH_FULL_TAGS},
    **{
        (tag, group, 15): 50_000
        for tag in ("tree_gather", "node_xor")
        for group in (24, 25, 26, 27, 28, 30)
    },
}
GROUP_FINE_OFFSETS = (0, 0, 0, 0, 0, 0, 0, 1) + (0,) * 24
BASIC_GROUP_OFFSETS = (0,) * N_GROUPS
BASIC_ROUND_OFFSETS = (0,) * 16
BASIC_HEIGHT_DIVISOR = 2
BASIC_REACH_DIVISOR = 32
VECTOR_NODE_XOR_SET: frozenset[tuple[int, int]] = frozenset()
VECTOR_DYNAMIC_XOR_SET: frozenset[tuple[int, int]] = frozenset()
SCALAR_DYNAMIC_XOR_SET = frozenset(
    (group, rnd) for group in range(7) for rnd in range(4, 16)
)


def _words(base: int, size: int = VLEN) -> tuple[int, ...]:
    return tuple(range(base, base + size))


@dataclass(slots=True)
class _Op:
    engine: str
    slot: tuple
    reads: tuple[int, ...]
    writes: tuple[int, ...]
    parents: dict[int, int] = field(default_factory=dict)
    tag: str = ""
    group: int = -1
    round: int = -1


class _Scratch:
    def __init__(self) -> None:
        self.ptr = 0
        self.debug: dict[int, tuple[str, int]] = {}

    def alloc(self, name: str, size: int = 1) -> int:
        base = self.ptr
        self.ptr += size
        if self.ptr > SCRATCH_SIZE:
            raise AssertionError(
                f"scratch overflow: {self.ptr} > {SCRATCH_SIZE} while allocating {name}"
            )
        self.debug[base] = (name, size)
        return base


class _Graph:
    """Build exact scratch dependencies for end-of-cycle writes.

    RAW and WAW require a later cycle.  WAR only requires program order: the
    reader and following writer may share a bundle because every engine reads
    the old scratch image and commits writes at the end of the cycle.
    """

    def __init__(self) -> None:
        self.ops: list[_Op] = []
        self.last_writer: dict[int, int] = {}
        self.readers: dict[int, list[int]] = defaultdict(list)

    @staticmethod
    def _add_parent(parents: dict[int, int], parent: int, lag: int) -> None:
        if parent < 0:
            return
        old = parents.get(parent)
        if old is None or lag > old:
            parents[parent] = lag

    def emit(
        self,
        engine: str,
        slot: tuple,
        reads: Iterable[int] = (),
        writes: Iterable[int] = (),
        *,
        deps: Iterable[tuple[int, int]] = (),
        tag: str = "",
        group: int = -1,
        round: int = -1,
    ) -> int:
        reads_t = tuple(dict.fromkeys(reads))
        writes_t = tuple(dict.fromkeys(writes))
        idx = len(self.ops)
        parents: dict[int, int] = {}

        for parent, lag in deps:
            self._add_parent(parents, parent, lag)

        # Register all reads before writes so an in-place instruction depends
        # on the preceding writer but never on itself.
        for addr in reads_t:
            writer = self.last_writer.get(addr)
            if writer is not None:
                self._add_parent(parents, writer, 1)
            self.readers[addr].append(idx)

        for addr in writes_t:
            writer = self.last_writer.get(addr)
            if writer is not None:
                self._add_parent(parents, writer, 1)
            for reader in self.readers.get(addr, ()):
                if reader != idx:
                    self._add_parent(parents, reader, 0)

        self.ops.append(
            _Op(
                engine=engine,
                slot=slot,
                reads=reads_t,
                writes=writes_t,
                parents=parents,
                tag=tag,
                group=group,
                round=round,
            )
        )

        for addr in writes_t:
            self.last_writer[addr] = idx
            self.readers[addr] = []
        return idx


class KernelBuilder:
    def __init__(self) -> None:
        self.instrs: list[dict[str, list[tuple]]] = []
        self.scratch: dict[str, int] = {}
        self.scratch_debug: dict[int, tuple[str, int]] = {}
        self.scratch_ptr = 0

    def debug_info(self) -> DebugInfo:
        return DebugInfo(scratch_map=self.scratch_debug)

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ) -> None:
        shape = (forest_height, n_nodes, batch_size, rounds)
        if shape != OFFICIAL_SHAPE:
            raise ValueError(
                "the extreme kernel is intentionally specialized for the scored "
                f"shape {OFFICIAL_SHAPE}, got {shape}"
            )

        scratch = _Scratch()
        graph = _Graph()

        def alloc(name: str, size: int = 1) -> int:
            addr = scratch.alloc(name, size)
            self.scratch[name] = addr
            return addr

        # Scalar constants are deduplicated.  Pointer registers are separate
        # because streaming loads/stores update them in place.
        scalar_constants: dict[int, int] = {}

        def scalar_const(value: int, name: str) -> int:
            value &= 0xFFFFFFFF
            if value not in scalar_constants:
                scalar_constants[value] = alloc(name)
            return scalar_constants[value]

        s_one = scalar_const(1, "one")
        s_two = scalar_const(2, "two")
        if not OVERLAP_DEEP_ADDRESS:
            s_four = scalar_const(4, "four")
            s_eight = scalar_const(8, "eight")
        else:
            s_four = s_eight = -1
        s_sixteen = scalar_const(16, "sixteen")
        s_nineteen = scalar_const(19, "nineteen")
        s_c0 = scalar_const(C0, "hash_c0")
        s_c1 = scalar_const(C1, "hash_c1")
        s_c23 = scalar_const(C23, "hash_c2_plus_c3")
        s_m23 = scalar_const(M23, "hash_mul2_shift9")
        s_c2_shift9 = scalar_const(C2_SHIFT9, "hash_c2_shift9")
        s_c4 = scalar_const(C4, "hash_c4")
        s_c5 = scalar_const(C5, "hash_c5")
        s_m4 = scalar_const(9, "hash_mul4_shift3")
        s_m0 = scalar_const(4097, "hash_mul0")
        s_m2 = scalar_const(33, "hash_mul2")

        # Mirrored local-path address: mem_addr = 2**(depth+1) + 5 - mirror.
        depth_range = range(4, 5) if OVERLAP_DEEP_ADDRESS else range(4, 11)
        depth_base = {
            d: scalar_const((1 << (d + 1)) + 5, f"depth_{d}_base")
            for d in depth_range
        }

        # These constants are dead after their vector broadcasts.  Reusing
        # their scalar words for the three streaming pointer pairs saves six
        # registers without extending a hot live range.
        top_p0 = s_c0
        top_p1 = s_c23
        raw_root_scalar = s_c23
        io_p0, io_p1 = s_m0, s_m2

        # Vector constants needed by fixed VALU instructions.
        vector_constants: dict[int, int] = {}

        def vector_const(value: int, name: str) -> int:
            value &= 0xFFFFFFFF
            if value not in vector_constants:
                vector_constants[value] = alloc(name, VLEN)
            return vector_constants[value]

        v_one = vector_const(1, "v_one")
        v_two = vector_const(2, "v_two")
        if OVERLAP_DEEP_ADDRESS:
            v_four = alloc("v_four", VLEN)
            v_eight = alloc("v_eight", VLEN)
        else:
            v_four = vector_const(4, "v_four")
            v_eight = vector_const(8, "v_eight")
        v_nineteen = vector_const(19, "v_nineteen")
        v_sixteen = vector_const(16, "v_sixteen")
        v_c0 = vector_const(C0, "v_hash_c0")
        v_c1 = vector_const(C1, "v_hash_c1")
        v_c23 = vector_const(C23, "v_hash_c2_plus_c3")
        v_m23 = vector_const(M23, "v_hash_mul2_shift9")
        v_c2_shift9 = vector_const(C2_SHIFT9, "v_hash_c2_shift9")
        v_c4 = vector_const(C4, "v_hash_c4")
        # In the reversed/relocated design all C5 users are scalarized.  Reuse
        # its eight words for the original-layout address bias needed only
        # after the relocated depth-8 level drains.
        v_c5 = (
            alloc("v_address_negative_five", VLEN)
            if REVERSED_RELOCATED_TREE
            else vector_const(C5, "v_hash_c5")
        )
        v_m0 = vector_const(4097, "v_hash_mul0")
        v_m2 = vector_const(33, "v_hash_mul2")
        v_m4 = vector_const(9, "v_hash_mul4_shift3")
        v_neg_five = (
            alloc("v_negative_five", VLEN) if OVERLAP_DEEP_ADDRESS else -1
        )

        top_words = alloc("top_tree_words", 32)
        node_vec = [alloc(f"cached_node_{i}", VLEN) for i in range(31)]
        depth1_diff = -1

        # Every SIMD group keeps only its long-lived value/mirror/temp state.
        values = [alloc(f"value_{g}", VLEN) for g in range(N_GROUPS)]
        mirrors = [alloc(f"mirror_{g}", VLEN) for g in range(N_GROUPS)]
        temps = [alloc(f"temp_{g}", VLEN) for g in range(N_GROUPS)]
        if INDEPENDENT_INPUT_POINTERS:
            io_p0, io_p1 = temps[26], temps[27]
        paired_candidate_yes = mirrors[0]
        paired_candidate_no = temps[0]
        paired_base_registers = [
            (mirrors[0] if DELAYED_PAIR_BRANCH_GROUPS else mirrors[1]) + pair
            for pair in range(4)
        ]
        paired_jump_registers = [
            (temps[0] if DELAYED_PAIR_BRANCH_GROUPS else temps[1]) + pair
            for pair in range(4)
        ]
        # The last group is the critical tail.  During its second traversal,
        # keep the three hash parity bits in registers that are already dead
        # at the tail instead of rebuilding them from the packed mirror with
        # five VALU ``&`` instructions.  The first two vectors subsequently
        # become the paired-dispatch candidates; the third is dead group-2
        # mirror storage (group 1 is reserved for dispatch bases).
        saved_second_path_bits = (
            {
                BRANCH_FINAL_GROUP: (
                    paired_candidate_yes,
                    paired_candidate_no,
                    mirrors[2],
                )
            }
            if PAIRED_BRANCH_FINAL
            else {}
        )
        saved_second_path_mux = (
            {
                BRANCH_FINAL_GROUP: (
                    temps[2],
                    mirrors[3],
                    temps[3],
                    mirrors[4],
                )
            }
            if PAIRED_BRANCH_FINAL
            else {}
        )

        # Shallow path bits and mux spills are shared by seven software-
        # pipelined workspaces.  They are released during depths 4..10.
        bits = []
        for workspace in range(N_WORKSPACES):
            workspace_bits = []
            for bit in range(3):
                addr = alloc(f"workspace_bit_{workspace}_{bit}", VLEN)
                workspace_bits.append(addr)
            bits.append(workspace_bits)

        # A depth-first mux order needs one spill vector rather than two.
        select_spill = [
            [alloc(f"select_spill_{workspace}", VLEN)]
            for workspace in range(N_SPILL_WORKSPACES)
        ]
        if INDEPENDENT_TOP_P0:
            # This vector is setup staging storage and is otherwise dead until
            # round 2.  Its first word can hold the first top-tree pointer;
            # relocation later overwrites the full vector after all top loads
            # and the -5 broadcast have consumed it.
            top_p0 = bits[0][2]
        if INDEPENDENT_TOP_P1:
            top_p1 = select_spill[0][0]
        branch_table_base = (
            alloc("branch_table_base")
            if BRANCH_FINAL_LANES or DELAYED_PAIR_BRANCH_GROUPS
            else -1
        )
        physical_workspace_vectors = tuple(
            addr for workspace_bits in bits for addr in workspace_bits
        ) + tuple(workspace_spill[0] for workspace_spill in select_spill)
        # Mirrors and hash temporaries become dead as their owning software
        # pipeline group drains.  They are valid register-allocation colors
        # for later virtual saved paths, provided the exact scheduled live
        # intervals do not overlap.  Keep the low registers used by the
        # paired final dispatch and group 31's bespoke saved path reserved.
        physical_color_vectors = (
            physical_workspace_vectors
            + tuple(mirrors[5:31])
            + tuple(temps[4:31])
        )

        # Virtual addresses deliberately live far outside the architectural
        # scratch range.  They only exist while constructing and scheduling
        # the DAG; ``color_virtual_workspaces`` below rewrites every one of
        # them to an allocated physical vector before validation/execution.
        virtual_next = 1_000_000
        virtual_vectors: dict[int, str] = {}
        virtual_workspaces: dict[tuple[str, int] | tuple[int, int], tuple[list[int], list[int]]] = {}

        def virtual_vector(name: str) -> int:
            nonlocal virtual_next
            base = virtual_next
            virtual_next += VLEN
            virtual_vectors[base] = name
            return base

        if MADD_FIRST_DEPTH1 or MADD_FIRST_DEPTH1_SET:
            # This pair difference is live only through selected first-
            # traversal depth-1 lookups.  Give it an SSA name and let the
            # scheduled live-range colorer reuse a not-yet-live group vector;
            # reserving another architectural vector would exceed scratch.
            depth1_diff = virtual_vector("depth1_pair_diff")

        valu_final_cache_scratch = {
            group: virtual_vector(f"valu_final_cache_scratch_{group}")
            for group in VALU_FINAL_CACHE_SET | VALU_FINAL_CACHE_COUNTS.keys()
        }
        ssa_level4_pools: dict[tuple[int, int], int] = {}
        ssa_level4_conditions: dict[tuple[int, int], int] = {}

        # Extra saved second paths are long-lived across four hash rounds.
        # Giving them unique SSA names is essential: assigning them directly
        # to the rotating workspace pool lets intervening groups overwrite a
        # parity bit before its mux consumes it.  After scheduling, the same
        # linear-scan colorer used by experimental SSA workspaces maps these
        # names back onto the existing 36 physical vectors, so this consumes
        # no additional architectural scratch.
        for group in SAVED_SECOND_PATH_EXTRA_GROUPS:
            saved_bit_count = 4 if group in FINAL_CACHE_SET else 3
            reusable_mirror = (
                mirrors[group]
                if group in FINAL_CACHE_SET
                and 5 <= group < BRANCH_FINAL_GROUP
                else virtual_vector(f"saved_second_path_bit_{group}_0")
            )
            saved_second_path_bits[group] = (reusable_mirror,) + tuple(
                virtual_vector(f"saved_second_path_bit_{group}_{bit}")
                for bit in range(1, saved_bit_count)
            )
            saved_second_path_mux[group] = tuple(
                virtual_vector(f"saved_second_path_mux_{group}_{mux}")
                for mux in range(4)
            )

        def workspace_registers(
            workspace: int, gg: int, rnd: int
        ) -> tuple[list[int], list[int]]:
            if SSA_ALL_WORKSPACES or (
                rnd < 11 and gg in SSA_FIRST_WORKSPACE_GROUPS
            ):
                # Bits p0..p2 must share one name across rounds 0..4 of a
                # first traversal.  Later cached selections derive conditions
                # afresh, so each (round, group) receives a short interval.
                key: tuple[str, int] | tuple[int, int]
                key = ("first", gg) if rnd < 11 else (rnd, gg)
                registers = virtual_workspaces.get(key)
                if registers is None:
                    key_name = f"{key[0]}_{key[1]}"
                    registers = (
                        [
                            virtual_vector(f"ssa_bit_{key_name}_{bit}")
                            for bit in range(3)
                        ],
                        [virtual_vector(f"ssa_spill_{key_name}")],
                    )
                    virtual_workspaces[key] = registers
                return registers
            # The first traversal intentionally uses rotating physical
            # workspaces: its saved path bits span several rounds and fully
            # renaming them exceeds the 36-vector register budget.  The second
            # traversal's conditions are short-lived, so it is both safe and
            # profitable to allocate those after scheduling.
            if not SSA_WORKSPACES or rnd < rounds - 1:
                return bits[workspace], select_spill[workspace % N_SPILL_WORKSPACES]

            # Later cached selections derive their conditions directly from
            # the mirror and therefore have independent live ranges.
            key: tuple[str, int] | tuple[int, int]
            key = (rnd, gg)
            registers = virtual_workspaces.get(key)
            if registers is None:
                key_name = f"{key[0]}_{key[1]}"
                registers = (
                    # Keep path conditions on the rotating physical registers:
                    # the list scheduler may otherwise hoist them hundreds of
                    # cycles before the mux and create impossible live ranges.
                    bits[workspace],
                    [virtual_vector(f"virtual_spill_{key_name}")],
                )
                virtual_workspaces[key] = registers
            return registers
        # Preprocessing borrows the two latest groups' temp vectors.  Keeping
        # these separate from the persistent level-4 mux workspace prevents a
        # global setup barrier; only groups 30/31 wait for preprocessing.
        preprocess_buffers = [temps[30], temps[31]]
        level4_pool = top_words + 2 * VLEN
        level4_condition = top_words + 3 * VLEN
        # The lower staging words become persistent pair differences after
        # their node broadcasts, avoiding any extra scratch cost.
        level4_diff = [top_words, top_words + VLEN] + [
            alloc(f"level4_pair_diff_{i}", VLEN)
            # The paired indirect-branch table needs all eight differences
            # even when some ordinary cached lookups use vselect instead of
            # MADD for their bottom pairs.
            for i in range(2, 8)
        ]
        first_level4_pool = [level4_pool]
        first_level4_condition = level4_condition
        final_address_vectors_ready = [False]

        if scratch.ptr > SCRATCH_SIZE:
            raise AssertionError(f"scratch overflow: {scratch.ptr}")

        def emit_const(dest: int, value: int, tag: str = "const") -> int:
            return graph.emit("load", ("const", dest, value), writes=(dest,), tag=tag)

        def emit_immediate(dest: int, value: int, tag: str) -> int:
            """Materialize a scalar on the otherwise underused flow engine."""
            if tag in LOAD_IMMEDIATE_TAGS:
                return emit_const(dest, value, tag)
            return graph.emit(
                "flow",
                ("add_imm", dest, s_one, (value - 1) & 0xFFFFFFFF),
                reads=(s_one,),
                writes=(dest,),
                tag=tag,
            )

        def emit_vbroadcast(dest: int, src: int, tag: str = "broadcast") -> int:
            return graph.emit(
                "valu",
                ("vbroadcast", dest, src),
                reads=(src,),
                writes=_words(dest),
                tag=tag,
            )

        def emit_valu(
            op: str,
            dest: int,
            a: int,
            b: int,
            *,
            tag: str,
            group: int = -1,
            round: int = -1,
        ) -> int:
            return graph.emit(
                "valu",
                (op, dest, a, b),
                reads=_words(a) + _words(b),
                writes=_words(dest),
                tag=tag,
                group=group,
                round=round,
            )

        def emit_madd(
            dest: int,
            a: int,
            b: int,
            c: int,
            *,
            tag: str,
            group: int = -1,
            round: int = -1,
        ) -> int:
            return graph.emit(
                "valu",
                ("multiply_add", dest, a, b, c),
                reads=_words(a) + _words(b) + _words(c),
                writes=_words(dest),
                tag=tag,
                group=group,
                round=round,
            )

        def emit_scalarized(
            op: str,
            dest: int,
            a: int,
            b: int,
            *,
            a_scalar: bool = False,
            b_scalar: bool = False,
            tag: str,
            group: int = -1,
            round: int = -1,
        ) -> list[int]:
            ids = []
            for lane in range(VLEN):
                aa = a if a_scalar else a + lane
                bb = b if b_scalar else b + lane
                ids.append(
                    graph.emit(
                        "alu",
                        (op, dest + lane, aa, bb),
                        reads=(aa, bb),
                        writes=(dest + lane,),
                        tag=tag,
                        group=group,
                        round=round,
                    )
                )
            return ids

        def emit_vbasic(
            op: str,
            dest: int,
            a: int,
            b_vector: int,
            *,
            scalarize: bool,
            scalar_b: int,
            tag: str,
            group: int,
            round: int,
        ) -> None:
            if scalarize:
                emit_scalarized(
                    op,
                    dest,
                    a,
                    scalar_b,
                    b_scalar=True,
                    tag=tag,
                    group=group,
                    round=round,
                )
            else:
                emit_valu(
                    op,
                    dest,
                    a,
                    b_vector,
                    tag=tag,
                    group=group,
                    round=round,
                )

        def emit_vselect(
            dest: int,
            cond: int,
            yes: int,
            no: int,
            *,
            group: int,
            round: int,
        ) -> int:
            return graph.emit(
                "flow",
                ("vselect", dest, cond, yes, no),
                reads=_words(cond) + _words(yes) + _words(no),
                writes=_words(dest),
                tag="tree_select",
                group=group,
                round=round,
            )

        # Load all immutable scalar constants, then broadcast the subset used
        # by VALU.  The scheduler overlaps these with tree and input traffic.
        if FLOW_ONE_CONSTANT:
            graph.emit(
                "flow",
                ("add_imm", s_one, s_one, 1),
                reads=(s_one,),
                writes=(s_one,),
                tag="one_immediate",
            )
        else:
            emit_const(s_one, 1)
        misc_addresses = {
            "nineteen": s_nineteen,
            "m4": s_m4,
            "m0": s_m0,
            "m2": s_m2,
            "m23": s_m23,
        }
        derived_misc_addresses = {
            misc_addresses[name] for name in ALU_DERIVED_MISC_SET
        }
        derived_depth_addresses = {
            addr for depth, addr in depth_base.items()
            if depth >= ALU_DERIVED_DEPTH_START
        }
        for constant_index, (value, addr) in enumerate(scalar_constants.items()):
            if addr != s_one:
                if addr in derived_depth_addresses or addr in derived_misc_addresses:
                    continue
                elif (
                    value in (2, 4, 8, 16)
                    and value.bit_length() - 1 <= ALU_DERIVED_POWER_COUNT
                ):
                    half = scalar_constants[value // 2]
                    graph.emit(
                        "alu",
                        ("+", addr, half, half),
                        reads=(half,),
                        writes=(addr,),
                        tag="scalar_power_derive",
                    )
                elif (
                    constant_index < FLOW_SCALAR_CONSTANT_COUNT
                    or constant_index in FLOW_SCALAR_CONSTANT_SET
                ):
                    if constant_index in FLOW_ZERO_BASE_CONSTANT_SET:
                        graph.emit(
                            "flow",
                            ("add_imm", addr, addr, value),
                            reads=(addr,),
                            writes=(addr,),
                            tag="scalar_zero_immediate",
                        )
                    else:
                        emit_immediate(addr, value, "scalar_immediate")
                else:
                    emit_const(addr, value, "scalar_constant")
        if "nineteen" in ALU_DERIVED_MISC_SET:
            graph.emit(
                "alu", ("+", s_nineteen, s_sixteen, s_two),
                reads=(s_sixteen, s_two), writes=(s_nineteen,),
                tag="scalar_misc_derive",
            )
            graph.emit(
                "alu", ("+", s_nineteen, s_nineteen, s_one),
                reads=(s_nineteen, s_one), writes=(s_nineteen,),
                tag="scalar_misc_derive",
            )
        if "m4" in ALU_DERIVED_MISC_SET:
            graph.emit(
                "alu", ("+", s_m4, s_eight, s_one),
                reads=(s_eight, s_one), writes=(s_m4,),
                tag="scalar_misc_derive",
            )
        if "m0" in ALU_DERIVED_MISC_SET:
            graph.emit(
                "alu", ("*", s_m0, s_sixteen, s_sixteen),
                reads=(s_sixteen,), writes=(s_m0,), tag="scalar_misc_derive",
            )
            graph.emit(
                "alu", ("*", s_m0, s_m0, s_sixteen),
                reads=(s_m0, s_sixteen), writes=(s_m0,), tag="scalar_misc_derive",
            )
            graph.emit(
                "alu", ("+", s_m0, s_m0, s_one),
                reads=(s_m0, s_one), writes=(s_m0,), tag="scalar_misc_derive",
            )
        if "m2" in ALU_DERIVED_MISC_SET:
            graph.emit(
                "alu", ("*", s_m2, s_sixteen, s_two),
                reads=(s_sixteen, s_two), writes=(s_m2,), tag="scalar_misc_derive",
            )
            graph.emit(
                "alu", ("+", s_m2, s_m2, s_one),
                reads=(s_m2, s_one), writes=(s_m2,), tag="scalar_misc_derive",
            )
        if "m23" in ALU_DERIVED_MISC_SET:
            graph.emit(
                "alu", ("<<", s_m23, s_m2, s_m4),
                reads=(s_m2, s_m4), writes=(s_m23,), tag="scalar_misc_derive",
            )
        if derived_depth_addresses:
            previous = s_sixteen
            for depth in depth_range:
                if depth < ALU_DERIVED_DEPTH_START:
                    previous = depth_base[depth]
                    continue
                dest = depth_base[depth]
                graph.emit(
                    "alu", ("+", dest, previous, previous),
                    reads=(previous,), writes=(dest,), tag="depth_base_double",
                )
                if depth == 4:
                    graph.emit(
                        "alu", ("+", dest, dest, scalar_constants[4]),
                        reads=(dest, scalar_constants[4]), writes=(dest,),
                        tag="depth_base_adjust",
                    )
                    graph.emit(
                        "alu", ("+", dest, dest, s_one),
                        reads=(dest, s_one), writes=(dest,),
                        tag="depth_base_adjust",
                    )
                else:
                    graph.emit(
                        "alu", ("-", dest, dest, scalar_constants[4]),
                        reads=(dest, scalar_constants[4]), writes=(dest,),
                        tag="depth_base_adjust",
                    )
                    graph.emit(
                        "alu", ("-", dest, dest, s_one),
                        reads=(dest, s_one), writes=(dest,),
                        tag="depth_base_adjust",
                    )
                previous = dest
        for value, dest in vector_constants.items():
            emit_vbroadcast(dest, scalar_constants[value], "constant_broadcast")
        if OVERLAP_DEEP_ADDRESS:
            emit_valu("+", v_four, v_two, v_two, tag="derive_v_four")
            emit_valu("+", v_eight, v_four, v_four, tag="derive_v_eight")
            # This scalar register is overwritten by the top-tree pointer
            # before any later use.  Reusing it avoids a permanent constant.
            emit_immediate(top_p0, -5, "negative_five_immediate")
            emit_vbroadcast(v_neg_five, top_p0, "negative_five_broadcast")
        elif REVERSED_RELOCATED_TREE:
            if VECTOR_TOP_C5_BLOCKS:
                if not 0 <= VECTOR_TOP_C5_BLOCKS <= 4:
                    raise ValueError("vector top C5 blocks must be in 0..4")
                emit_vbroadcast(v_c5, s_c5, "setup_c5_broadcast")
            else:
                emit_immediate(top_p0, -5, "negative_five_immediate")
                emit_vbroadcast(v_c5, top_p0, "negative_five_broadcast")

        # Fetch nodes 0..31 using two rolling pointers and four vector loads.
        if "top_pointer" in LOAD_IMMEDIATE_TAGS:
            emit_const(top_p0, 7, "top_pointer")
        elif INDEPENDENT_TOP_P0:
            graph.emit(
                "flow",
                ("add_imm", top_p0, top_p0, 7),
                reads=(top_p0,),
                writes=(top_p0,),
                tag="top_pointer",
            )
        else:
            emit_immediate(top_p0, 7, "top_pointer")
        if "top_pointer" in LOAD_IMMEDIATE_TAGS:
            emit_const(top_p1, 15, "top_pointer")
        elif INDEPENDENT_TOP_P1:
            if DERIVE_TOP_P1_FROM_P0:
                graph.emit(
                    "alu",
                    ("+", top_p1, top_p0, s_eight),
                    reads=(top_p0, s_eight),
                    writes=(top_p1,),
                    tag="top_pointer_derive",
                )
            else:
                graph.emit(
                    "flow",
                    ("add_imm", top_p1, top_p1, 15),
                    reads=(top_p1,),
                    writes=(top_p1,),
                    tag="top_pointer",
                )
        else:
            emit_immediate(top_p1, 15, "top_pointer")
        top_loads: list[int] = []
        for pair in range(2):
            top_loads.append(
                graph.emit(
                    "load",
                    ("vload", top_words + pair * 16, top_p0),
                    reads=(top_p0,),
                    writes=_words(top_words + pair * 16),
                    tag="top_tree_load",
                )
            )
            top_loads.append(
                graph.emit(
                    "load",
                    ("vload", top_words + pair * 16 + 8, top_p1),
                    reads=(top_p1,),
                    writes=_words(top_words + pair * 16 + 8),
                    tag="top_tree_load",
                )
            )
            if pair == 0:
                graph.emit(
                    "alu",
                    ("+", top_p0, top_p0, s_sixteen),
                    reads=(top_p0, s_sixteen),
                    writes=(top_p0,),
                    tag="pointer_advance",
                )
                graph.emit(
                    "alu",
                    ("+", top_p1, top_p1, s_sixteen),
                    reads=(top_p1, s_sixteen),
                    writes=(top_p1,),
                    tag="pointer_advance",
                )

        # The direct-path variant preserves one raw scalar root and broadcasts
        # its transformed form.  This removes the global round-0/round-11 root
        # barrier and lets all sixteen rounds form one software wavefront.
        if INDEPENDENT_ROOT_CACHE or DIRECT_MIRROR_PATH:
            graph.emit(
                "flow",
                ("add_imm", raw_root_scalar, top_words, 0),
                reads=(top_words,),
                writes=(raw_root_scalar,),
                tag="raw_root_copy",
            )
            if REVERSED_RELOCATED_TREE:
                for chunk in range(4):
                    if chunk < VECTOR_TOP_C5_BLOCKS:
                        emit_valu(
                            "^",
                            top_words + chunk * VLEN,
                            top_words + chunk * VLEN,
                            v_c5,
                            tag="cached_nodes_vector_transform",
                        )
                    else:
                        for lane in range(VLEN):
                            word = top_words + chunk * VLEN + lane
                            graph.emit(
                                "alu",
                                ("^", word, word, s_c5),
                                reads=(word, s_c5),
                                writes=(word,),
                                tag="cached_nodes_scalar_transform",
                            )
                if VECTOR_TOP_C5_BLOCKS:
                    emit_immediate(top_p0, -5, "negative_five_immediate")
                    emit_vbroadcast(v_c5, top_p0, "negative_five_broadcast")
            else:
                for chunk in range(4):
                    emit_valu(
                        "^",
                        top_words + chunk * VLEN,
                        top_words + chunk * VLEN,
                        v_c5,
                        tag="cached_nodes_vector_transform",
                    )
            for i in range(31):
                emit_vbroadcast(
                    node_vec[i], top_words + i, "cached_node_broadcast"
                )
        else:
            # Node vector 0 starts as the raw root for round 0.  It is
            # transformed only after every first-root reader is emitted.
            emit_vbroadcast(node_vec[0], top_words, "raw_root_broadcast")
            for i in range(1, 31):
                graph.emit(
                    "alu",
                    ("^", top_words + i, top_words + i, s_c5),
                    reads=(top_words + i, s_c5),
                    writes=(top_words + i,),
                    tag="cached_node_transform",
                )
                emit_vbroadcast(node_vec[i], top_words + i, "cached_node_broadcast")

        # Optional one-time tree preprocessing experiment.
        # every SIMD group.  Transform them once in private machine memory so
        # each later gather directly receives node^C5.  The two staging
        # buffers and pointer scalars are setup-only storage that is reused
        # below, so this costs no persistent scratch.
        preprocess_p0, preprocess_p1 = s_c4, s_m4
        if INDEPENDENT_RELOCATION_LOAD_POINTERS:
            preprocess_p0, preprocess_p1 = temps[28], temps[29]
        preprocess_stores: list[int] = []
        preprocess_pairs = {0: 0, 4: 1, 5: 3, 6: 7, 7: 15, 8: 31}[
            PREPROCESS_MAX_DEPTH
        ]
        if REVERSED_RELOCATED_TREE:
            if PREPROCESS_MAX_DEPTH != 8:
                raise ValueError(
                    "reversed relocated tree requires PREPROCESS_MAX_DEPTH=8"
                )
            if SCALAR_FINAL_HASH4_SET or SCALAR_FIRST_DEPTH1_SET:
                raise ValueError(
                    "reversed relocated tree owns the dead scalar setup registers"
                )

            # Destination blocks are consecutive heap levels beginning at
            # address 16.  Within each level, consume staged source blocks in
            # reverse order and reverse their eight scalar lanes while XORing
            # C5.  Compute that consumption order before assigning staging
            # registers: a register becomes available to its owning SIMD
            # group immediately after the corresponding source is consumed.
            source_order: list[int] = []
            source_start = 0
            for depth in range(4, 9):
                vector_count = 1 << (depth - 3)
                source_order.extend(
                    range(
                        source_start + vector_count - 1,
                        source_start - 1,
                        -1,
                    )
                )
                source_start += vector_count
            if source_start != 62:
                raise AssertionError(source_start)

            # Stage every depth-4..8 vector before writing any relocated node.
            # The destination ranges overlap the original ranges, so this
            # global read-before-write phase is required for correctness.  All
            # mirror registers and temp[0:30] are dead at setup time and give
            # exactly the 62 vector colors required, at no scratch cost.
            launch_order = sorted(
                range(N_GROUPS),
                key=lambda group: (FULL_ROUND_OFFSETS[group], group),
            )
            if RELOCATION_STAGE_ORDER == "linear":
                release_vectors = list(mirrors) + list(temps[:30])
            elif RELOCATION_STAGE_ORDER == "temp_first":
                release_vectors = [
                    temps[group] for group in launch_order if group < 30
                ] + [mirrors[group] for group in launch_order]
            elif RELOCATION_STAGE_ORDER == "interleave":
                release_vectors = []
                for group in launch_order:
                    if group < 30:
                        release_vectors.append(temps[group])
                    release_vectors.append(mirrors[group])
            elif RELOCATION_STAGE_ORDER.startswith("late_state"):
                # Keep every value, hash temporary, and round-0/1 condition
                # register free.  These 62 colors are semantically dead until
                # round 2 or later: eight round-2 condition/spill pairs, all
                # mirrors (first written after round 3), the ten level-4 mux
                # vectors, and workspace 8 which is reserved for the second
                # traversal.  This removes the large forced VALU startup
                # bubble caused by staging into per-group temporaries.
                active_workspaces = sorted(
                    range(N_WORKSPACES - 1),
                    key=lambda workspace: min(
                        FULL_ROUND_OFFSETS[group]
                        for group in range(N_GROUPS)
                        if WORKSPACE_ASSIGNMENT[group] == workspace
                    ),
                )
                round2_vectors = [
                    vector
                    for workspace in active_workspaces
                    for vector in (bits[workspace][2], select_spill[workspace][0])
                ]
                level4_vectors = list(level4_diff) + [
                    level4_pool,
                    level4_condition,
                ]
                second_workspace_vectors = list(bits[N_WORKSPACES - 1]) + [
                    select_spill[N_WORKSPACES - 1][0]
                ]
                ordered_mirrors = [mirrors[group] for group in launch_order]
                if RELOCATION_STAGE_ORDER == "late_state":
                    release_vectors = (
                        round2_vectors
                        + ordered_mirrors[:16]
                        + level4_vectors
                        + ordered_mirrors[16:]
                        + second_workspace_vectors
                    )
                elif RELOCATION_STAGE_ORDER == "late_state_g0_early":
                    # mirror[0] is the sole staging color on the Hall-critical
                    # path to group 0's round-4 gather.  Release it one copy
                    # earlier by exchanging it with the last round-2 vector,
                    # which belongs to a much later-starting workspace.
                    release_vectors = (
                        round2_vectors[:15]
                        + [mirrors[0]]
                        + round2_vectors[15:]
                        + [vector for vector in ordered_mirrors[:16] if vector != mirrors[0]]
                        + level4_vectors
                        + ordered_mirrors[16:]
                        + second_workspace_vectors
                    )
                elif RELOCATION_STAGE_ORDER == "late_state_mirror_first":
                    release_vectors = (
                        ordered_mirrors
                        + round2_vectors
                        + level4_vectors
                        + second_workspace_vectors
                    )
                elif RELOCATION_STAGE_ORDER == "late_state_level4_early":
                    release_vectors = (
                        round2_vectors
                        + ordered_mirrors[:8]
                        + level4_vectors
                        + ordered_mirrors[8:]
                        + second_workspace_vectors
                    )
                else:
                    raise ValueError(
                        f"unknown relocation stage order: {RELOCATION_STAGE_ORDER}"
                    )
            else:
                raise ValueError(
                    f"unknown relocation stage order: {RELOCATION_STAGE_ORDER}"
                )
            stage_vectors = [-1] * 62
            for output_index, source_index in enumerate(source_order):
                stage_vectors[source_index] = release_vectors[output_index]
            if len(stage_vectors) != 62:
                raise AssertionError(len(stage_vectors))
            if len(set(stage_vectors)) != 62 or -1 in stage_vectors:
                raise AssertionError("invalid relocation staging assignment")
            stage_loads: list[int] = []
            stage_load_for_source: dict[int, int] = {}
            first_stage_pair = 1 if REUSE_TOP_RELOCATION_LEVEL4 else 0
            base_load_pointers = (preprocess_p0, preprocess_p1)
            base_load_initials = (
                22 + first_stage_pair * 16,
                30 + first_stage_pair * 16,
            )
            for pointer_index, (pointer, initial) in enumerate(zip(
                base_load_pointers,
                base_load_initials,
            )):
                if DERIVE_SETUP_SECOND_POINTERS and pointer_index == 1:
                    graph.emit(
                        "alu",
                        ("+", pointer, preprocess_p0, s_eight),
                        reads=(preprocess_p0, s_eight),
                        writes=(pointer,),
                        tag="preprocess_pointer_derive",
                    )
                    continue
                if "preprocess_pointer" in LOAD_IMMEDIATE_TAGS:
                    emit_const(pointer, initial, "preprocess_pointer")
                elif INDEPENDENT_RELOCATION_LOAD_POINTERS:
                    graph.emit(
                        "flow",
                        ("add_imm", pointer, pointer, initial),
                        reads=(pointer,),
                        writes=(pointer,),
                        tag="preprocess_pointer",
                    )
                else:
                    emit_immediate(pointer, initial, "preprocess_pointer")

            if RELOCATION_LOAD_STREAMS == 2:
                for pair_index in range(first_stage_pair, 31):
                    for lane_index, (pointer, buffer) in enumerate(zip(
                            base_load_pointers,
                            stage_vectors[2 * pair_index : 2 * pair_index + 2],
                    )):
                        load_id = graph.emit(
                            "load",
                            ("vload", buffer, pointer),
                            reads=(pointer,),
                            writes=_words(buffer),
                            tag="tree_relocation_stage_load",
                        )
                        stage_loads.append(load_id)
                        stage_load_for_source[
                            2 * pair_index + lane_index
                        ] = load_id
                    if pair_index != 30:
                        for pointer in base_load_pointers:
                            graph.emit(
                                "alu",
                                ("+", pointer, pointer, s_sixteen),
                                reads=(pointer, s_sixteen),
                                writes=(pointer,),
                                tag="pointer_advance",
                            )
            elif RELOCATION_LOAD_STREAMS == 4:
                # Four scalar streams sustain two vector loads every cycle:
                # each stream is revisited only after its previous load can
                # packet with an in-place +32 pointer advance.  Lanes 1/2 of
                # the two setup-only temp vectors are otherwise dead here.
                load_pointers = (
                    preprocess_p0,
                    preprocess_p1,
                    preprocess_p0 + 1,
                    preprocess_p1 + 1,
                )
                load_stride = preprocess_p0 + 2
                graph.emit(
                    "alu",
                    ("+", load_pointers[2], preprocess_p0, s_sixteen),
                    reads=(preprocess_p0, s_sixteen),
                    writes=(load_pointers[2],),
                    tag="preprocess_pointer_derive",
                )
                graph.emit(
                    "alu",
                    ("+", load_pointers[3], preprocess_p1, s_sixteen),
                    reads=(preprocess_p1, s_sixteen),
                    writes=(load_pointers[3],),
                    tag="preprocess_pointer_derive",
                )
                graph.emit(
                    "alu",
                    ("+", load_stride, s_sixteen, s_sixteen),
                    reads=(s_sixteen,),
                    writes=(load_stride,),
                    tag="preprocess_pointer_stride",
                )
                first_source = 2 * first_stage_pair
                staged_buffers = stage_vectors[first_source:]
                for load_index, buffer in enumerate(staged_buffers):
                    pointer = load_pointers[load_index % 4]
                    load_id = graph.emit(
                            "load",
                            ("vload", buffer, pointer),
                            reads=(pointer,),
                            writes=_words(buffer),
                            tag="tree_relocation_stage_load",
                        )
                    stage_loads.append(load_id)
                    stage_load_for_source[first_source + load_index] = load_id
                    if load_index + 4 < len(staged_buffers):
                        graph.emit(
                            "alu",
                            ("+", pointer, pointer, load_stride),
                            reads=(pointer, load_stride),
                            writes=(pointer,),
                            tag="pointer_advance",
                        )
            else:
                raise ValueError("relocation load streams must be two or four")

            # The copy and the preceding vector transform collapse into eight
            # ALU operations, freeing scarce VALU bandwidth.
            if RELOCATION_STORE_STREAMS not in (1, 2):
                raise ValueError("relocation store streams must be one or two")
            store_pointers = [s_c2_shift9]
            output_buffers = [temps[30]]
            emit_immediate(
                store_pointers[0], 16, "tree_relocation_store_pointer"
            )
            if RELOCATION_STORE_STREAMS == 2:
                # The two load-stream pointers are dead once global staging
                # completes.  Reuse one for odd destination blocks and the
                # otherwise-free group-31 temp as a second transform buffer.
                # This removes the false one-block-per-cycle release chain and
                # matches the machine's two-wide store engine.
                store_pointers.append(preprocess_p0)
                output_buffers.append(temps[31])
                graph.emit(
                    "alu",
                    (
                        "+",
                        store_pointers[1],
                        store_pointers[0],
                        s_eight,
                    ),
                    reads=(store_pointers[0], s_eight),
                    writes=(store_pointers[1],),
                    tag="tree_relocation_store_pointer",
                )
            # Derive the read-before-write barrier from actual byte/word
            # intervals rather than heap levels.  Source block ``s`` occupies
            # [22+8s, 29+8s], while destination block ``o`` occupies
            # [16+8o, 23+8o].  Consequently a destination overlaps only source
            # blocks o and o-1.  The four cached-top loads cover [7, 38] and
            # are handled by the same interval test.  Loads observe old memory
            # and stores commit at cycle end, so an overlapping load may share
            # the store's bundle (lag zero).
            source_load_intervals = [
                (22 + VLEN * source_index, 29 + VLEN * source_index, load_id)
                for source_index, load_id in stage_load_for_source.items()
            ]
            top_load_intervals = [
                (7, 14, top_loads[0]),
                (15, 22, top_loads[1]),
                (23, 30, top_loads[2]),
                (31, 38, top_loads[3]),
            ]
            memory_barrier_by_output: dict[
                int, tuple[tuple[int, int], ...]
            ] = {}
            for output_index in range(len(source_order)):
                destination_lower = 16 + VLEN * output_index
                destination_upper = destination_lower + VLEN - 1
                protected = {
                    load_id
                    for source_lower, source_upper, load_id in (
                        source_load_intervals + top_load_intervals
                    )
                    if destination_lower <= source_upper
                    and source_lower <= destination_upper
                }
                memory_barrier_by_output[output_index] = tuple(
                    (load_id, 0) for load_id in sorted(protected)
                )
            for output_index, source_index in enumerate(source_order):
                reused_top = REUSE_TOP_RELOCATION_LEVEL4 and source_index < 2
                source = stage_vectors[source_index]
                stream = output_index % RELOCATION_STORE_STREAMS
                store_pointer = store_pointers[stream]
                output_buffer = output_buffers[stream]
                for lane in range(VLEN):
                    # ``top_words`` is intentionally recycled as four of the
                    # level-4 staging vectors, so its nodes 15..30 no longer
                    # survive the global read-before-write phase.  Their
                    # transformed values are also resident in the immutable
                    # per-node broadcasts; read lane zero from those vectors
                    # and retain the two-load saving without extending a live
                    # range or adding preservation copies.
                    source_word = (
                        node_vec[
                            15
                            + source_index * VLEN
                            + (VLEN - 1 - lane)
                        ]
                        if reused_top
                        else source + (VLEN - 1 - lane)
                    )
                    graph.emit(
                        "alu",
                        (
                            "|" if reused_top else "^",
                            output_buffer + lane,
                            source_word,
                            source_word if reused_top else s_c5,
                        ),
                        reads=(
                            (source_word,)
                            if reused_top
                            else (source_word, s_c5)
                        ),
                        writes=(output_buffer + lane,),
                        tag="tree_relocation_reverse_xor",
                    )
                preprocess_stores.append(
                    graph.emit(
                        "store",
                        ("vstore", store_pointer, output_buffer),
                        reads=(store_pointer,) + _words(output_buffer),
                        deps=memory_barrier_by_output[output_index],
                        tag="tree_preprocess_store",
                    )
                )
                if output_index + RELOCATION_STORE_STREAMS < len(source_order):
                    graph.emit(
                        "alu",
                        (
                            "+",
                            store_pointer,
                            store_pointer,
                            s_sixteen
                            if RELOCATION_STORE_STREAMS == 2
                            else s_eight,
                        ),
                        reads=(
                            store_pointer,
                            s_sixteen
                            if RELOCATION_STORE_STREAMS == 2
                            else s_eight,
                        ),
                        writes=(store_pointer,),
                        tag="pointer_advance",
                    )

            # The same dead scalar becomes the reflection constant at the
            # depth-8/depth-9 boundary.  After the last relocated update the
            # address is 512+mirror; the original depth-9 location is
            # 1029-mirror = 1541-(512+mirror).
            emit_immediate(
                store_pointers[0], 1541, "tree_original_layout_reflect"
            )
            preprocess_pairs = 0
        elif PREPROCESS_MAX_DEPTH == 4:
            emit_const(preprocess_p0, 22, "preprocess_pointer")
            emit_const(preprocess_p1, 30, "preprocess_pointer")
            for pointer, source in (
                (preprocess_p0, top_words + 15),
                (preprocess_p1, top_words + 23),
            ):
                preprocess_stores.append(graph.emit(
                    "store",
                    ("vstore", pointer, source),
                    reads=(pointer,) + _words(source),
                    deps=tuple((load_id, 0) for load_id in top_loads),
                    tag="tree_preprocess_store",
                ))
            preprocess_pairs = 0
        elif preprocess_pairs:
            emit_immediate(preprocess_p0, 22, "preprocess_pointer")
            emit_immediate(preprocess_p1, 30, "preprocess_pointer")
            if REUSE_CACHED_LEVEL4_PREPROCESS:
                # Nodes 15..30 are already resident in top_words and have
                # just been transformed by ^C5 for the shallow-node cache.
                # Write those transformed vectors back directly rather than
                # loading and transforming the same two vectors a second
                # time.  This is the memory-side reuse trick: two stores, no
                # extra loads, and two fewer scarce VALU operations.
                for pointer, source in (
                    (preprocess_p0, top_words + 15),
                    (preprocess_p1, top_words + 23),
                ):
                    preprocess_stores.append(
                        graph.emit(
                            "store",
                            ("vstore", pointer, source),
                            reads=(pointer,) + _words(source),
                            deps=tuple((load_id, 0) for load_id in top_loads),
                            tag="tree_preprocess_store",
                        )
                    )
                if preprocess_pairs > 1:
                    for pointer in (preprocess_p0, preprocess_p1):
                        graph.emit(
                            "alu",
                            ("+", pointer, pointer, s_sixteen),
                            reads=(pointer, s_sixteen),
                            writes=(pointer,),
                            tag="pointer_advance",
                        )
        preprocess_start_pair = (
            1
            if preprocess_pairs and REUSE_CACHED_LEVEL4_PREPROCESS
            else 0
        )
        for pair_index in range(preprocess_start_pair, preprocess_pairs):
            for pointer, buffer in zip(
                (preprocess_p0, preprocess_p1), preprocess_buffers
            ):
                graph.emit(
                    "load",
                    ("vload", buffer, pointer),
                    reads=(pointer,),
                    writes=_words(buffer),
                    tag="tree_preprocess_load",
                )
                emit_valu(
                    "^",
                    buffer,
                    buffer,
                    v_c5,
                    tag="tree_preprocess_xor",
                )
                preprocess_stores.append(graph.emit(
                    "store",
                    ("vstore", pointer, buffer),
                    reads=(pointer,) + _words(buffer),
                    deps=tuple((load_id, 0) for load_id in top_loads),
                    tag="tree_preprocess_store",
                ))
            if pair_index != preprocess_pairs - 1:
                for pointer in (preprocess_p0, preprocess_p1):
                    graph.emit(
                        "alu",
                        ("+", pointer, pointer, s_sixteen),
                        reads=(pointer, s_sixteen),
                        writes=(pointer,),
                        tag="pointer_advance",
                    )

        if MADD_FIRST_DEPTH1 or MADD_FIRST_DEPTH1_SET:
            # Keep node 2 as the base and turn node 1 into (node1-node2).
            # A depth-1 lookup then needs one MADD and no flow slot.
            emit_valu(
                "-",
                depth1_diff,
                node_vec[1],
                node_vec[2],
                tag="depth1_pair_diff",
            )
        if SCALAR_FIRST_DEPTH1_SET:
            graph.emit(
                "flow",
                ("add_imm", s_c2_shift9, top_words + 2, 0),
                reads=(top_words + 2,),
                writes=(s_c2_shift9,),
                tag="depth1_scalar_base",
            )
            graph.emit(
                "alu",
                ("-", s_c1, top_words + 1, top_words + 2),
                reads=(top_words + 1, top_words + 2),
                writes=(s_c1,),
                tag="depth1_scalar_diff",
            )

        level4_reversed = [node_vec[i] for i in range(30, 14, -1)]
        direct_branch_records = tuple(
            (group, lane)
            for group in sorted(DIRECT_BRANCH_LOOKUPS)
            for lane in DIRECT_BRANCH_LOOKUPS[group]
        )
        if not CHAINED_DIRECT_BRANCH_BASE and len(direct_branch_records) > 9:
            raise ValueError("legacy direct branch lookup supports at most nine lanes")
        direct_branch_index = {
            record: index for index, record in enumerate(direct_branch_records)
        }
        direct_table_offsets = (0, 16, 69, 133, 261, 517, 1029, 2053, M23)
        direct_offset_registers = (
            -1,
            s_sixteen,
            depth_base.get(5, -1),
            depth_base.get(6, -1),
            depth_base.get(7, -1),
            depth_base.get(8, -1),
            depth_base.get(9, -1),
            depth_base.get(10, -1),
            s_m23,
        )
        paired_direct_records = tuple(
            (group, lanes)
            for group in sorted(PAIRED_DIRECT_BRANCH_LOOKUPS)
            for lanes in PAIRED_DIRECT_BRANCH_LOOKUPS[group]
        )
        if len(paired_direct_records) > 4:
            raise ValueError("paired direct lookup supports at most four pairs")
        paired_direct_index = {
            record: index for index, record in enumerate(paired_direct_records)
        }
        paired_direct_offsets = (0, 1029, 2053, 4097)
        paired_direct_offset_registers = (
            -1,
            depth_base.get(9, -1),
            depth_base.get(10, -1),
            s_m0,
        )
        for i, diff in enumerate(level4_diff):
            emit_valu(
                "-",
                diff,
                level4_reversed[2 * i + 1],
                level4_reversed[2 * i],
                tag="level4_pair_diff",
            )

        def select_cached(
            depth: int, state: int, workspace: int, gg: int, rnd: int
        ) -> None:
            width = 1 << depth
            start = width - 1
            # Mirror offset 0 names the actual rightmost node.
            leaves = [node_vec[start + width - 1 - i] for i in range(width)]
            temp = temps[state]
            workspace_bits, workspace_spill = workspace_registers(
                workspace, gg, rnd
            )
            saved_path = saved_second_path_bits.get(gg) if rnd >= 11 else None

            if depth == 1:
                condition = (
                    saved_path[0]
                    if saved_path is not None
                    else (
                        mirrors[state]
                        if DIRECT_MIRROR_PATH or rnd >= 11
                        else workspace_bits[0]
                    )
                )
                if rnd < 11 and gg in SCALAR_FIRST_DEPTH1_SET:
                    emit_scalarized(
                        "*",
                        temp,
                        condition,
                        s_c1,
                        b_scalar=True,
                        tag="depth1_scalar_multiply",
                        group=gg,
                        round=rnd,
                    )
                    emit_scalarized(
                        "+",
                        temp,
                        temp,
                        s_c2_shift9,
                        b_scalar=True,
                        tag="depth1_scalar_add",
                        group=gg,
                        round=rnd,
                    )
                elif (
                    MADD_FIRST_DEPTH1 or gg in MADD_FIRST_DEPTH1_SET
                ) and rnd < 11:
                    emit_madd(
                        temp,
                        condition,
                        depth1_diff,
                        node_vec[2],
                        tag="depth1_madd",
                        group=gg,
                        round=rnd,
                    )
                else:
                    emit_vselect(
                        temp, condition, leaves[1], leaves[0], group=gg, round=rnd
                    )
                return

            if depth == 2:
                if saved_path is not None:
                    condition_1 = saved_path[1]
                    condition_0 = saved_path[0]
                    mux = saved_second_path_mux[gg]
                    # Schedule the mux from the oldest (most-significant)
                    # path bit toward the newest one.  p0 is available a
                    # whole hash round before p1, so both half selections can
                    # execute speculatively while the next parity is still in
                    # flight; only the final select waits for p1.
                    emit_vselect(
                        temp,
                        condition_0,
                        leaves[2],
                        leaves[0],
                        group=gg,
                        round=rnd,
                    )
                    emit_vselect(
                        mux[0],
                        condition_0,
                        leaves[3],
                        leaves[1],
                        group=gg,
                        round=rnd,
                    )
                    emit_vselect(
                        temp,
                        condition_1,
                        mux[0],
                        temp,
                        group=gg,
                        round=rnd,
                    )
                    return
                else:
                    condition_1 = workspace_bits[1]
                    condition_0 = workspace_bits[0]
                if saved_path is None and (DIRECT_MIRROR_PATH or rnd >= 11):
                    for dest, vector_mask, scalar_mask, tag in (
                        (
                            workspace_bits[1],
                            v_one,
                            s_one,
                            "second_path_condition_1",
                        ),
                        (
                            workspace_bits[0],
                            v_two,
                            s_two,
                            "second_path_condition_0",
                        ),
                    ):
                        if (
                            gg in SCALAR_SECOND_PATH_GROUPS
                            or gg in SCALAR_SECOND_PATH_DEPTH2_GROUPS
                        ):
                            emit_scalarized(
                                "&",
                                dest,
                                mirrors[state],
                                scalar_mask,
                                b_scalar=True,
                                tag=tag,
                                group=gg,
                                round=rnd,
                            )
                        else:
                            emit_valu(
                                "&",
                                dest,
                                mirrors[state],
                                vector_mask,
                                tag=tag,
                                group=gg,
                                round=rnd,
                            )
                emit_vselect(
                    temp, condition_1, leaves[1], leaves[0], group=gg, round=rnd
                )
                emit_vselect(
                    workspace_spill[0],
                    condition_1,
                    leaves[3],
                    leaves[2],
                    group=gg,
                    round=rnd,
                )
                emit_vselect(
                    temp,
                    condition_0,
                    workspace_spill[0],
                    temp,
                    group=gg,
                    round=rnd,
                )
                return

            if depth != 3:
                raise AssertionError(depth)

            if gg in PIPELINED_DEPTH3_GROUPS and rnd == 14:
                pipeline_workspace = PIPELINED_DEPTH3_WORKSPACE_OVERRIDES.get(
                    gg, WORKSPACE_ASSIGNMENT[gg]
                )
                pipeline_bits = bits[pipeline_workspace]
                pipeline_spill = select_spill[
                    pipeline_workspace % N_SPILL_WORKSPACES
                ][0]
                emit_vselect(
                    temp,
                    pipeline_spill,
                    pipeline_bits[2],
                    pipeline_bits[1],
                    group=gg,
                    round=rnd,
                )
                return

            if saved_path is not None:
                condition_2, condition_1, condition_0 = (
                    saved_path[2],
                    saved_path[1],
                    saved_path[0],
                )
                mux = saved_second_path_mux[gg]
                # A conventional depth-first mux waits for the newest p2 bit
                # before issuing any of its seven flow operations.  Reverse
                # the tree: issue four p0 selections as soon as round 11
                # finishes, collapse them with p1 after round 12, and leave
                # only one select dependent on round-13 parity p2.
                emit_vselect(
                    temp, condition_0, leaves[4], leaves[0], group=gg, round=rnd
                )
                emit_vselect(
                    mux[0],
                    condition_0,
                    leaves[5],
                    leaves[1],
                    group=gg,
                    round=rnd,
                )
                emit_vselect(
                    mux[1],
                    condition_0,
                    leaves[6],
                    leaves[2],
                    group=gg,
                    round=rnd,
                )
                emit_vselect(
                    mux[2],
                    condition_0,
                    leaves[7],
                    leaves[3],
                    group=gg,
                    round=rnd,
                )
                emit_vselect(
                    temp,
                    condition_1,
                    mux[1],
                    temp,
                    group=gg,
                    round=rnd,
                )
                emit_vselect(
                    mux[3],
                    condition_1,
                    mux[2],
                    mux[0],
                    group=gg,
                    round=rnd,
                )
                emit_vselect(
                    temp,
                    condition_2,
                    mux[3],
                    temp,
                    group=gg,
                    round=rnd,
                )
                return
            else:
                condition_2, condition_1, condition_0 = (
                    workspace_bits[2],
                    workspace_bits[1],
                    workspace_bits[0],
                )
            if saved_path is None and (DIRECT_MIRROR_PATH or rnd >= 11):
                for dest, vector_mask, scalar_mask, tag in (
                    (
                        workspace_bits[2],
                        v_one,
                        s_one,
                        "second_path_condition_2",
                    ),
                    (
                        workspace_bits[1],
                        v_two,
                        s_two,
                        "second_path_condition_1",
                    ),
                    (
                        workspace_bits[0],
                        v_four,
                        s_four,
                        "second_path_condition_0",
                    ),
                ):
                    if (
                        gg in SCALAR_SECOND_PATH_GROUPS
                        or gg in SCALAR_SECOND_PATH_DEPTH3_GROUPS
                    ):
                        emit_scalarized(
                            "&",
                            dest,
                            mirrors[state],
                            scalar_mask,
                            b_scalar=True,
                            tag=tag,
                            group=gg,
                            round=rnd,
                        )
                    else:
                        emit_valu(
                            "&",
                            dest,
                            mirrors[state],
                            vector_mask,
                            tag=tag,
                            group=gg,
                            round=rnd,
                        )

            # Evaluate the two four-leaf halves depth-first.  Once the final
            # bottom pair has consumed bit 2, that register itself becomes a
            # legal destination, reducing the live mux stack to one spill.
            spill = workspace_spill[0]
            emit_vselect(temp, condition_2, leaves[1], leaves[0], group=gg, round=rnd)
            emit_vselect(spill, condition_2, leaves[3], leaves[2], group=gg, round=rnd)
            emit_vselect(temp, condition_1, spill, temp, group=gg, round=rnd)
            emit_vselect(spill, condition_2, leaves[5], leaves[4], group=gg, round=rnd)
            emit_vselect(condition_2, condition_2, leaves[7], leaves[6], group=gg, round=rnd)
            emit_vselect(spill, condition_1, condition_2, spill, group=gg, round=rnd)
            emit_vselect(temp, condition_0, spill, temp, group=gg, round=rnd)

        def select_level4_hybrid(
            state: int,
            workspace: int,
            gg: int,
            rnd: int,
            *,
            prepared: bool = False,
        ) -> None:
            workspace_bits, workspace_spill = workspace_registers(
                workspace, gg, rnd
            )
            saved_path = saved_second_path_bits.get(gg) if rnd >= 11 else None
            saved_level4 = saved_path if saved_path is not None and len(saved_path) >= 4 else None
            level4_key = (gg, rnd)
            if SSA_LEVEL4_WORKSPACES:
                pool = ssa_level4_pools.get(level4_key)
                if pool is None:
                    pool = virtual_vector(f"ssa_level4_pool_{gg}_{rnd}")
                    ssa_level4_pools[level4_key] = pool
                active_pool = [pool]
            else:
                active_pool = first_level4_pool
            active_condition = (
                saved_level4[3] if saved_level4 is not None else first_level4_condition
            )
            if SSA_LEVEL4_WORKSPACES and saved_level4 is None and not prepared:
                active_condition = ssa_level4_conditions.get(level4_key, -1)
                if active_condition < 0:
                    active_condition = virtual_vector(
                        f"ssa_level4_condition_{gg}_{rnd}"
                    )
                    ssa_level4_conditions[level4_key] = active_condition
            level4_condition_2 = (
                saved_level4[2] if saved_level4 is not None else workspace_bits[2]
            )
            cond = active_condition
            if prepared:
                pass
            elif saved_level4 is not None:
                pass
            elif DIRECT_MIRROR_PATH or rnd >= 11:
                for dest, vector_mask, scalar_mask, tag in (
                    (cond, v_one, s_one, "level4_condition_3"),
                    (level4_condition_2, v_two, s_two, "level4_condition_2"),
                    (workspace_bits[1], v_four, s_four, "level4_condition_1"),
                    (workspace_bits[0], v_eight, s_eight, "level4_condition_0"),
                ):
                    if gg in SCALAR_LEVEL4_CONDITION_GROUPS:
                        emit_scalarized(
                            "&",
                            dest,
                            mirrors[state],
                            scalar_mask,
                            b_scalar=True,
                            tag=tag,
                            group=gg,
                            round=rnd,
                        )
                    else:
                        emit_valu(
                            "&",
                            dest,
                            mirrors[state],
                            vector_mask,
                            tag=tag,
                            group=gg,
                            round=rnd,
                        )
            else:
                emit_scalarized(
                    "&",
                    cond,
                    mirrors[state],
                    s_one,
                    b_scalar=True,
                    tag="level4_condition_3",
                    group=gg,
                    round=rnd,
                )
                emit_scalarized(
                    "&",
                    level4_condition_2,
                    mirrors[state],
                    s_two,
                    b_scalar=True,
                    tag="level4_condition_2",
                    group=gg,
                    round=rnd,
                )
            # Evaluate the 16-leaf mux depth-first.  The saved p2/p1/p0 bits
            # are already valid conditions, and the p3 condition becomes the
            # final pair's destination after its last read.
            a = temps[state]
            b = workspace_spill[0]
            c = active_pool[0]
            c3 = active_condition
            c2 = level4_condition_2
            c1 = saved_level4[1] if saved_level4 is not None else workspace_bits[1]
            c0 = saved_level4[0] if saved_level4 is not None else workspace_bits[0]

            def pair(dest: int, pair_index: int) -> None:
                pair_limit = HYBRID_MADD_OVERRIDES.get(
                    (gg, rnd), HYBRID_MADD_PAIRS
                )
                if pair_index < pair_limit:
                    emit_madd(
                        dest,
                        c3,
                        level4_diff[pair_index],
                        level4_reversed[2 * pair_index],
                        tag="level4_hybrid_bottom",
                        group=gg,
                        round=rnd,
                    )
                else:
                    emit_vselect(
                        dest,
                        c3,
                        level4_reversed[2 * pair_index + 1],
                        level4_reversed[2 * pair_index],
                        group=gg,
                        round=rnd,
                    )

            upper_select_index = 0

            def upper_select(
                dest: int,
                condition: int,
                yes: int,
                no: int,
            ) -> None:
                nonlocal upper_select_index
                select_index = upper_select_index
                valu_count = (
                    7
                    if gg in VALU_FINAL_CACHE_SET
                    else VALU_FINAL_CACHE_COUNTS.get(gg, 0)
                )
                use_valu = (
                    rnd == rounds - 1
                    and upper_select_index < valu_count
                )
                upper_select_index += 1
                if use_valu:
                    scratch = valu_final_cache_scratch[gg]
                    if (gg, select_index) in SCALAR_VALU_FINAL_DIFF_SET:
                        emit_scalarized(
                            "-",
                            scratch,
                            yes,
                            no,
                            tag="level4_scalar_select_difference",
                            group=gg,
                            round=rnd,
                        )
                    else:
                        emit_valu(
                            "-",
                            scratch,
                            yes,
                            no,
                            tag="level4_valu_select_difference",
                            group=gg,
                            round=rnd,
                        )
                    emit_madd(
                        dest,
                        condition,
                        scratch,
                        no,
                        tag="level4_valu_select",
                        group=gg,
                        round=rnd,
                    )
                else:
                    emit_vselect(
                        dest,
                        condition,
                        yes,
                        no,
                        group=gg,
                        round=rnd,
                    )

            pair(a, 0)
            pair(b, 1)
            upper_select(a, c2, b, a)
            pair(b, 2)
            pair(c, 3)
            upper_select(b, c2, c, b)
            upper_select(a, c1, b, a)

            pair(b, 4)
            pair(c, 5)
            upper_select(b, c2, c, b)
            pair(c, 6)
            pair(c3, 7)
            upper_select(c, c2, c3, c)
            upper_select(b, c1, c, b)
            upper_select(a, c0, b, a)

        direct_global_previous_target = [-1]
        delayed_pair_previous_barrier: list[int] = []
        delayed_pair_dispatch_order: list[int] = []

        if DELAYED_PAIR_BRANCH_GROUPS:
            if PAIRED_BRANCH_FINAL or BRANCH_FINAL_LANES:
                raise ValueError(
                    "delayed pair branch prototype owns the shared branch registers"
                )
            # 16 * depth_base[5] = 16 * 69 = 1104, the padded size of
            # one group's four 256-entry pair tables.  s_c0/top_p0 is dead
            # after the top-cache loads and is recycled as the stride word.
            graph.emit(
                "alu",
                ("*", s_c0, s_sixteen, depth_base[5]),
                reads=(s_sixteen, depth_base[5]),
                writes=(s_c0,),
                tag="delayed_pair_table_stride",
            )

        def gather_node(depth: int, state: int, gg: int, rnd: int) -> None:
            mirror = mirrors[state]
            temp = temps[state]
            value = values[state]
            branch_lanes = (
                frozenset(range(VLEN))
                if PAIRED_BRANCH_FINAL
                and gg == BRANCH_FINAL_GROUP
                and rnd == rounds - 1
                and depth == 4
                else frozenset(BRANCH_FINAL_LANES)
                if gg == BRANCH_FINAL_GROUP and rnd == rounds - 1 and depth == 4
                else frozenset()
            )
            direct_branch_lanes = (
                frozenset(DIRECT_BRANCH_LOOKUPS.get(gg, ()))
                if rnd == rounds - 1 and depth == 4
                else frozenset()
            )
            paired_direct_lanes = (
                frozenset(
                    lane
                    for pair in PAIRED_DIRECT_BRANCH_LOOKUPS.get(gg, ())
                    for lane in pair
                )
                if rnd == rounds - 1 and depth == 4
                else frozenset()
            )
            delayed_pair_lanes = (
                frozenset(range(VLEN))
                if gg in DELAYED_PAIR_BRANCH_GROUPS
                and rnd == rounds - 1
                and depth == 4
                else frozenset()
            )
            skipped_lanes = (
                branch_lanes
                | direct_branch_lanes
                | paired_direct_lanes
                | delayed_pair_lanes
            )
            final_address_prepared = (
                depth == 4
                and rnd == rounds - 1
                and (
                    gg in EARLY_FINAL_ADDRESS_SET
                    or gg in VECTOR_EARLY_FINAL_ADDRESS_SET
                )
            )
            if REVERSED_RELOCATED_TREE and depth == 4 and rnd == 4:
                # The four-bit mirror is exactly the offset in the reversed
                # heap level stored at addresses 16..31.  Seed the persistent
                # absolute address in place; later relocated levels need no
                # gather-address instructions at all.
                emit_scalarized(
                    "+",
                    mirror,
                    s_sixteen,
                    mirror,
                    a_scalar=True,
                    tag="relocated_address_seed",
                    group=gg,
                    round=rnd,
                )
                address = mirror
            elif (
                REVERSED_RELOCATED_TREE
                and depth == 4
                and rnd == rounds - 1
            ):
                # The second traversal still holds a four-bit mirror rather
                # than an address, while level 4 remains in relocated memory.
                for lane in range(VLEN):
                    if lane in skipped_lanes:
                        continue
                    graph.emit(
                        "alu",
                        ("+", temp + lane, s_sixteen, mirror + lane),
                        reads=(s_sixteen, mirror + lane),
                        writes=(temp + lane,),
                        tag="relocated_final_address",
                        group=gg,
                        round=rnd,
                    )
                address = temp
            elif REVERSED_RELOCATED_TREE and 5 <= depth <= 10:
                address = mirror
            elif final_address_prepared:
                address = mirror
            elif OVERLAP_DEEP_ADDRESS:
                # At depth 4 the mirror still encodes complemented path bits;
                # convert it in place to an absolute memory address.  Later
                # deep rounds keep that address representation directly.
                if depth == 4 and not final_address_prepared and not (
                    OVERLAP_SHALLOW_ADDRESS
                    and rnd == 4
                    and gg not in FIRST_CACHE_SET
                ):
                    emit_scalarized(
                        "-",
                        mirror,
                        depth_base[depth],
                        mirror,
                        a_scalar=True,
                        tag="gather_address_seed",
                        group=gg,
                        round=rnd,
                    )
                address = mirror
            else:
                # Address generation is intentionally scalar: eight ALU slots
                # are cheaper than a scarce VALU slot in steady state.
                for lane in range(VLEN):
                    if lane in skipped_lanes:
                        continue
                    graph.emit(
                        "alu",
                        ("-", temp + lane, depth_base[depth], mirror + lane),
                        reads=(depth_base[depth], mirror + lane),
                        writes=(temp + lane,),
                        tag="gather_address",
                        group=gg,
                        round=rnd,
                    )
                address = temp
            for lane in range(VLEN):
                if lane in skipped_lanes:
                    continue
                # Each deeper level occupies twice as many contiguous vectors.
                # Explicitly order its gathers after the exact preprocessing
                # stores which transform that level in memory.
                preprocess_slice = {
                    4: (0, 2),
                    5: (2, 6),
                    6: (6, 14),
                    7: (14, 30),
                    8: (30, 62),
                }.get(depth)
                graph.emit(
                    "load",
                    ("load_offset", temp, address, lane),
                    reads=(address + lane,),
                    writes=(temp + lane,),
                    deps=(
                        tuple(
                            (store_id, 1)
                            for store_id in preprocess_stores[
                                preprocess_slice[0] : preprocess_slice[1]
                            ]
                        )
                        if preprocess_slice is not None
                        else ()
                    ),
                    tag="tree_gather",
                    group=gg,
                    round=rnd,
                )
            if branch_lanes and not PAIRED_BRANCH_FINAL:
                if SECOND_WORKSPACE_FIXED < 0:
                    raise ValueError("branch lookup requires a fixed second workspace")
                if BRANCH_DIRECT_FULL_TABLE:
                    for lane in sorted(branch_lanes):
                        graph.emit(
                            "flow",
                            ("add_imm", temp + lane, temp + lane, 0),
                            reads=(mirror + lane,),
                            writes=(temp + lane,),
                            tag="branch_final_direct",
                            group=gg,
                            round=rnd,
                        )
                    return
                candidate_yes = (
                    mirrors[BRANCH_DEAD_CANDIDATE_GROUP]
                    if BRANCH_DEDICATED_DEAD_REGS
                    else bits[SECOND_WORKSPACE_FIXED][0]
                )
                candidate_no = (
                    temps[BRANCH_DEAD_CANDIDATE_GROUP]
                    if BRANCH_DEDICATED_DEAD_REGS
                    else bits[SECOND_WORKSPACE_FIXED][1]
                )
                for lane in sorted(branch_lanes):
                    graph.emit(
                        "flow",
                        (
                            "select",
                            temp + lane,
                            temp + lane,
                            candidate_yes + lane,
                            candidate_no + lane,
                        ),
                        reads=(temp + lane, candidate_yes + lane, candidate_no + lane),
                        writes=(temp + lane,),
                        tag="branch_final_select",
                        group=gg,
                        round=rnd,
                    )
            elif branch_lanes:
                value = values[state]
                if PAIRED_EARLY_XOR:
                    emit_valu(
                        "-",
                        paired_candidate_yes,
                        paired_candidate_yes,
                        paired_candidate_no,
                        tag="paired_branch_difference",
                        group=gg,
                        round=rnd,
                    )
                    emit_madd(
                        value,
                        temp,
                        paired_candidate_yes,
                        paired_candidate_no,
                        tag="paired_branch_select",
                        group=gg,
                        round=rnd,
                    )
                elif PAIRED_FLOW_SELECT:
                    graph.emit(
                        "flow",
                        (
                            "vselect",
                            temp,
                            temp,
                            paired_candidate_yes,
                            paired_candidate_no,
                        ),
                        reads=(
                            _words(temp)
                            + _words(paired_candidate_yes)
                            + _words(paired_candidate_no)
                        ),
                        writes=_words(temp),
                        tag="paired_branch_select",
                        group=gg,
                        round=rnd,
                    )
                    emit_valu(
                        "^",
                        value,
                        value,
                        temp,
                        tag="paired_branch_node_xor",
                        group=gg,
                        round=rnd,
                    )
                else:
                    emit_madd(
                        temp,
                        temp,
                        paired_candidate_yes,
                        paired_candidate_no,
                        tag="paired_branch_select",
                        group=gg,
                        round=rnd,
                    )
                    emit_valu(
                        "^",
                        value,
                        value,
                        temp,
                        tag="paired_branch_node_xor",
                        group=gg,
                        round=rnd,
                    )
            if direct_branch_lanes:
                previous_target = direct_global_previous_target[0]
                for lane in sorted(direct_branch_lanes):
                    record_index = direct_branch_index[(gg, lane)]
                    base_ready = previous_target
                    if CHAINED_DIRECT_BRANCH_BASE and record_index:
                        # Direct records are laid out densely in 16-entry
                        # blocks.  Advance the shared table base between the
                        # globally serialized traces instead of requiring a
                        # distinct live scalar offset for every record.  This
                        # removes the former nine-record ceiling at exactly
                        # the same one-ALU cost as the old offset addition.
                        base_ready = graph.emit(
                            "alu",
                            ("+", s_c0, s_c0, s_sixteen),
                            reads=(s_c0, s_sixteen),
                            writes=(s_c0,),
                            deps=((previous_target, 1),) if previous_target >= 0 else (),
                            tag="direct_branch_base_advance",
                            group=gg,
                            round=rnd,
                        )
                    prep = graph.emit(
                        "alu",
                        (
                            "+",
                            mirror + lane,
                            mirror + lane,
                            s_c0,
                        ),
                        reads=(mirror + lane, s_c0),
                        writes=(mirror + lane,),
                        deps=((base_ready, 1),) if base_ready >= 0 else (),
                        tag="direct_branch_base",
                        group=gg,
                        round=rnd,
                    )
                    if not CHAINED_DIRECT_BRANCH_BASE and record_index:
                        offset_register = direct_offset_registers[record_index]
                        if offset_register < 0:
                            raise ValueError("missing direct branch offset register")
                        prep = graph.emit(
                            "alu",
                            (
                                "+",
                                mirror + lane,
                                mirror + lane,
                                offset_register,
                            ),
                            reads=(mirror + lane, offset_register),
                            writes=(mirror + lane,),
                            deps=((prep, 1),),
                            tag="direct_branch_offset",
                            group=gg,
                            round=rnd,
                        )
                    jump = graph.emit(
                        "flow",
                        ("jump_indirect", mirror + lane),
                        reads=(mirror + lane,),
                        deps=((prep, 1),),
                        tag="direct_branch_jump",
                        group=gg,
                        round=rnd,
                    )
                    copy = graph.emit(
                        "alu",
                        (
                            "^" if FUSED_DIRECT_BRANCH_XOR else "|",
                            value + lane if FUSED_DIRECT_BRANCH_XOR else temp + lane,
                            value + lane if FUSED_DIRECT_BRANCH_XOR else node_vec[0] + lane,
                            node_vec[0] + lane,
                        ),
                        reads=(
                            (value + lane, node_vec[0] + lane)
                            if FUSED_DIRECT_BRANCH_XOR
                            else (node_vec[0] + lane,)
                        ),
                        writes=(
                            value + lane if FUSED_DIRECT_BRANCH_XOR else temp + lane,
                        ),
                        deps=((jump, 1),),
                        tag="direct_branch_copy",
                        group=gg,
                        round=rnd,
                    )
                    previous_target = graph.emit(
                        "flow",
                        (
                            "add_imm",
                            mirror + lane,
                            mirror + lane,
                            DIRECT_TARGET_SENTINEL,
                        ),
                        deps=((copy, 0),),
                        tag="direct_branch_target",
                        group=gg,
                        round=rnd,
                    )
                direct_global_previous_target[0] = previous_target
            if paired_direct_lanes:
                previous_target = -1
                for lanes in PAIRED_DIRECT_BRANCH_LOOKUPS[gg]:
                    lane0, lane1 = lanes
                    record_index = paired_direct_index[(gg, lanes)]
                    prep = graph.emit(
                        "alu",
                        ("*", mirror + lane0, mirror + lane0, s_sixteen),
                        reads=(mirror + lane0, s_sixteen),
                        writes=(mirror + lane0,),
                        deps=((previous_target, 1),) if previous_target >= 0 else (),
                        tag="paired_direct_branch_index_high",
                        group=gg,
                        round=rnd,
                    )
                    prep = graph.emit(
                        "alu",
                        ("+", mirror + lane0, mirror + lane0, mirror + lane1),
                        reads=(mirror + lane0, mirror + lane1),
                        writes=(mirror + lane0,),
                        deps=((prep, 1),),
                        tag="paired_direct_branch_index_low",
                        group=gg,
                        round=rnd,
                    )
                    prep = graph.emit(
                        "alu",
                        ("*", mirror + lane0, mirror + lane0, s_two),
                        reads=(mirror + lane0, s_two),
                        writes=(mirror + lane0,),
                        deps=((prep, 1),),
                        tag="paired_direct_branch_stride",
                        group=gg,
                        round=rnd,
                    )
                    prep = graph.emit(
                        "alu",
                        ("*", mirror + lane0, mirror + lane0, s_two),
                        reads=(mirror + lane0, s_two),
                        writes=(mirror + lane0,),
                        deps=((prep, 1),),
                        tag="paired_direct_branch_stride",
                        group=gg,
                        round=rnd,
                    )
                    prep = graph.emit(
                        "alu",
                        ("+", mirror + lane0, mirror + lane0, s_c0),
                        reads=(mirror + lane0, s_c0),
                        writes=(mirror + lane0,),
                        deps=((prep, 1),),
                        tag="paired_direct_branch_base",
                        group=gg,
                        round=rnd,
                    )
                    if record_index:
                        offset_register = paired_direct_offset_registers[
                            record_index
                        ]
                        prep = graph.emit(
                            "alu",
                            (
                                "+",
                                mirror + lane0,
                                mirror + lane0,
                                offset_register,
                            ),
                            reads=(mirror + lane0, offset_register),
                            writes=(mirror + lane0,),
                            deps=((prep, 1),),
                            tag="paired_direct_branch_offset",
                            group=gg,
                            round=rnd,
                        )
                    jump = graph.emit(
                        "flow",
                        ("jump_indirect", mirror + lane0),
                        reads=(mirror + lane0,),
                        deps=((prep, 1),),
                        tag="paired_direct_branch_jump",
                        group=gg,
                        round=rnd,
                    )
                    copies = []
                    for lane in lanes:
                        copies.append(
                            graph.emit(
                                "alu",
                                (
                                    "^" if FUSED_DIRECT_BRANCH_XOR else "|",
                                    value + lane if FUSED_DIRECT_BRANCH_XOR else temp + lane,
                                    value + lane if FUSED_DIRECT_BRANCH_XOR else node_vec[0] + lane,
                                    node_vec[0] + lane,
                                ),
                                reads=(
                                    (value + lane, node_vec[0] + lane)
                                    if FUSED_DIRECT_BRANCH_XOR
                                    else (node_vec[0] + lane,)
                                ),
                                writes=(
                                    value + lane if FUSED_DIRECT_BRANCH_XOR else temp + lane,
                                ),
                                deps=((jump, 1),),
                                tag="paired_direct_branch_copy",
                                group=gg,
                                round=rnd,
                            )
                        )
                    previous_target = graph.emit(
                        "flow",
                        (
                            "add_imm",
                            mirror + lane0,
                            mirror + lane0,
                            DIRECT_TARGET_SENTINEL,
                        ),
                        deps=tuple((copy, 0) for copy in copies),
                        tag="paired_direct_branch_target",
                        group=gg,
                        round=rnd,
                    )
            if 4 <= depth <= PREPROCESS_MAX_DEPTH:
                # This level was transformed once during setup.
                pass
            elif REVERSED_RELOCATED_TREE or (gg, rnd) in SCALAR_DYNAMIC_XOR_SET:
                emit_scalarized(
                    "^",
                    temp,
                    temp,
                    s_c5,
                    b_scalar=True,
                    tag="dynamic_node_transform_scalar",
                    group=gg,
                    round=rnd,
                )
            else:
                emit_valu(
                    "^",
                    temp,
                    temp,
                    v_c5,
                    tag="dynamic_node_transform_vector",
                    group=gg,
                    round=rnd,
                )

        def xor_node(
            value: int,
            node: int,
            gg: int,
            rnd: int,
            *,
            node_scalar: bool = False,
            skip_lanes: frozenset[int] = frozenset(),
        ) -> None:
            if not skip_lanes and not node_scalar and (gg, rnd) in VECTOR_NODE_XOR_SET:
                emit_valu(
                    "^",
                    value,
                    value,
                    node,
                    tag="node_xor_vector",
                    group=gg,
                    round=rnd,
                )
            else:
                for lane in range(VLEN):
                    if lane in skip_lanes:
                        continue
                    source = node if node_scalar else node + lane
                    graph.emit(
                        "alu",
                        ("^", value + lane, value + lane, source),
                        reads=(value + lane, source),
                        writes=(value + lane,),
                        tag="node_xor",
                        group=gg,
                        round=rnd,
                    )

        def emit_hash(value: int, temp: int, gg: int, rnd: int) -> None:
            emit_madd(value, value, v_m0, v_c0, tag="hash_0", group=gg, round=rnd)

            # Stage 1 branches read the same old value.  Emit the reader first;
            # the following in-place writer has a zero-lag WAR dependency.
            scalar_hash = (
                (gg + rnd) % HASH_SCALAR_MOD == 0
                or (gg, rnd) in HASH_SCALAR_EXTRA
            )
            scalar_shift = scalar_hash and HASH_SCALAR_STAGE == 1
            emit_vbasic(
                ">>",
                temp,
                value,
                v_nineteen,
                scalarize=scalar_shift,
                scalar_b=s_nineteen,
                tag="hash_1_shift_scalar" if scalar_shift else "hash_1_shift_vector",
                group=gg,
                round=rnd,
            )
            emit_vbasic(
                "^",
                value,
                value,
                v_c1,
                scalarize=scalar_hash and HASH_SCALAR_STAGE == -1,
                scalar_b=s_c1,
                tag="hash_1_const",
                group=gg,
                round=rnd,
            )
            if (gg, rnd) in SCALAR_HASH1_JOIN_SET:
                emit_scalarized(
                    "^", value, value, temp,
                    tag="hash_1_join_scalar", group=gg, round=rnd,
                )
            else:
                emit_valu(
                    "^", value, value, temp,
                    tag="hash_1_join", group=gg, round=rnd,
                )

            # If y = 33*x + C2, stage 3 is
            #   (y + C3) ^ (y << 9).
            # Both branches are affine in x, so two independent MADDs compute
            # them directly and the join follows one cycle later.  This saves
            # one VALU instruction and one dependency level per hash.
            emit_madd(
                temp, value, v_m2, v_c23, tag="hash_23_add", group=gg, round=rnd
            )
            emit_madd(
                value,
                value,
                v_m23,
                v_c2_shift9,
                tag="hash_23_shift",
                group=gg,
                round=rnd,
            )
            if (gg, rnd) in SCALAR_HASH23_JOIN_SET or (
                rnd == rounds - 1 and gg in SCALAR_FINAL_HASH23_JOIN_SET
            ):
                emit_scalarized(
                    "^", value, value, temp,
                    tag="hash_23_join_scalar", group=gg, round=rnd,
                )
            else:
                emit_valu(
                    "^", value, value, temp,
                    tag="hash_23_join", group=gg, round=rnd,
                )

            if rnd == rounds - 1 and gg in SCALAR_FINAL_HASH4_SET:
                emit_scalarized(
                    "*", temp, value, s_m4,
                    b_scalar=True,
                    tag="hash_4_scalar_mul", group=gg, round=rnd,
                )
                emit_scalarized(
                    "+", value, temp, s_c4,
                    b_scalar=True,
                    tag="hash_4_scalar_add", group=gg, round=rnd,
                )
            else:
                emit_madd(
                    value, value, v_m4, v_c4,
                    tag="hash_4", group=gg, round=rnd,
                )

            emit_vbasic(
                ">>",
                temp,
                value,
                v_sixteen,
                scalarize=(scalar_hash and HASH_SCALAR_STAGE == 5)
                or (rnd == rounds - 1 and gg in SCALAR_FINAL_SHIFT_SET),
                scalar_b=s_sixteen,
                tag="hash_5_shift",
                group=gg,
                round=rnd,
            )
            if (
                PAIRED_BRANCH_FINAL
                and PAIRED_EARLY_XOR
                and gg == BRANCH_FINAL_GROUP
                and rnd == 14
            ):
                for candidate, suffix in (
                    (paired_candidate_yes, "yes"),
                    (paired_candidate_no, "no"),
                ):
                    emit_valu(
                        "^",
                        candidate,
                        candidate,
                        value,
                        tag=f"paired_branch_{suffix}_hash_value",
                        group=gg,
                        round=rnd,
                    )
                    emit_valu(
                        "^",
                        candidate,
                        candidate,
                        temp,
                        tag=f"paired_branch_{suffix}_hash_shift",
                        group=gg,
                        round=rnd,
                    )
            if rnd == rounds - 1:
                # The final round materializes the true value directly.
                if REVERSED_RELOCATED_TREE or gg in SCALAR_FINAL_C5_SET:
                    emit_scalarized(
                        "^",
                        value,
                        value,
                        s_c5,
                        b_scalar=True,
                        tag="hash_5_const_scalar",
                        group=gg,
                        round=rnd,
                    )
                else:
                    emit_valu(
                        "^",
                        value,
                        value,
                        v_c5,
                        tag="hash_5_const",
                        group=gg,
                        round=rnd,
                    )
            if (gg, rnd) in SCALAR_HASH5_JOIN_SET or (
                rnd == rounds - 1 and gg in SCALAR_FINAL_JOIN_SET
            ):
                emit_scalarized(
                    "^",
                    value,
                    value,
                    temp,
                    tag="hash_5_join_scalar",
                    group=gg,
                    round=rnd,
                )
            else:
                emit_valu(
                    "^",
                    value,
                    value,
                    temp,
                    tag="hash_5_join",
                    group=gg,
                    round=rnd,
                )

        def emit_parity(dest: int, value: int, gg: int, rnd: int) -> None:
            if (gg, rnd) in VECTOR_PARITY_SET:
                emit_valu(
                    "&",
                    dest,
                    value,
                    v_one,
                    tag="mirror_bit_vector",
                    group=gg,
                    round=rnd,
                )
            else:
                emit_scalarized(
                    "&",
                    dest,
                    value,
                    s_one,
                    b_scalar=True,
                    tag="mirror_bit",
                    group=gg,
                    round=rnd,
                )

        def process_round(state: int, workspace: int, gg: int, rnd: int) -> None:
            nonlocal delayed_pair_previous_barrier
            depth = rnd if rnd <= 10 else rnd - 11
            value = values[state]
            temp = temps[state]
            workspace_bits, _ = workspace_registers(workspace, gg, rnd)

            if not DIRECT_MIRROR_PATH and rnd == 2:
                emit_madd(
                    mirrors[state],
                    workspace_bits[0],
                    v_two,
                    workspace_bits[1],
                    tag="mirror_build_2",
                    group=gg,
                    round=rnd,
                )
            elif not DIRECT_MIRROR_PATH and rnd == 3:
                emit_madd(
                    mirrors[state],
                    mirrors[state],
                    v_two,
                    workspace_bits[2],
                    tag="mirror_build_3",
                    group=gg,
                    round=rnd,
                )

            if depth == 0:
                if (INDEPENDENT_ROOT_CACHE or DIRECT_MIRROR_PATH) and rnd == 0:
                    xor_node(value, raw_root_scalar, gg, rnd, node_scalar=True)
                else:
                    xor_node(value, node_vec[0], gg, rnd)
            elif depth <= 3:
                select_cached(depth, state, workspace, gg, rnd)
                xor_node(value, temp, gg, rnd)
            elif (rnd == 4 and gg in FIRST_CACHE_SET) or (
                rnd == rounds - 1 and gg in FINAL_CACHE_SET
            ):
                if not (
                    gg in EARLY_FINAL_CACHE_SET
                    and rnd == rounds - 1
                ):
                    select_level4_hybrid(state, workspace, gg, rnd)
                xor_node(value, temp, gg, rnd)
                if REVERSED_RELOCATED_TREE and rnd == 4:
                    emit_scalarized(
                        "+",
                        mirrors[state],
                        s_sixteen,
                        mirrors[state],
                        a_scalar=True,
                        tag="relocated_cached_address_seed",
                        group=gg,
                        round=rnd,
                    )
                elif (
                    OVERLAP_DEEP_ADDRESS
                    and (
                        not OVERLAP_SHALLOW_ADDRESS
                        or gg in FIRST_CACHE_SET
                    )
                    and rnd == 4
                ):
                    emit_scalarized(
                        "-",
                        mirrors[state],
                        depth_base[4],
                        mirrors[state],
                        a_scalar=True,
                        tag="cached_address_seed",
                        group=gg,
                        round=rnd,
                    )
            else:
                gather_node(depth, state, gg, rnd)
                if not (
                    PAIRED_BRANCH_FINAL
                    and gg == BRANCH_FINAL_GROUP
                    and rnd == rounds - 1
                ):
                    fused_lanes = (
                        frozenset(DIRECT_BRANCH_LOOKUPS.get(gg, ()))
                        | frozenset(
                            lane
                            for pair in PAIRED_DIRECT_BRANCH_LOOKUPS.get(gg, ())
                            for lane in pair
                        )
                        if FUSED_DIRECT_BRANCH_XOR
                        and rnd == rounds - 1
                        and depth == 4
                        else frozenset()
                    )
                    xor_node(value, temp, gg, rnd, skip_lanes=fused_lanes)

            if PAIRED_BRANCH_FINAL and gg == BRANCH_FINAL_GROUP and rnd == 14:
                previous_target = -1
                paired_start_dep = max(
                    i
                    for i, op in enumerate(graph.ops)
                    if op.tag == "tree_select"
                    and op.group == gg
                    and op.round == rnd
                )
                for pair in range(4):
                    lane0 = 2 * pair
                    lane1 = lane0 + 1
                    jump_register = paired_jump_registers[pair]
                    prep = graph.emit(
                        "alu",
                        ("*", jump_register, mirrors[state] + lane0, s_eight),
                        reads=(mirrors[state] + lane0, s_eight),
                        writes=(jump_register,),
                        deps=(),
                        tag="paired_branch_index_high",
                        group=gg,
                        round=rnd,
                    )
                    prep = graph.emit(
                        "alu",
                        (
                            "+",
                            jump_register,
                            jump_register,
                            mirrors[state] + lane1,
                        ),
                        reads=(jump_register, mirrors[state] + lane1),
                        writes=(jump_register,),
                        deps=((prep, 1),),
                        tag="paired_branch_index_low",
                        group=gg,
                        round=rnd,
                    )
                    prep = graph.emit(
                        "alu",
                        (
                            "+",
                            jump_register,
                            jump_register,
                            paired_base_registers[pair],
                        ),
                        reads=(jump_register, paired_base_registers[pair]),
                        writes=(jump_register,),
                        deps=((prep, 1),),
                        tag="paired_branch_table_base",
                        group=gg,
                        round=rnd,
                    )
                    jump = graph.emit(
                        "flow",
                        ("jump_indirect", jump_register),
                        reads=(jump_register,),
                        deps=(
                            ((prep, 1), (previous_target, 1))
                            if previous_target >= 0
                            else ((prep, 1), (paired_start_dep, 0))
                        ),
                        tag="paired_branch_jump",
                        group=gg,
                        round=rnd,
                    )
                    copies = []
                    for dest in (
                        paired_candidate_yes + lane0,
                        paired_candidate_no + lane0,
                        paired_candidate_yes + lane1,
                        paired_candidate_no + lane1,
                    ):
                        copies.append(
                            graph.emit(
                                "alu",
                                ("|", dest, node_vec[0], node_vec[0]),
                                reads=(node_vec[0],),
                                writes=(dest,),
                                deps=((jump, 1),),
                                tag="paired_branch_copy",
                                group=gg,
                                round=rnd,
                            )
                        )
                    previous_target = graph.emit(
                        "flow",
                        (
                            "add_imm",
                            jump_register,
                            jump_register,
                            PAIRED_TARGET_SENTINEL,
                        ),
                        deps=tuple((copy, 0) for copy in copies),
                        tag="paired_branch_target",
                        group=gg,
                        round=rnd,
                    )

            if gg in EARLY_FINAL_CACHE_SET and rnd == 14:
                for dest, mask, tag in (
                    (workspace_bits[2], v_one, "early_level4_condition_2"),
                    (workspace_bits[1], v_two, "early_level4_condition_1"),
                    (workspace_bits[0], v_four, "early_level4_condition_0"),
                ):
                    emit_valu(
                        "&",
                        dest,
                        mirrors[state],
                        mask,
                        tag=tag,
                        group=gg,
                        round=rnd,
                    )

            if gg in PIPELINED_DEPTH3_GROUPS and rnd == 13:
                pipeline_workspace = PIPELINED_DEPTH3_WORKSPACE_OVERRIDES.get(
                    gg, WORKSPACE_ASSIGNMENT[gg]
                )
                pipeline_bits = bits[pipeline_workspace]
                pipeline_spill = select_spill[
                    pipeline_workspace % N_SPILL_WORKSPACES
                ][0]
                leaves = [node_vec[14 - index] for index in range(8)]
                emit_valu(
                    "&",
                    pipeline_bits[0],
                    mirrors[state],
                    v_two,
                    tag="pipelined_depth3_condition_0",
                    group=gg,
                    round=rnd,
                )
                emit_vselect(
                    pipeline_bits[1],
                    pipeline_bits[0],
                    leaves[4],
                    leaves[0],
                    group=gg,
                    round=rnd,
                )
                emit_vselect(
                    pipeline_bits[2],
                    pipeline_bits[0],
                    leaves[5],
                    leaves[1],
                    group=gg,
                    round=rnd,
                )
                emit_vselect(
                    pipeline_spill,
                    pipeline_bits[0],
                    leaves[6],
                    leaves[2],
                    group=gg,
                    round=rnd,
                )
                emit_vselect(
                    pipeline_bits[0],
                    pipeline_bits[0],
                    leaves[7],
                    leaves[3],
                    group=gg,
                    round=rnd,
                )

            if (
                OVERLAP_SHALLOW_ADDRESS
                and gg not in FIRST_CACHE_SET
                and rnd == 3
            ):
                emit_valu(
                    "*",
                    mirrors[state],
                    mirrors[state],
                    v_two,
                    tag="depth4_address_double",
                    group=gg,
                    round=rnd,
                )
                emit_scalarized(
                    "-",
                    mirrors[state],
                    depth_base[4],
                    mirrors[state],
                    a_scalar=True,
                    tag="depth4_address_affine",
                    group=gg,
                    round=rnd,
                )
            elif OVERLAP_DEEP_ADDRESS and 4 <= rnd <= 9:
                # Once this round's node lookup has consumed the current
                # address, precompute 2*address-5 alongside the hash.  Only
                # the final parity subtraction remains on the critical path.
                emit_madd(
                    mirrors[state],
                    mirrors[state],
                    v_two,
                    v_neg_five,
                    tag="next_address_affine",
                    group=gg,
                    round=rnd,
                )
            elif (
                OVERLAP_DEEP_ADDRESS
                and rnd == 14
                and gg in VECTOR_EARLY_FINAL_ADDRESS_SET
                and gg not in FINAL_CACHE_SET
            ):
                # Once all level-4 cached selections have been emitted, their
                # first two difference vectors are dead.  Recycle them as
                # -2 and depth4_base so the final address affine is one MADD
                # and costs no additional scratch.
                if not final_address_vectors_ready[0]:
                    graph.emit(
                        "flow",
                        (
                            "add_imm",
                            level4_diff[0],
                            s_one,
                            (-3) & 0xFFFFFFFF,
                        ),
                        reads=(s_one,),
                        writes=(level4_diff[0],),
                        tag="final_address_neg_two_immediate",
                    )
                    emit_vbroadcast(
                        level4_diff[0],
                        level4_diff[0],
                        "final_address_neg_two_broadcast",
                    )
                    emit_vbroadcast(
                        level4_diff[1],
                        depth_base[4],
                        "final_address_base_broadcast",
                    )
                    final_address_vectors_ready[0] = True
                emit_madd(
                    mirrors[state],
                    mirrors[state],
                    level4_diff[0],
                    level4_diff[1],
                    tag="final_address_affine_vector_early",
                    group=gg,
                    round=rnd,
                )
            elif (
                rnd == 14
                and gg in EARLY_FINAL_ADDRESS_SET
                and gg not in FINAL_CACHE_SET
            ):
                # The depth-3 lookup is the final consumer of the path
                # offset.  Compute depth4_base - 2*offset while the hash is
                # in flight; after the hash only the parity subtraction is
                # left before the final gather can start.
                emit_scalarized(
                    "*",
                    mirrors[state],
                    mirrors[state],
                    s_two,
                    b_scalar=True,
                    tag="final_address_double_early",
                    group=gg,
                    round=rnd,
                )
                emit_scalarized(
                    "-",
                    mirrors[state],
                    depth_base[4],
                    mirrors[state],
                    a_scalar=True,
                    tag="final_address_affine_early",
                    group=gg,
                    round=rnd,
                )

            if REVERSED_RELOCATED_TREE and rnd == 9:
                # The depth-9 gather has consumed the original-layout address.
                # Precompute 2*address-5 while its node/hash are in flight;
                # only the parity subtraction remains before depth 10.
                emit_madd(
                    mirrors[state],
                    mirrors[state],
                    v_two,
                    v_c5,
                    tag="original_address_affine",
                    group=gg,
                    round=rnd,
                )

            emit_hash(value, temp, gg, rnd)

            if REVERSED_RELOCATED_TREE and 4 <= rnd <= 8:
                emit_valu(
                    "&",
                    temp,
                    value,
                    v_one,
                    tag="relocated_parity",
                    group=gg,
                    round=rnd,
                )
                emit_madd(
                    mirrors[state],
                    mirrors[state],
                    v_two,
                    temp,
                    tag="relocated_address_update",
                    group=gg,
                    round=rnd,
                )
                if rnd == 8:
                    emit_scalarized(
                        "-",
                        mirrors[state],
                        s_c2_shift9,
                        mirrors[state],
                        a_scalar=True,
                        tag="restore_original_tree_layout",
                        group=gg,
                        round=rnd,
                    )
            elif REVERSED_RELOCATED_TREE and rnd == 9:
                emit_valu(
                    "&",
                    temp,
                    value,
                    v_one,
                    tag="original_address_parity",
                    group=gg,
                    round=rnd,
                )
                emit_scalarized(
                    "-",
                    mirrors[state],
                    mirrors[state],
                    temp,
                    tag="original_address_parity_subtract",
                    group=gg,
                    round=rnd,
                )
            elif DIRECT_MIRROR_PATH:
                if rnd in (0, 11):
                    emit_parity(mirrors[state], value, gg, rnd)
                elif rnd not in (10, rounds - 1):
                    emit_parity(temp, value, gg, rnd)
                    emit_madd(
                        mirrors[state],
                        mirrors[state],
                        v_two,
                        temp,
                        tag="mirror_update",
                        group=gg,
                        round=rnd,
                    )
            elif rnd in (0, 1, 2):
                emit_parity(workspace_bits[rnd], value, gg, rnd)
            elif gg in PIPELINED_DEPTH3_GROUPS and rnd == 13:
                pipeline_workspace = PIPELINED_DEPTH3_WORKSPACE_OVERRIDES.get(
                    gg, WORKSPACE_ASSIGNMENT[gg]
                )
                pipeline_bits = bits[pipeline_workspace]
                pipeline_spill = select_spill[
                    pipeline_workspace % N_SPILL_WORKSPACES
                ][0]
                emit_valu(
                    "&",
                    temp,
                    mirrors[state],
                    v_one,
                    tag="pipelined_depth3_condition_1",
                    group=gg,
                    round=rnd,
                )
                emit_vselect(
                    pipeline_bits[1],
                    temp,
                    pipeline_spill,
                    pipeline_bits[1],
                    group=gg,
                    round=rnd,
                )
                emit_vselect(
                    pipeline_bits[2],
                    temp,
                    pipeline_bits[0],
                    pipeline_bits[2],
                    group=gg,
                    round=rnd,
                )
                emit_parity(pipeline_spill, value, gg, rnd)
                emit_madd(
                    mirrors[state],
                    mirrors[state],
                    v_two,
                    pipeline_spill,
                    tag="mirror_update_pipelined_depth3",
                    group=gg,
                    round=rnd,
                )
            elif gg in saved_second_path_bits and rnd == 11:
                emit_parity(saved_second_path_bits[gg][0], value, gg, rnd)
            elif gg in saved_second_path_bits and rnd in (12, 13):
                saved_path = saved_second_path_bits[gg]
                parity = saved_path[rnd - 11]
                emit_parity(parity, value, gg, rnd)
                # The bespoke three-bit branch-final path still needs the
                # packed mirror after depth 3.  Four-bit SSA paths carry every
                # remaining condition explicitly, so both mirror MADDs are
                # dead and can be omitted.
                if len(saved_path) == 3:
                    if rnd == 12:
                        emit_madd(
                            mirrors[state],
                            saved_path[0],
                            v_two,
                            saved_path[1],
                            tag="mirror_update_saved_path",
                            group=gg,
                            round=rnd,
                        )
                    else:
                        emit_madd(
                            mirrors[state],
                            mirrors[state],
                            v_two,
                            saved_path[2],
                            tag="mirror_update_saved_path",
                            group=gg,
                            round=rnd,
                        )
            elif gg in EARLY_FINAL_CACHE_SET and rnd == 14:
                saved_path = saved_second_path_bits.get(gg)
                early_condition = (
                    saved_path[3]
                    if saved_path is not None and len(saved_path) >= 4
                    else first_level4_condition
                )
                emit_parity(early_condition, value, gg, rnd)
                select_level4_hybrid(
                    state,
                    workspace,
                    gg,
                    rounds - 1,
                    prepared=True,
                )
            elif (
                gg in saved_second_path_bits
                and len(saved_second_path_bits[gg]) >= 4
                and rnd == 14
            ):
                emit_parity(saved_second_path_bits[gg][3], value, gg, rnd)
            elif rnd == 11:
                emit_parity(mirrors[state], value, gg, rnd)
            elif (
                rnd == 14
                and (
                    gg in EARLY_FINAL_ADDRESS_SET
                    or gg in VECTOR_EARLY_FINAL_ADDRESS_SET
                )
                and gg not in FINAL_CACHE_SET
            ):
                emit_parity(temp, value, gg, rnd)
                emit_scalarized(
                    "-",
                    mirrors[state],
                    mirrors[state],
                    temp,
                    tag="final_address_parity_early",
                    group=gg,
                    round=rnd,
                )
            elif PAIRED_BRANCH_FINAL and gg == BRANCH_FINAL_GROUP and rnd == 14:
                # The paired final lookup consumes only the new parity bit;
                # the packed four-bit mirror has no remaining user.  Avoid a
                # dead MADD and release mirror[31] for its output pointer.
                emit_parity(temp, value, gg, rnd)
            elif rnd in (12, 13, 14):
                emit_parity(temp, value, gg, rnd)
                emit_madd(
                    mirrors[state],
                    mirrors[state],
                    v_two,
                    temp,
                    tag="mirror_update",
                    group=gg,
                    round=rnd,
                )
            elif (
                OVERLAP_SHALLOW_ADDRESS
                and gg not in FIRST_CACHE_SET
                and rnd == 3
            ):
                emit_parity(temp, value, gg, rnd)
                emit_scalarized(
                    "-",
                    mirrors[state],
                    mirrors[state],
                    temp,
                    tag="depth4_address_parity",
                    group=gg,
                    round=rnd,
                )
            elif OVERLAP_DEEP_ADDRESS and 4 <= rnd <= 9:
                emit_parity(temp, value, gg, rnd)
                emit_scalarized(
                    "-",
                    mirrors[state],
                    mirrors[state],
                    temp,
                    tag="next_address_parity",
                    group=gg,
                    round=rnd,
                )
            elif rnd == 3 or 4 <= rnd <= 9:
                emit_parity(temp, value, gg, rnd)
                emit_madd(
                    mirrors[state],
                    mirrors[state],
                    v_two,
                    temp,
                    tag="mirror_update",
                    group=gg,
                    round=rnd,
                )

            if gg in DELAYED_PAIR_BRANCH_GROUPS and rnd == rounds - 2:
                # The round-14 mirror now contains the complete 4-bit level-4
                # index for every lane.  Four 16x16 indirect tables copy two
                # selected scalar nodes at a time into this group's own temp
                # vector.  Waiting until the hash finishes makes temp dead;
                # no persistent candidate vectors or memory reloads are
                # required, while the following final hash can reuse temp.
                delayed_pair_dispatch_order.append(gg)
                barrier = tuple((op_id, 1) for op_id in delayed_pair_previous_barrier)
                previous_target = -1
                for pair in range(4):
                    lane0 = 2 * pair
                    lane1 = lane0 + 1
                    jump_register = paired_jump_registers[pair]
                    prep = graph.emit(
                        "alu",
                        (
                            "*",
                            jump_register,
                            mirrors[state] + lane0,
                            s_sixteen,
                        ),
                        reads=(mirrors[state] + lane0, s_sixteen),
                        writes=(jump_register,),
                        deps=(
                            barrier
                            if pair == 0
                            else ((previous_target, 1),)
                        ),
                        tag="paired_branch_delayed_index_high",
                        group=gg,
                        round=rnd,
                    )
                    prep = graph.emit(
                        "alu",
                        (
                            "+",
                            jump_register,
                            jump_register,
                            mirrors[state] + lane1,
                        ),
                        reads=(jump_register, mirrors[state] + lane1),
                        writes=(jump_register,),
                        deps=((prep, 1),),
                        tag="paired_branch_delayed_index_low",
                        group=gg,
                        round=rnd,
                    )
                    prep = graph.emit(
                        "alu",
                        (
                            "+",
                            jump_register,
                            jump_register,
                            paired_base_registers[pair],
                        ),
                        reads=(jump_register, paired_base_registers[pair]),
                        writes=(jump_register,),
                        deps=((prep, 1),),
                        tag="paired_branch_delayed_table_base",
                        group=gg,
                        round=rnd,
                    )
                    jump = graph.emit(
                        "flow",
                        ("jump_indirect", jump_register),
                        reads=(jump_register,),
                        deps=((prep, 1),),
                        tag="paired_branch_delayed_jump",
                        group=gg,
                        round=rnd,
                    )
                    copies = []
                    for lane in (lane0, lane1):
                        copies.append(
                            graph.emit(
                                "alu",
                                (
                                    "|",
                                    temps[state] + lane,
                                    node_vec[0],
                                    node_vec[0],
                                ),
                                reads=(node_vec[0],),
                                writes=(temps[state] + lane,),
                                deps=((jump, 1),),
                                tag="paired_branch_delayed_copy",
                                group=gg,
                                round=rnd,
                            )
                        )
                    previous_target = graph.emit(
                        "flow",
                        (
                            "add_imm",
                            jump_register,
                            jump_register,
                            DELAYED_PAIR_TARGET_SENTINEL,
                        ),
                        deps=tuple((copy, 0) for copy in copies),
                        tag="paired_branch_delayed_target",
                        group=gg,
                        round=rnd,
                    )

                delayed_pair_previous_barrier = [
                    graph.emit(
                        "alu",
                        (
                            "+",
                            base_register,
                            base_register,
                            s_c0,
                        ),
                        reads=(base_register, s_c0),
                        writes=(base_register,),
                        deps=((previous_target, 1),),
                        tag="paired_branch_delayed_base_advance",
                        group=gg,
                        round=rnd,
                    )
                    for base_register in paired_base_registers
                ]

            # The final lookup is the last consumer of this group's mirror.
            # Recycle its first scalar word as a private output pointer while
            # the final hash is still running, eliminating all rolling-store
            # address dependencies without consuming scratch.
            if PER_GROUP_OUTPUT_POINTERS and rnd == rounds - 1:
                emit_const(
                    mirrors[state],
                    values_base + gg * VLEN,
                    "output_pointer",
                )

        values_base = 7 + n_nodes + batch_size
        for pointer_index, (pointer, initial) in enumerate((
            (io_p0, values_base),
            (io_p1, values_base + VLEN),
        )):
            if DERIVE_SETUP_SECOND_POINTERS and pointer_index == 1:
                graph.emit(
                    "alu",
                    ("+", pointer, io_p0, s_eight),
                    reads=(io_p0, s_eight),
                    writes=(pointer,),
                    tag="input_pointer_derive",
                )
                continue
            if "input_pointer" in LOAD_IMMEDIATE_TAGS:
                emit_const(pointer, initial, "input_pointer")
            elif INDEPENDENT_INPUT_POINTERS:
                graph.emit(
                    "flow",
                    ("add_imm", pointer, pointer, initial),
                    reads=(pointer,),
                    writes=(pointer,),
                    tag="input_pointer",
                )
            else:
                emit_immediate(pointer, initial, "input_pointer")
        for pair in range(N_GROUPS // 2):
            g0 = 2 * pair
            g1 = g0 + 1
            graph.emit(
                "load",
                ("vload", values[g0], io_p0),
                reads=(io_p0,),
                writes=_words(values[g0]),
                tag="input_load",
                group=g0,
            )
            graph.emit(
                "load",
                ("vload", values[g1], io_p1),
                reads=(io_p1,),
                writes=_words(values[g1]),
                tag="input_load",
                group=g1,
            )
            if pair != N_GROUPS // 2 - 1:
                graph.emit(
                    "alu",
                    ("+", io_p0, io_p0, s_sixteen),
                    reads=(io_p0, s_sixteen),
                    writes=(io_p0,),
                    tag="pointer_advance",
                )
                graph.emit(
                    "alu",
                    ("+", io_p1, io_p1, s_sixteen),
                    reads=(io_p1, s_sixteen),
                    writes=(io_p1,),
                    tag="pointer_advance",
                )

        def round_workspace(gg: int, rnd: int) -> int:
            workspace = WORKSPACE_ASSIGNMENT[gg]
            if rnd >= 11:
                if SECOND_WORKSPACE_ASSIGNMENT is not None:
                    workspace = SECOND_WORKSPACE_ASSIGNMENT[gg]
                elif SECOND_WORKSPACE_FIXED >= 0:
                    workspace = SECOND_WORKSPACE_FIXED + (
                        (gg + rnd - 11) % SECOND_WORKSPACE_COUNT
                    )
                else:
                    workspace = (
                        workspace + (rnd - 11) * SECOND_WORKSPACE_STRIDE
                    ) % N_WORKSPACES
            return workspace

        if TAIL_EMISSION_MODE == "full_offset":
            if not (INDEPENDENT_ROOT_CACHE or DIRECT_MIRROR_PATH):
                raise ValueError("full_offset requires an independent root cache")
            for schedule_round in range(rounds + max(FULL_ROUND_OFFSETS)):
                for gg in TAIL_GROUP_ORDER:
                    rnd = schedule_round - FULL_ROUND_OFFSETS[gg]
                    if 0 <= rnd < rounds:
                        process_round(gg, round_workspace(gg, rnd), gg, rnd)
        else:
            # Phase A acquires a workspace only through depth 3, then releases
            # it while the group's core state advances through deep rounds.
            for gg in range(N_GROUPS):
                workspace = WORKSPACE_ASSIGNMENT[gg]
                for rnd in range(4):
                    process_round(gg, workspace, gg, rnd)
                if gg in FIRST_CACHE_SET:
                    process_round(gg, workspace, gg, 4)

            if not (INDEPENDENT_ROOT_CACHE or DIRECT_MIRROR_PATH):
                emit_valu(
                    "^",
                    node_vec[0],
                    node_vec[0],
                    v_c5,
                    tag="cached_root_transform",
                )

            def emit_tail_round(gg: int, rnd: int) -> None:
                if rnd == 4 and gg in FIRST_CACHE_SET:
                    return
                process_round(gg, round_workspace(gg, rnd), gg, rnd)

            # Emission order is a scheduler hint as well as a scratch-lifetime
            # choice: the per-group state carries the true round ordering.
            if TAIL_EMISSION_MODE == "split":
                for gg in TAIL_GROUP_ORDER:
                    for rnd in range(4, 11):
                        emit_tail_round(gg, rnd)
                for gg in TAIL_GROUP_ORDER:
                    for rnd in range(11, 16):
                        emit_tail_round(gg, rnd)
            elif TAIL_EMISSION_MODE == "group":
                for gg in TAIL_GROUP_ORDER:
                    for rnd in range(4, 16):
                        emit_tail_round(gg, rnd)
            elif TAIL_EMISSION_MODE == "round":
                for rnd in range(4, 16):
                    for gg in TAIL_GROUP_ORDER:
                        emit_tail_round(gg, rnd)
            elif TAIL_EMISSION_MODE == "wave":
                cohorts = min(TAIL_EMISSION_COHORTS, N_GROUPS)
                chunk_size = (N_GROUPS + cohorts - 1) // cohorts
                chunks = [
                    TAIL_GROUP_ORDER[start : start + chunk_size]
                    for start in range(0, N_GROUPS, chunk_size)
                ]
                width = 12 + (len(chunks) - 1) * TAIL_EMISSION_STAGGER
                for wave in range(width):
                    for cohort, groups in enumerate(chunks):
                        rnd = 4 + wave - cohort * TAIL_EMISSION_STAGGER
                        if 4 <= rnd < 16:
                            for gg in groups:
                                emit_tail_round(gg, rnd)
            else:
                raise ValueError(f"unknown tail emission mode: {TAIL_EMISSION_MODE}")

        if PER_GROUP_OUTPUT_POINTERS:
            for group in range(N_GROUPS):
                graph.emit(
                    "store",
                    ("vstore", mirrors[group], values[group]),
                    reads=(mirrors[group],) + _words(values[group]),
                    tag="output_store",
                    group=group,
                )
        elif PRESERVED_TAIL_OUTPUT_POINTERS:
            # The two input streams naturally finish at groups 30 and 31.
            # Preserve those endpoint addresses for the last two stores.
            # Constants whose vector broadcasts are already complete become
            # independent pointers for groups 26..29, while c0/c23 roll only
            # through groups 0..25.  This removes all late address-chain
            # dependencies without any additional scratch allocation.
            emit_immediate(top_p0, values_base, "output_pointer")
            emit_immediate(top_p1, values_base + VLEN, "output_pointer")
            for pair in range(13):
                g0 = 2 * pair
                g1 = g0 + 1
                graph.emit(
                    "store",
                    ("vstore", top_p0, values[g0]),
                    reads=(top_p0,) + _words(values[g0]),
                    tag="output_store",
                    group=g0,
                )
                graph.emit(
                    "store",
                    ("vstore", top_p1, values[g1]),
                    reads=(top_p1,) + _words(values[g1]),
                    tag="output_store",
                    group=g1,
                )
                if pair != 12:
                    for pointer in (top_p0, top_p1):
                        graph.emit(
                            "alu",
                            ("+", pointer, pointer, s_sixteen),
                            reads=(pointer, s_sixteen),
                            writes=(pointer,),
                            tag="pointer_advance",
                        )

            for group, pointer in zip(
                range(26, 30),
                (s_nineteen, s_c2_shift9, s_c4, s_four),
            ):
                emit_const(
                    pointer,
                    values_base + group * VLEN,
                    "tail_output_pointer",
                )
                graph.emit(
                    "store",
                    ("vstore", pointer, values[group]),
                    reads=(pointer,) + _words(values[group]),
                    tag="output_store",
                    group=group,
                )

            for group, pointer in ((30, io_p0), (31, io_p1)):
                graph.emit(
                    "store",
                    ("vstore", pointer, values[group]),
                    reads=(pointer,) + _words(values[group]),
                    tag="output_store",
                    group=group,
                )
        elif OUTPUT_POINTER_STREAMS == 4:
            # Profile-guided output chains.  Groups are ordered by their
            # predicted value completion and divided between two pointers.
            # add_imm supports arbitrary (including wrapped negative) jumps,
            # so physical output order no longer constrains scheduling.
            completion_order = (
                0, 1, 2, 3, 4, 6, 8, 10, 12, 5, 14, 7, 16, 9, 18, 11,
                13, 15, 20, 17, 19, 22, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            )
            for stream, pointer in enumerate((io_p0, io_p1)):
                groups = completion_order[stream::2]
                emit_immediate(
                    pointer,
                    values_base + groups[0] * VLEN,
                    "output_pointer",
                )
                for position, group in enumerate(groups):
                    graph.emit(
                        "store",
                        ("vstore", pointer, values[group]),
                        reads=(pointer,) + _words(values[group]),
                        tag="output_store",
                        group=group,
                    )
                    if position != len(groups) - 1:
                        delta = (groups[position + 1] - group) * VLEN
                        graph.emit(
                            "flow",
                            ("add_imm", pointer, pointer, delta & 0xFFFFFFFF),
                            reads=(pointer,),
                            writes=(pointer,),
                            tag="pointer_advance",
                        )
        elif OUTPUT_POINTER_STREAMS == 16:
            # Four independent modulo-4 store chains.  Five scalar lanes from
            # group 5's mirror are dead after its final lookup,
            # so they provide pointer/stride registers without extending any
            # hash-constant lifetime.  The fifth word holds the 32-word stride, keeping
            # the total scalar-ALU work below the original two-chain design
            # while avoiding contention with the critical paired flow trace.
            output_pointer_base = mirrors[5]
            output_pointers = tuple(
                output_pointer_base + stream for stream in range(4)
            )
            output_stride = output_pointer_base + 4
            emit_const(output_stride, 4 * VLEN, "output_stride")
            for stream, pointer in enumerate(output_pointers):
                emit_const(
                    pointer,
                    values_base + stream * VLEN,
                    "output_pointer",
                )
                groups = tuple(range(stream, N_GROUPS, 4))
                for position, group in enumerate(groups):
                    graph.emit(
                        "store",
                        ("vstore", pointer, values[group]),
                        reads=(pointer,) + _words(values[group]),
                        tag="output_store",
                        group=group,
                    )
                    if position != len(groups) - 1:
                        graph.emit(
                            "alu",
                            ("+", pointer, pointer, output_stride),
                            reads=(pointer, output_stride),
                            writes=(pointer,),
                            tag="pointer_advance",
                        )
        elif OUTPUT_POINTER_STREAMS == 32:
            # Every output gets an independent address.  The top-tree staging
            # words are dead after cached-node construction, so recycling them
            # as scalar pointers costs no scratch.  This trades the two long
            # ALU pointer chains for independent load-engine immediates.
            for group in range(N_GROUPS):
                pointer = top_words + group
                emit_const(
                    pointer,
                    values_base + group * VLEN,
                    "output_pointer",
                )
                graph.emit(
                    "store",
                    ("vstore", pointer, values[group]),
                    reads=(pointer,) + _words(values[group]),
                    tag="output_store",
                    group=group,
                )
        elif OUTPUT_POINTER_STREAMS == 8:
            # Setup constants are dead by the time results become ready.  Use
            # eight of their scalar words as independent rolling store
            # pointers, preventing one late group from blocking half the batch.
            output_pointers = (
                io_p0,
                io_p1,
                top_p0,
                top_p1,
                s_c1,
                s_c2_shift9,
                s_c4,
                s_m4,
            )
            emit_immediate(
                s_nineteen,
                OUTPUT_POINTER_STREAMS * VLEN,
                "output_stride",
            )
            for stream, pointer in enumerate(output_pointers):
                emit_immediate(
                    pointer,
                    values_base + stream * VLEN,
                    "output_pointer",
                )
                groups = tuple(range(stream, N_GROUPS, OUTPUT_POINTER_STREAMS))
                for position, group in enumerate(groups):
                    graph.emit(
                        "store",
                        ("vstore", pointer, values[group]),
                        reads=(pointer,) + _words(values[group]),
                        tag="output_store",
                        group=group,
                    )
                    if position != len(groups) - 1:
                        graph.emit(
                            "alu",
                            ("+", pointer, pointer, s_nineteen),
                            reads=(pointer, s_nineteen),
                            writes=(pointer,),
                            tag="pointer_advance",
                        )
        elif SCATTERED_OUTPUT_STORES:
            # Arbitrary completion order avoids one slow value blocking the
            # rolling contiguous-address chain.  Two pointer registers form
            # interleaved one-cycle address/store pipelines on spare flow slots.
            emit_immediate(top_p0, values_base, "output_base")
            for position, group in enumerate(OUTPUT_GROUP_ORDER):
                pointer = io_p0 if position % 2 == 0 else io_p1
                graph.emit(
                    "flow",
                    ("add_imm", pointer, top_p0, group * VLEN),
                    reads=(top_p0,),
                    writes=(pointer,),
                    tag="output_address",
                    group=group,
                )
                graph.emit(
                    "store",
                    ("vstore", pointer, values[group]),
                    reads=(pointer,) + _words(values[group]),
                    tag="output_store",
                    group=group,
                )
        else:
            emit_immediate(io_p0, values_base, "output_pointer")
            emit_immediate(io_p1, values_base + VLEN, "output_pointer")
            rolling_pairs = (
                N_GROUPS // 2
                if INDEPENDENT_TAIL_OUTPUTS
                else N_GROUPS // 2
            )
            for pair in range(rolling_pairs):
                g0 = 2 * pair
                g1 = g0 + 1
                graph.emit(
                    "store",
                    ("vstore", io_p0, values[g0]),
                    reads=(io_p0,) + _words(values[g0]),
                    tag="output_store",
                    group=g0,
                )
                graph.emit(
                    "store",
                    ("vstore", io_p1, values[g1]),
                    reads=(io_p1,) + _words(values[g1]),
                    tag="output_store",
                    group=g1,
                )
                if pair != rolling_pairs - 1:
                    for pointer in (io_p0, io_p1):
                        if FLOW_OUTPUT_POINTER_ADVANCE or pair in FLOW_OUTPUT_ADVANCE_POSITIONS:
                            graph.emit(
                                "flow",
                                ("add_imm", pointer, pointer, 16),
                                reads=(pointer,),
                                writes=(pointer,),
                                tag="pointer_advance",
                            )
                        else:
                            graph.emit(
                                "alu",
                                ("+", pointer, pointer, s_sixteen),
                                reads=(pointer, s_sixteen),
                                writes=(pointer,),
                                tag="pointer_advance",
                            )
            if INDEPENDENT_TAIL_OUTPUTS:
                # Tail stores are filled into already-scheduled empty bundles
                # below.  Keeping them out of the DAG prevents their short
                # address loads from distorting the global load-distance
                # heuristic and delaying the actual computation.
                pass

        self.scratch_ptr = scratch.ptr
        self.scratch_debug = scratch.debug

        # Try several deterministic scheduling priorities.  Construction time
        # is unscored, so select the shortest legal bundle sequence.
        schedules = []
        self.dag_ops = graph.ops
        forward_cycles: list[list[int]] = []
        for policy in SCHEDULE_POLICIES:
            priority_noise = None
            if SCHEDULE_NOISE_SEED is not None:
                modulus = 2 * SCHEDULE_NOISE_AMPLITUDE + 1
                priority_noise = []
                for index in range(len(graph.ops)):
                    op = graph.ops[index]
                    if op.round != 15 and op.tag not in {
                        "output_store",
                        "pointer_advance",
                    }:
                        priority_noise.append(0)
                        continue
                    word = (
                        SCHEDULE_NOISE_SEED
                        + 0x9E3779B9 * (index + 1)
                    ) & 0xFFFFFFFF
                    word ^= word >> 16
                    word = (word * 0x7FEB352D) & 0xFFFFFFFF
                    word ^= word >> 15
                    priority_noise.append(
                        word % modulus - SCHEDULE_NOISE_AMPLITUDE
                    )
            if SCHEDULE_EXACT_CYCLES is not None:
                cycles = list(SCHEDULE_EXACT_CYCLES)
                if len(cycles) != len(graph.ops):
                    raise ValueError("exact cycle schedule does not match DAG")
                horizon = max(cycles) + 1
                exact_bundles: list[dict[str, list[tuple]]] = [
                    defaultdict(list) for _ in range(horizon)
                ]
                for op, cycle in zip(graph.ops, cycles):
                    exact_bundles[cycle][op.engine].append(op.slot)
                if any(not bundle for bundle in exact_bundles):
                    raise ValueError("exact cycle schedule contains an empty bundle")
                schedule = [dict(bundle) for bundle in exact_bundles]
            else:
                schedule, cycles = self._schedule(
                    graph.ops,
                    policy,
                    return_cycles=True,
                    external_scores=SCHEDULE_EXTERNAL_SCORES,
                    height_weight=SCHEDULE_EXTERNAL_HEIGHT_WEIGHT,
                    tie_scores=SCHEDULE_EXTERNAL_TIE_SCORES,
                    priority_noise=priority_noise,
                )
            schedules.append(schedule)
            forward_cycles.append(cycles)
        reversed_ops = self._reverse_ops(graph.ops)
        if virtual_vectors and BACKWARD_POLICIES:
            raise ValueError("virtual workspaces currently support forward scheduling only")
        for policy in BACKWARD_POLICIES:
            schedules.append(list(reversed(self._schedule(reversed_ops, policy))))
        self.schedule_lengths = {
            f"forward:{policy}": len(schedule)
            for policy, schedule in zip(SCHEDULE_POLICIES, schedules)
        }
        self.schedule_lengths.update(
            {
                f"backward:{policy}": len(schedule)
                for policy, schedule in zip(
                    BACKWARD_POLICIES, schedules[len(SCHEDULE_POLICIES) :]
                )
            }
        )
        best_index = min(range(len(schedules)), key=lambda i: len(schedules[i]))
        self.instrs = schedules[best_index]

        if virtual_vectors:
            # Calculate conservative live intervals in scheduled-cycle space,
            # then linear-scan color the virtual SIMD registers.  Intervals
            # that touch the same cycle conflict because all reads observe the
            # old scratch image and all writes commit at cycle end.
            cycles = forward_cycles[best_index]
            virtual_word_to_base = {
                base + lane: base
                for base in virtual_vectors
                for lane in range(VLEN)
            }
            intervals: dict[int, list[int]] = {}
            physical_word_to_base = {
                base + lane: base
                for base in physical_color_vectors
                for lane in range(VLEN)
            }
            physical_live_ranges: dict[int, list[list[int]]] = defaultdict(list)
            # Track definitions per lane.  ``load_offset`` reads one address
            # lane and overwrites that same lane with the loaded value; using
            # one current definition for the whole vector would incorrectly
            # kill the still-live address definitions of the other lanes and
            # could let a virtual register clobber them before their gathers.
            physical_current_definition: dict[int, list[int]] = {}
            for op_index, op in enumerate(graph.ops):
                cycle = cycles[op_index]
                for word in op.reads:
                    base = virtual_word_to_base.get(word)
                    if base is None:
                        base = physical_word_to_base.get(word)
                        if base is not None:
                            live_range = physical_current_definition.get(word)
                            if live_range is not None:
                                live_range[1] = max(live_range[1], cycle)
                        continue
                    interval = intervals.get(base)
                    if interval is None:
                        intervals[base] = [cycle, cycle]
                    else:
                        interval[0] = min(interval[0], cycle)
                        interval[1] = max(interval[1], cycle)
                for word in op.writes:
                    base = virtual_word_to_base.get(word)
                    if base is None:
                        base = physical_word_to_base.get(word)
                        if base is not None:
                            live_range = [cycle, cycle]
                            physical_live_ranges[base].append(live_range)
                            physical_current_definition[word] = live_range
                        continue
                    interval = intervals.get(base)
                    if interval is None:
                        intervals[base] = [cycle, cycle]
                    else:
                        interval[0] = min(interval[0], cycle)
                        interval[1] = max(interval[1], cycle)

            free_colors = list(physical_color_vectors)
            active: list[tuple[int, int, int]] = []
            colors: dict[int, int] = {}
            for base, (start, end) in sorted(
                intervals.items(), key=lambda item: (item[1][0], item[1][1])
            ):
                still_active: list[tuple[int, int, int]] = []
                for active_end, active_base, color in active:
                    # Use conservative cycle-disjoint intervals here.  Some
                    # mux spills are overwritten at their final occurrence,
                    # so endpoint sharing is not always the safe WAR case.
                    if active_end < start:
                        free_colors.append(color)
                    else:
                        still_active.append((active_end, active_base, color))
                active = still_active
                compatible_index = next(
                    (
                        index
                        for index, candidate in enumerate(free_colors)
                        if not any(
                            start <= physical_end and physical_start <= end
                            for physical_start, physical_end in physical_live_ranges[candidate]
                        )
                    ),
                    None,
                )
                if compatible_index is None:
                    peak = len(active) + 1
                    live_names = ",".join(
                        virtual_vectors[active_base]
                        for _, active_base, _ in active
                    )
                    raise AssertionError(
                        "virtual workspace coloring overflow: "
                        f"need at least {peak} vectors, have "
                        f"{len(physical_color_vectors)} at cycle {start}; "
                        f"new={virtual_vectors[base]}; live={live_names}"
                    )
                color = free_colors.pop(compatible_index)
                colors[base] = color
                active.append((end, base, color))
                active.sort()

            word_colors = {
                base + lane: color + lane
                for base, color in colors.items()
                for lane in range(VLEN)
            }

            def recolor_slot(slot: tuple) -> tuple:
                return tuple(
                    word_colors.get(item, item) if isinstance(item, int) else item
                    for item in slot
                )

            self.instrs = [
                {
                    engine: [recolor_slot(slot) for slot in slots]
                    for engine, slots in bundle.items()
                }
                for bundle in self.instrs
            ]
            self.virtual_workspace_colors = len(set(colors.values()))
            self.virtual_workspace_intervals = len(intervals)
            self.virtual_workspace_assignment = {
                virtual_vectors[base]: (color, tuple(intervals[base]))
                for base, color in colors.items()
            }
        else:
            self.virtual_workspace_colors = 0
            self.virtual_workspace_intervals = 0
            self.virtual_workspace_assignment = {}

        if INDEPENDENT_TAIL_OUTPUTS:
            if best_index >= len(SCHEDULE_POLICIES):
                raise ValueError("late tail stores require a forward schedule")
            op_cycles = forward_cycles[best_index]

            def first_bundle_with_room(engine: str, start: int) -> int:
                cycle = start
                while True:
                    if cycle == len(self.instrs):
                        self.instrs.append({})
                    if len(self.instrs[cycle].get(engine, ())) < SLOT_LIMITS[engine]:
                        return cycle
                    cycle += 1

            # Retain all rolling stores in the scheduling DAG because their
            # downstream height is a useful completion heuristic.  Replace
            # only groups 24..31 after scheduling: setup scalars provide
            # die during setup provide safe independent pointers, avoiding
            # both head-of-line blocking and hidden jump-table liveness.
            tail_groups = tuple(
                range(N_GROUPS - INDEPENDENT_TAIL_GROUP_COUNT, N_GROUPS)
            )
            output_indices = [
                index
                for index, op in enumerate(graph.ops)
                if op.tag == "output_store" and op.group in tail_groups
            ]
            last_prefix_store = max(
                index
                for index, op in enumerate(graph.ops)
                if op.tag == "output_store"
                and op.group == N_GROUPS - INDEPENDENT_TAIL_GROUP_COUNT - 1
            )
            removable = output_indices + [
                index
                for index, op in enumerate(graph.ops)
                if index > last_prefix_store and op.tag == "pointer_advance"
            ]
            for index in removable:
                op = graph.ops[index]
                slots = self.instrs[op_cycles[index]].get(op.engine, [])
                slots.remove(op.slot)
                if not slots:
                    self.instrs[op_cycles[index]].pop(op.engine, None)

            value_ready_by_group = {}
            for group in tail_groups:
                value_words = set(_words(values[group]))
                value_ready_by_group[group] = max(
                    op_cycles[index]
                    for index, op in enumerate(graph.ops)
                    if value_words.intersection(op.writes)
                )
            # Preserve the established eight-pointer assignment, while two
            # additional scalar constants make groups 22/23 independently
            # storable in experimental wider epilogues.  They are rewritten
            # only after their final scheduled use, so no scratch is added.
            tail_pointer_pool = (
                s_c1,
                s_m23,
                s_nineteen,
                s_c2_shift9,
                s_c0,
                s_one,
                s_c4,
                s_four,
                s_m4,
                s_c23,
            )
            if len(tail_groups) > len(tail_pointer_pool):
                raise ValueError("independent tail supports at most ten groups")
            pointer_for_group = dict(
                zip(tail_groups, tail_pointer_pool[-len(tail_groups) :])
            )
            late_store_jobs: list[tuple[int, int, int]] = []
            for group in tail_groups:
                pointer = pointer_for_group[group]
                pointer_last_use = max(
                    (
                        op_cycles[index]
                        for index, op in enumerate(graph.ops)
                        if pointer in op.reads or pointer in op.writes
                    ),
                    default=-1,
                )
                pointer_cycle = first_bundle_with_room(
                    "load", pointer_last_use + 1
                )
                self.instrs[pointer_cycle].setdefault("load", []).append(
                    ("const", pointer, values_base + group * VLEN)
                )

                value_ready = value_ready_by_group[group]
                late_store_jobs.append(
                    (max(pointer_cycle, value_ready) + 1, group, pointer)
                )

            # Unit-time, capacity-two jobs with release dates: earliest-release
            # order is optimal for makespan and avoids group-number ordering
            # leaving a usable store slot idle before a late critical value.
            for release, group, pointer in sorted(late_store_jobs):
                store_cycle = first_bundle_with_room("store", release)
                self.instrs[store_cycle].setdefault("store", []).append(
                    ("vstore", pointer, values[group])
                )

            while self.instrs and not any(self.instrs[-1].values()):
                self.instrs.pop()

        if (
            BRANCH_FINAL_LANES
            and not PAIRED_BRANCH_FINAL
            and BRANCH_DIRECT_FULL_TABLE
        ):
            if SSA_WORKSPACES:
                raise ValueError("branch final lookup and SSA workspaces are exclusive")
            if len(BRANCH_FINAL_LANES) != 1:
                raise ValueError("direct full table currently supports one lane")
            lane = BRANCH_FINAL_LANES[0]
            dest = temps[BRANCH_FINAL_GROUP] + lane
            select_pc = next(
                pc
                for pc, bundle in enumerate(self.instrs)
                for slot in bundle.get("flow", ())
                if slot[0] == "add_imm"
                and slot[1] == dest
                and slot[2] == dest
                and slot[3] == 0
            )
            if BRANCH_DISPATCH_PADDING:
                # Diagnostic semantic oracle: split the placeholder out of
                # its densely packed original bundle.  Two leading bundles
                # host address building and the indirect jump; the third is
                # the out-of-line table target.  Execution then
                # returns to the original non-flow bundle, so none of its
                # simultaneously scheduled work has to fit in a table entry.
                original_select_pc = select_pc
                for _ in range(3):
                    self.instrs.insert(
                        original_select_pc,
                        {"alu": [("|", top_p0, top_p0, top_p0)]},
                    )
                moved_pc = original_select_pc + 3
                moved_flow = [
                    slot
                    for slot in self.instrs[moved_pc].get("flow", ())
                    if not (
                        slot[0] == "add_imm"
                        and slot[1] == dest
                        and slot[2] == dest
                        and slot[3] == 0
                    )
                ]
                if moved_flow:
                    self.instrs[moved_pc]["flow"] = moved_flow
                else:
                    self.instrs[moved_pc].pop("flow", None)
                select_pc = original_select_pc + 2
            if select_pc < 2:
                raise AssertionError("direct branch has no preparation window")
            prep_pc, jump_pc = select_pc - 2, select_pc - 1
            if self.instrs[jump_pc].get("flow"):
                raise AssertionError("direct branch jump bundle already uses flow")
            if len(self.instrs[prep_pc].get("alu", ())) > 11:
                raise AssertionError("direct branch prep has no ALU room")
            target_nonflow = {
                engine: list(slots)
                for engine, slots in self.instrs[select_pc].items()
                if engine != "flow"
            }
            if len(target_nonflow.get("alu", ())) > 11:
                raise AssertionError("direct branch target has no ALU copy slot")

            if self.instrs[-1].get("flow"):
                raise AssertionError("final bundle has no flow slot for direct halt")
            self.instrs[-1]["flow"] = [("halt",)]
            main_length = len(self.instrs)
            load_pc = next(
                pc
                for pc, bundle in enumerate(self.instrs[:prep_pc])
                if len(bundle.get("load", ())) < SLOT_LIMITS["load"]
            )
            self.instrs[load_pc].setdefault("load", []).append(
                ("const", branch_table_base, main_length)
            )
            mirror_lane = mirrors[BRANCH_FINAL_GROUP] + lane
            self.instrs[prep_pc].setdefault("alu", []).append(
                ("+", mirror_lane, mirror_lane, branch_table_base)
            )
            self.instrs[jump_pc]["flow"] = [
                ("jump_indirect", mirror_lane)
            ]
            table_blocks: list[dict[str, list[tuple]]] = []
            for node_index in range(16):
                target = {
                    engine: list(slots)
                    for engine, slots in target_nonflow.items()
                }
                target.setdefault("alu", []).append(
                    (
                        "|",
                        dest,
                        level4_reversed[node_index] + lane,
                        level4_reversed[node_index] + lane,
                    )
                )
                target["flow"] = [("jump", select_pc + 1)]
                table_blocks.append(target)
            self.instrs.extend(table_blocks)
            self.branch_main_cycles = main_length
        elif BRANCH_FINAL_LANES and not PAIRED_BRANCH_FINAL:
            if SSA_WORKSPACES:
                raise ValueError("branch final lookup and SSA workspaces are exclusive")
            branch_select_dests = {
                temps[BRANCH_FINAL_GROUP] + lane for lane in BRANCH_FINAL_LANES
            }
            select_pcs = [
                pc
                for pc, bundle in enumerate(self.instrs)
                for slot in bundle.get("flow", ())
                if slot[0] == "select" and slot[1] in branch_select_dests
            ]
            if len(select_pcs) != len(BRANCH_FINAL_LANES):
                raise AssertionError(
                    f"expected {len(BRANCH_FINAL_LANES)} branch selects, "
                    f"found {len(select_pcs)}"
                )
            first_select_pc = min(select_pcs)
            last_vselect_pc = max(
                pc
                for pc, bundle in enumerate(self.instrs[:first_select_pc])
                for slot in bundle.get("flow", ())
                if slot[0] == "vselect"
            )

            lane_count = len(BRANCH_FINAL_LANES)
            if lane_count > 4:
                raise ValueError("precomputed branch tables currently support 4 lanes")
            if BRANCH_DISPATCH_PADDING:
                # Diagnostic-only semantic oracle support.  Production
                # schedules must expose these flow-free pairs naturally.
                padding_pc = last_vselect_pc + 1
                for _ in range(2 * lane_count):
                    self.instrs.insert(
                        padding_pc,
                        {"alu": [("|", top_p0, top_p0, top_p0)]},
                    )
                first_select_pc += 2 * lane_count
            dispatch_window: tuple[int, list[int], list[int]] | None = None
            if BRANCH_DEDICATED_DEAD_REGS:
                dead_base_words = set(
                    _words(mirrors[BRANCH_DEAD_CANDIDATE_GROUP])
                    + _words(temps[BRANCH_DEAD_CANDIDATE_GROUP])
                    + _words(mirrors[BRANCH_DEAD_CONTROL_GROUP])
                    + _words(temps[BRANCH_DEAD_CONTROL_GROUP])
                )
            else:
                dead_base_words = set(_words(temps[0]))
            last_dead_base_use = max(
                pc
                for pc, bundle in enumerate(self.instrs)
                if any(
                    isinstance(item, int) and item in dead_base_words
                    for slots in bundle.values()
                    for slot in slots
                    for item in slot[1:]
                )
            )
            dispatch_pairs: list[tuple[int, int]] = []
            pc = last_vselect_pc + 1
            while pc + 1 < first_select_pc and len(dispatch_pairs) < lane_count:
                target_pc = pc + 1
                target_alu = self.instrs[target_pc].get("alu", ())
                target_fits = len(target_alu) <= 10 or (
                    len(target_alu) == 11
                    and any(
                        slot[0] == "+" and slot[1] == slot[2]
                        for slot in target_alu
                    )
                )
                if (
                    not self.instrs[pc].get("flow")
                    and not self.instrs[target_pc].get("flow")
                    and target_fits
                ):
                    dispatch_pairs.append((pc, target_pc))
                    pc += 2
                else:
                    pc += 1

            if len(dispatch_pairs) == lane_count:
                base_pc = next(
                    (
                        pc
                        for pc in range(last_dead_base_use + 1, last_vselect_pc + 1)
                        if len(self.instrs[pc].get("alu", ()))
                        <= 12 - lane_count
                    ),
                    None,
                )
                if base_pc is not None:
                    reserved_copy = {target_pc: 2 for _, target_pc in dispatch_pairs}
                    prep_use: dict[int, int] = defaultdict(int)
                    target_pcs: list[int] = []
                    for jump_pc, _ in dispatch_pairs:
                        prep_pc = next(
                            (
                                candidate
                                for candidate in range(jump_pc - 1, base_pc, -1)
                                if len(self.instrs[candidate].get("alu", ()))
                                + reserved_copy.get(candidate, 0)
                                + prep_use[candidate]
                                < SLOT_LIMITS["alu"]
                            ),
                            None,
                        )
                        if prep_pc is None:
                            target_pcs = []
                            break
                        prep_use[prep_pc] += 1
                        target_pcs.append(prep_pc)
                    if len(target_pcs) == lane_count:
                        dispatch = [pc for pair in dispatch_pairs for pc in pair]
                        dispatch_window = (base_pc, target_pcs, dispatch)
            if dispatch_window is None:
                window_debug = [
                    (
                        pc,
                        len(self.instrs[pc].get("alu", ())),
                        len(self.instrs[pc].get("flow", ())),
                        self.instrs[pc].get("alu", ()),
                    )
                    for pc in range(last_vselect_pc, first_select_pc)
                ]
                raise AssertionError(
                    "no flow-free dispatch window before final parity selection: "
                    f"{window_debug}"
                )

            # The out-of-line dispatch tables are appended after the main
            # schedule.  Halt in the final real bundle prevents fall-through;
            # dynamically selected table entries jump back to the next main PC.
            if self.instrs[-1].get("flow"):
                raise AssertionError("final bundle has no flow slot for halt")
            self.instrs[-1]["flow"] = [("halt",)]

            candidate_yes = (
                mirrors[BRANCH_DEAD_CANDIDATE_GROUP]
                if BRANCH_DEDICATED_DEAD_REGS
                else bits[SECOND_WORKSPACE_FIXED][0]
            )
            candidate_no = (
                temps[BRANCH_DEAD_CANDIDATE_GROUP]
                if BRANCH_DEDICATED_DEAD_REGS
                else bits[SECOND_WORKSPACE_FIXED][1]
            )
            jump_vector = (
                temps[BRANCH_DEAD_CONTROL_GROUP]
                if BRANCH_DEDICATED_DEAD_REGS
                else bits[SECOND_WORKSPACE_FIXED][2]
            )
            table_blocks: list[dict[str, list[tuple]]] = []
            main_length = len(self.instrs)
            base_pc, target_pcs, dispatch_pcs = dispatch_window
            load_pc = next(
                pc
                for pc, bundle in enumerate(self.instrs[:base_pc])
                if len(bundle.get("load", ())) < SLOT_LIMITS["load"]
            )
            self.instrs[load_pc].setdefault("load", []).append(
                ("const", branch_table_base, main_length)
            )

            lane_base_registers = [
                (
                    mirrors[1]
                    if BRANCH_DEDICATED_DEAD_REGS
                    else temps[0]
                )
                + i
                for i in range(lane_count)
            ]
            if BRANCH_DEDICATED_DEAD_REGS:
                lane_base_registers = [
                    mirrors[BRANCH_DEAD_CONTROL_GROUP] + i
                    for i in range(lane_count)
                ]
            base_offsets = (None, s_eight, s_sixteen, s_sixteen)
            for lane_index, lane_base in enumerate(lane_base_registers):
                if lane_index == 0:
                    slot = (
                        "|", lane_base, branch_table_base, branch_table_base
                    )
                else:
                    slot = (
                        "+",
                        lane_base,
                        branch_table_base,
                        base_offsets[lane_index],
                    )
                self.instrs[base_pc].setdefault("alu", []).append(slot)

            if lane_count == 4:
                lane3_adjust_pc = next(
                    pc
                    for pc in range(base_pc + 1, target_pcs[3])
                    if len(self.instrs[pc].get("alu", ()))
                    < SLOT_LIMITS["alu"]
                )
                self.instrs[lane3_adjust_pc].setdefault("alu", []).append(
                    (
                        "+",
                        lane_base_registers[3],
                        lane_base_registers[3],
                        s_sixteen,
                    )
                )

            for lane_index, lane in enumerate(BRANCH_FINAL_LANES):
                self.instrs[target_pcs[lane_index]].setdefault("alu", []).append(
                    (
                        "+",
                        jump_vector + lane,
                        mirrors[BRANCH_FINAL_GROUP] + lane,
                        lane_base_registers[lane_index],
                    )
                )

            for lane_index, lane in enumerate(BRANCH_FINAL_LANES):
                jump_pc, target_pc = dispatch_pcs[
                    2 * lane_index : 2 + 2 * lane_index
                ]
                if len(self.instrs[target_pc].get("alu", ())) == 11:
                    alu_slots = self.instrs[target_pc]["alu"]
                    move_index = next(
                        i
                        for i, slot in enumerate(alu_slots)
                        if slot[0] == "+" and slot[1] == slot[2]
                    )
                    moved = alu_slots.pop(move_index)
                    move_pc = next(
                        pc
                        for pc in range(target_pc + 1, len(self.instrs))
                        if len(self.instrs[pc].get("alu", ())) < SLOT_LIMITS["alu"]
                    )
                    self.instrs[move_pc].setdefault("alu", []).append(moved)
                self.instrs[jump_pc]["flow"] = [
                    ("jump_indirect", jump_vector + lane)
                ]
                desired_table_offset = (0, 8, 16, 32)[lane_index]
                while len(table_blocks) < desired_table_offset:
                    table_blocks.append({"flow": [("halt",)]})
                for pair_index in range(8):
                    target = {
                        engine: list(slots)
                        for engine, slots in self.instrs[target_pc].items()
                    }
                    target.setdefault("alu", []).extend(
                        [
                            (
                                "|",
                                candidate_yes + lane,
                                level4_reversed[2 * pair_index + 1] + lane,
                                level4_reversed[2 * pair_index + 1] + lane,
                            ),
                            (
                                "|",
                                candidate_no + lane,
                                level4_reversed[2 * pair_index] + lane,
                                level4_reversed[2 * pair_index] + lane,
                            ),
                        ]
                    )
                    target["flow"] = [("jump", target_pc + 1)]
                    table_blocks.append(target)
            self.instrs.extend(table_blocks)
            self.branch_main_cycles = main_length
        else:
            self.branch_main_cycles = 0

        if DELAYED_PAIR_BRANCH_GROUPS:
            main_length = len(self.instrs)
            if self.instrs[-1].get("flow"):
                raise AssertionError("final bundle has no flow slot for delayed-pair halt")
            self.instrs[-1]["flow"] = [("halt",)]

            delayed_targets: list[tuple[int, int, int]] = []
            first_jump_pc = main_length
            for group in delayed_pair_dispatch_order:
                for pair, jump_register in enumerate(paired_jump_registers):
                    lane0 = 2 * pair
                    lane1 = lane0 + 1
                    placeholders = (
                        ("|", temps[group] + lane0, node_vec[0], node_vec[0]),
                        ("|", temps[group] + lane1, node_vec[0], node_vec[0]),
                    )
                    target_pc = next(
                        pc
                        for pc, bundle in enumerate(self.instrs[:main_length])
                        if (
                            "add_imm",
                            jump_register,
                            jump_register,
                            DELAYED_PAIR_TARGET_SENTINEL,
                        )
                        in bundle.get("flow", ())
                        and all(
                            placeholder in bundle.get("alu", ())
                            for placeholder in placeholders
                        )
                    )
                    jump_pc = target_pc - 1
                    if (
                        "jump_indirect",
                        jump_register,
                    ) not in self.instrs[jump_pc].get("flow", ()):
                        raise AssertionError(
                            f"delayed pair trace is not contiguous: {jump_pc},{target_pc}"
                        )
                    first_jump_pc = min(first_jump_pc, jump_pc)
                    delayed_targets.append((group, pair, target_pc))

            base_words = set(paired_base_registers)
            first_base_use_pc = min(
                pc
                for pc, bundle in enumerate(self.instrs[:first_jump_pc])
                for slot in bundle.get("alu", ())
                if slot[0] == "+" and slot[3] in base_words
            )
            last_old_base_use = max(
                pc
                for pc, bundle in enumerate(self.instrs[:first_base_use_pc])
                if any(
                    isinstance(item, int) and item in base_words
                    for slots in bundle.values()
                    for slot in slots
                    for item in slot[1:]
                )
            )
            base_prep_pc = next(
                pc
                for pc in range(last_old_base_use + 1, first_base_use_pc - 1)
                if len(self.instrs[pc].get("alu", ())) <= 9
                and len(self.instrs[pc + 1].get("alu", ())) <= 11
            )
            base0, base1, base2, base3 = paired_base_registers
            self.instrs[base_prep_pc].setdefault("alu", []).extend(
                [
                    ("|", base0, branch_table_base, branch_table_base),
                    ("+", base1, branch_table_base, depth_base[7]),
                    ("+", base2, branch_table_base, depth_base[8]),
                ]
            )
            self.instrs[base_prep_pc + 1].setdefault("alu", []).append(
                ("+", base3, base1, depth_base[8])
            )
            load_base_pc = next(
                pc
                for pc in range(base_prep_pc)
                if len(self.instrs[pc].get("load", ())) < SLOT_LIMITS["load"]
            )
            self.instrs[load_base_pc].setdefault("load", []).append(
                ("const", branch_table_base, main_length)
            )

            pair_offsets = (0, 261, 517, 778)
            delayed_blocks: list[dict[str, list[tuple]]] = []
            for dispatch_index, (group, pair, target_pc) in enumerate(delayed_targets):
                group_index = dispatch_index // 4
                desired_offset = (
                    group_index * DELAYED_PAIR_TABLE_STRIDE + pair_offsets[pair]
                )
                while len(delayed_blocks) < desired_offset:
                    delayed_blocks.append({"flow": [("halt",)]})
                lane0 = 2 * pair
                lane1 = lane0 + 1
                placeholders = (
                    ("|", temps[group] + lane0, node_vec[0], node_vec[0]),
                    ("|", temps[group] + lane1, node_vec[0], node_vec[0]),
                )
                for combined_index in range(256):
                    index0, index1 = divmod(combined_index, 16)
                    sources = (
                        level4_reversed[index0] + lane0,
                        level4_reversed[index1] + lane1,
                    )
                    target = {
                        engine: list(slots)
                        for engine, slots in self.instrs[target_pc].items()
                    }
                    for placeholder, source in zip(placeholders, sources):
                        copy_index = target["alu"].index(placeholder)
                        target["alu"][copy_index] = (
                            "|",
                            placeholder[1],
                            source,
                            source,
                        )
                    target["flow"] = [("jump", target_pc + 1)]
                    delayed_blocks.append(target)
            self.instrs.extend(delayed_blocks)
            self.branch_main_cycles = main_length

        if PAIRED_BRANCH_FINAL:
            main_length = len(self.instrs)
            if self.instrs[-1].get("flow"):
                raise AssertionError("final bundle has no flow slot for paired halt")
            self.instrs[-1]["flow"] = [("halt",)]

            paired_jump_pcs = []
            paired_target_pcs = []
            for pair, jump_register in enumerate(paired_jump_registers):
                jump_pc = next(
                    pc
                    for pc, bundle in enumerate(self.instrs[:main_length])
                    if ("jump_indirect", jump_register) in bundle.get("flow", ())
                )
                target_pc = next(
                    pc
                    for pc, bundle in enumerate(self.instrs[:main_length])
                    if (
                        "add_imm",
                        jump_register,
                        jump_register,
                        PAIRED_TARGET_SENTINEL,
                    )
                    in bundle.get("flow", ())
                )
                if target_pc != jump_pc + 1:
                    raise AssertionError(
                        f"paired trace {pair} is not contiguous: "
                        f"{jump_pc},{target_pc}"
                    )
                paired_jump_pcs.append(jump_pc)
                paired_target_pcs.append(target_pc)

            first_base_use_pc = min(
                pc
                for pc, bundle in enumerate(self.instrs[:main_length])
                for slot in bundle.get("alu", ())
                if slot[0] == "+" and slot[3] in paired_base_registers
            )
            base_words = set(paired_base_registers)
            last_old_base_use = max(
                pc
                for pc, bundle in enumerate(self.instrs[:first_base_use_pc])
                if any(
                    isinstance(item, int) and item in base_words
                    for slots in bundle.values()
                    for slot in slots
                    for item in slot[1:]
                )
            )
            base_prep_pc = next(
                pc
                for pc in range(last_old_base_use + 1, first_base_use_pc)
                if len(self.instrs[pc].get("alu", ())) <= 8
            )
            paired_offsets = (0, 69, 133, 261)
            paired_offset_registers = (
                -1,
                depth_base[5],
                depth_base[6],
                depth_base[7],
            )
            for pair, base_register in enumerate(paired_base_registers):
                if pair == 0:
                    slot = (
                        "|",
                        base_register,
                        branch_table_base,
                        branch_table_base,
                    )
                else:
                    slot = (
                        "+",
                        base_register,
                        branch_table_base,
                        paired_offset_registers[pair],
                    )
                self.instrs[base_prep_pc].setdefault("alu", []).append(slot)

            load_base_pc = next(
                pc
                for pc in range(base_prep_pc)
                if len(self.instrs[pc].get("load", ())) < SLOT_LIMITS["load"]
            )
            self.instrs[load_base_pc].setdefault("load", []).append(
                ("const", branch_table_base, main_length)
            )

            paired_table_blocks: list[dict[str, list[tuple]]] = []
            for pair, target_pc in enumerate(paired_target_pcs):
                while len(paired_table_blocks) < paired_offsets[pair]:
                    paired_table_blocks.append({"flow": [("halt",)]})
                lane0 = 2 * pair
                lane1 = lane0 + 1
                placeholders = (
                    (
                        "|",
                        paired_candidate_yes + lane0,
                        node_vec[0],
                        node_vec[0],
                    ),
                    (
                        "|",
                        paired_candidate_no + lane0,
                        node_vec[0],
                        node_vec[0],
                    ),
                    (
                        "|",
                        paired_candidate_yes + lane1,
                        node_vec[0],
                        node_vec[0],
                    ),
                    (
                        "|",
                        paired_candidate_no + lane1,
                        node_vec[0],
                        node_vec[0],
                    ),
                )
                for placeholder in placeholders:
                    if placeholder not in self.instrs[target_pc].get("alu", ()):
                        raise AssertionError(
                            f"paired copy is not packetized at pc={target_pc}"
                        )
                for combined_index in range(64):
                    mirror0, mirror1 = divmod(combined_index, 8)
                    replacements = (
                        (
                            level4_reversed[2 * mirror0 + 1] + lane0
                            if PAIRED_EARLY_XOR or PAIRED_FLOW_SELECT
                            else level4_diff[mirror0] + lane0
                        ),
                        level4_reversed[2 * mirror0] + lane0,
                        (
                            level4_reversed[2 * mirror1 + 1] + lane1
                            if PAIRED_EARLY_XOR or PAIRED_FLOW_SELECT
                            else level4_diff[mirror1] + lane1
                        ),
                        level4_reversed[2 * mirror1] + lane1,
                    )
                    target = {
                        engine: list(slots)
                        for engine, slots in self.instrs[target_pc].items()
                    }
                    for placeholder, source in zip(placeholders, replacements):
                        copy_index = target["alu"].index(placeholder)
                        target["alu"][copy_index] = (
                            "|",
                            placeholder[1],
                            source,
                            source,
                        )
                    target["flow"] = [("jump", target_pc + 1)]
                    paired_table_blocks.append(target)
            self.instrs.extend(paired_table_blocks)
            self.branch_main_cycles = main_length

        def bundle_mentions_scalar(bundle: dict[str, list[tuple]], word: int) -> bool:
            """Return true for real register operands, excluding immediates/lane ids."""
            for engine, slots in bundle.items():
                for slot in slots:
                    if engine in {"alu", "valu"} and word in slot[1:]:
                        return True
                    if engine == "load":
                        if slot[0] == "const" and slot[1] == word:
                            return True
                        if slot[0] in {"vload", "load_offset"} and slot[2] == word:
                            return True
                    if engine == "store" and word in slot[1:3]:
                        return True
                    if engine == "flow" and word in slot[1:3]:
                        return True
            return False

        if DIRECT_BRANCH_LOOKUPS:
            if not BRANCH_FINAL_LANES and not DELAYED_PAIR_BRANCH_GROUPS:
                main_length = len(self.instrs)
                if self.instrs[-1].get("flow"):
                    raise AssertionError("final bundle has no flow slot for halt")
                self.instrs[-1]["flow"] = [("halt",)]
                self.branch_main_cycles = main_length
            direct_table_blocks: list[dict[str, list[tuple]]] = []
            direct_base = len(self.instrs)
            first_jump_pc = min(
                pc
                for pc, bundle in enumerate(
                    self.instrs[: self.branch_main_cycles]
                )
                for slot in bundle.get("flow", ())
                if slot[0] == "jump_indirect"
                and any(
                    slot[1] == mirrors[group] + lane
                    for group in DIRECT_BRANCH_LOOKUPS
                    for lane in DIRECT_BRANCH_LOOKUPS[group]
                )
            )
            direct_dest_words = {
                mirrors[group] + lane
                for group in DIRECT_BRANCH_LOOKUPS
                for lane in DIRECT_BRANCH_LOOKUPS[group]
            }
            first_base_prep_pc = min(
                pc
                for pc, bundle in enumerate(
                    self.instrs[: self.branch_main_cycles]
                )
                for slot in bundle.get("alu", ())
                if slot[0] == "+"
                and slot[1] == slot[2]
                and slot[1] in direct_dest_words
                and slot[3] == s_c0
            )
            last_old_base_use = max(
                pc
                for pc, bundle in enumerate(self.instrs[:first_base_prep_pc])
                if bundle_mentions_scalar(bundle, s_c0)
            )
            base_load_pc = next(
                pc
                for pc in range(last_old_base_use + 1, first_base_prep_pc)
                if len(self.instrs[pc].get("load", ())) < SLOT_LIMITS["load"]
            )
            self.instrs[base_load_pc].setdefault("load", []).append(
                ("const", s_c0, direct_base)
            )
            for group in sorted(DIRECT_BRANCH_LOOKUPS):
                for lane in DIRECT_BRANCH_LOOKUPS[group]:
                    mirror_word = mirrors[group] + lane
                    temp_word = temps[group] + lane
                    jump_pcs = [
                        pc
                        for pc, bundle in enumerate(
                            self.instrs[: self.branch_main_cycles]
                        )
                        for slot in bundle.get("flow", ())
                        if slot == ("jump_indirect", mirror_word)
                    ]
                    target_pcs = [
                        pc
                        for pc, bundle in enumerate(
                            self.instrs[: self.branch_main_cycles]
                        )
                        for slot in bundle.get("flow", ())
                        if slot
                        == (
                            "add_imm",
                            mirror_word,
                            mirror_word,
                            DIRECT_TARGET_SENTINEL,
                        )
                    ]
                    if not (len(jump_pcs) == len(target_pcs) == 1):
                        raise AssertionError(
                            f"missing direct branch trace for group={group} lane={lane}"
                        )
                    jump_pc, target_pc = jump_pcs[0], target_pcs[0]
                    if target_pc != jump_pc + 1:
                        raise AssertionError(
                            "direct branch trace is not contiguous: "
                            f"group={group} lane={lane} pcs="
                            f"{jump_pc},{target_pc}"
                        )
                    record_index = direct_branch_index[(group, lane)]
                    desired_offset = (
                        16 * record_index
                        if CHAINED_DIRECT_BRANCH_BASE
                        else direct_table_offsets[record_index]
                    )
                    while len(direct_table_blocks) < desired_offset:
                        direct_table_blocks.append({"flow": [("halt",)]})
                    placeholder_copy = (
                        (
                            "^",
                            values[group] + lane,
                            values[group] + lane,
                            node_vec[0] + lane,
                        )
                        if FUSED_DIRECT_BRANCH_XOR
                        else (
                            "|",
                            temp_word,
                            node_vec[0] + lane,
                            node_vec[0] + lane,
                        )
                    )
                    if placeholder_copy not in self.instrs[target_pc].get(
                        "alu", ()
                    ):
                        raise AssertionError(
                            "direct branch copy is not packetized with target: "
                            f"group={group} lane={lane} pc={target_pc}"
                        )
                    for mirror_value in range(16):
                        target = {
                            engine: list(slots)
                            for engine, slots in self.instrs[target_pc].items()
                        }
                        copy_index = target["alu"].index(placeholder_copy)
                        source = level4_reversed[mirror_value] + lane
                        target["alu"][copy_index] = (
                            (
                                "^",
                                values[group] + lane,
                                values[group] + lane,
                                source,
                            )
                            if FUSED_DIRECT_BRANCH_XOR
                            else ("|", temp_word, source, source)
                        )
                        target["flow"] = [("jump", target_pc + 1)]
                        direct_table_blocks.append(target)
            self.instrs.extend(direct_table_blocks)
            self.direct_branch_table_bundles = len(direct_table_blocks)
        else:
            self.direct_branch_table_bundles = 0

        if PAIRED_DIRECT_BRANCH_LOOKUPS:
            if DIRECT_BRANCH_LOOKUPS:
                raise ValueError("individual and paired direct lookups are exclusive")
            paired_direct_base = len(self.instrs)
            first_jump_pc = min(
                pc
                for pc, bundle in enumerate(
                    self.instrs[: self.branch_main_cycles]
                )
                for group, lanes in paired_direct_records
                if ("jump_indirect", mirrors[group] + lanes[0])
                in bundle.get("flow", ())
            )
            paired_dest_words = {
                mirrors[group] + lanes[0]
                for group, lanes in paired_direct_records
            }
            first_base_prep_pc = min(
                pc
                for pc, bundle in enumerate(
                    self.instrs[: self.branch_main_cycles]
                )
                for slot in bundle.get("alu", ())
                if slot[0] == "+"
                and slot[1] == slot[2]
                and slot[1] in paired_dest_words
                and slot[3] == s_c0
            )
            last_old_base_use = max(
                pc
                for pc, bundle in enumerate(self.instrs[:first_base_prep_pc])
                if bundle_mentions_scalar(bundle, s_c0)
            )
            base_load_pc = next(
                pc
                for pc in range(last_old_base_use + 1, first_base_prep_pc)
                if len(self.instrs[pc].get("load", ())) < SLOT_LIMITS["load"]
            )
            self.instrs[base_load_pc].setdefault("load", []).append(
                ("const", s_c0, paired_direct_base)
            )

            paired_direct_blocks: list[dict[str, list[tuple]]] = []
            for group, lanes in paired_direct_records:
                lane0, lane1 = lanes
                mirror_word = mirrors[group] + lane0
                jump_pc = next(
                    pc
                    for pc, bundle in enumerate(
                        self.instrs[: self.branch_main_cycles]
                    )
                    if ("jump_indirect", mirror_word) in bundle.get("flow", ())
                )
                target_pc = next(
                    pc
                    for pc, bundle in enumerate(
                        self.instrs[: self.branch_main_cycles]
                    )
                    if (
                        "add_imm",
                        mirror_word,
                        mirror_word,
                        DIRECT_TARGET_SENTINEL,
                    )
                    in bundle.get("flow", ())
                )
                trace_length = target_pc - jump_pc
                if not 1 <= trace_length <= 4:
                    raise AssertionError(
                        f"paired direct trace is too long: "
                        f"group={group} lanes={lanes} pcs={jump_pc},{target_pc}"
                    )
                record_index = paired_direct_index[(group, lanes)]
                while len(paired_direct_blocks) < paired_direct_offsets[record_index]:
                    paired_direct_blocks.append({"flow": [("halt",)]})
                placeholders = tuple(
                    (
                        (
                            "^",
                            values[group] + lane,
                            values[group] + lane,
                            node_vec[0] + lane,
                        )
                        if FUSED_DIRECT_BRANCH_XOR
                        else (
                            "|",
                            temps[group] + lane,
                            node_vec[0] + lane,
                            node_vec[0] + lane,
                        )
                    )
                    for lane in lanes
                )
                copy_pcs = {
                    pc
                    for pc in range(jump_pc + 1, target_pc + 1)
                    if all(
                        placeholder in self.instrs[pc].get("alu", ())
                        for placeholder in placeholders
                    )
                }
                if len(copy_pcs) != 1:
                    raise AssertionError(
                        f"paired direct copies are split: pcs={sorted(copy_pcs)}"
                    )
                copy_pc = next(iter(copy_pcs))
                for combined_index in range(256):
                    mirror0, mirror1 = divmod(combined_index, 16)
                    entry_blocks = []
                    for trace_pc in range(jump_pc + 1, target_pc + 1):
                        target = {
                            engine: list(slots)
                            for engine, slots in self.instrs[trace_pc].items()
                        }
                        if trace_pc == copy_pc:
                            for placeholder, mirror_value, lane in zip(
                                placeholders, (mirror0, mirror1), lanes
                            ):
                                source = level4_reversed[mirror_value] + lane
                                copy_index = target["alu"].index(placeholder)
                                target["alu"][copy_index] = (
                                    (
                                        "^",
                                        values[group] + lane,
                                        values[group] + lane,
                                        source,
                                    )
                                    if FUSED_DIRECT_BRANCH_XOR
                                    else ("|", placeholder[1], source, source)
                                )
                        if trace_pc == target_pc:
                            target["flow"] = [("jump", target_pc + 1)]
                        entry_blocks.append(target)
                    while len(entry_blocks) < 4:
                        entry_blocks.append({"flow": [("halt",)]})
                    paired_direct_blocks.extend(entry_blocks)
            self.instrs.extend(paired_direct_blocks)
            self.direct_branch_table_bundles = len(paired_direct_blocks)

        self._validate_program()

    @staticmethod
    def _reverse_ops(ops: list[_Op]) -> list[_Op]:
        n = len(ops)
        original_children: list[dict[int, int]] = [dict() for _ in range(n)]
        for child, op in enumerate(ops):
            for parent, lag in op.parents.items():
                old = original_children[parent].get(child)
                if old is None or lag > old:
                    original_children[parent][child] = lag

        reversed_ops: list[_Op] = []
        for old_index in range(n - 1, -1, -1):
            parents = {
                n - 1 - child: lag
                for child, lag in original_children[old_index].items()
            }
            op = ops[old_index]
            reversed_ops.append(
                _Op(
                    engine=op.engine,
                    slot=op.slot,
                    reads=op.reads,
                    writes=op.writes,
                    parents=parents,
                    tag=op.tag,
                    group=op.group,
                    round=op.round,
                )
            )
        return reversed_ops

    @staticmethod
    def _schedule(
        ops: list[_Op],
        policy: int,
        return_cycles: bool = False,
        external_scores: list[int] | None = None,
        height_weight: int = 2,
        tie_scores: list[int] | None = None,
        priority_noise: list[int] | None = None,
    ) -> list[dict[str, list[tuple]]] | tuple[list[dict[str, list[tuple]]], list[int]]:
        n = len(ops)
        children: list[list[tuple[int, int]]] = [[] for _ in range(n)]
        indegree = [0] * n
        for child, op in enumerate(ops):
            indegree[child] = len(op.parents)
            for parent, lag in op.parents.items():
                children[parent].append((child, lag))

        # Latency critical height and a cheap downstream-work proxy.
        height = [0] * n
        reach = [0] * n
        data_height = [0] * n
        data_reach = [0] * n
        resource_weight_sets = (
            {"alu": 1, "valu": 2, "load": 6, "store": 6, "flow": 12},
            {"alu": 1, "valu": 3, "load": 6, "store": 2, "flow": 10},
            {"alu": 1, "valu": 2, "load": 8, "store": 2, "flow": 8},
            {"alu": 1, "valu": 4, "load": 5, "store": 2, "flow": 9},
        )
        resource_height = [[0] * n for _ in resource_weight_sets]
        inf = 1_000_000
        distance_to_load = [inf] * n
        for node in range(n - 1, -1, -1):
            if children[node]:
                height[node] = max(lag + height[ch] for ch, lag in children[node])
                reach[node] = min(
                    1_000_000,
                    len(children[node]) + sum(reach[ch] for ch, _ in children[node]),
                )
                data_children = [ch for ch, lag in children[node] if lag]
                if data_children:
                    data_height[node] = 1 + max(data_height[ch] for ch in data_children)
                    data_reach[node] = min(
                        1_000_000,
                        len(data_children) + sum(data_reach[ch] for ch in data_children),
                    )
            if ops[node].engine == "load":
                distance_to_load[node] = 0
            elif children[node]:
                distance_to_load[node] = min(
                    (lag + distance_to_load[ch] for ch, lag in children[node]),
                    default=inf,
                )
            for variant, weights in enumerate(resource_weight_sets):
                tail = max(
                    (resource_height[variant][ch] for ch, _ in children[node]),
                    default=0,
                )
                resource_height[variant][node] = weights[ops[node].engine] + tail

        engine_rank_sets = (
            {"load": 4, "valu": 3, "alu": 2, "flow": 1, "store": 0},
            {"valu": 4, "load": 3, "alu": 2, "flow": 1, "store": 0},
            {"alu": 4, "load": 3, "valu": 2, "flow": 1, "store": 0},
            {"flow": 4, "load": 3, "valu": 2, "alu": 1, "store": 0},
        )
        engine_rank = engine_rank_sets[policy % len(engine_rank_sets)]

        def priority(i: int) -> tuple[int, int, int, int, int]:
            op = ops[i]
            if external_scores is not None:
                group_offset = 64 if op.group < 0 else GROUP_FINE_OFFSETS[op.group]
                branch_bias = (
                    DIRECT_BRANCH_PRIORITY
                    if op.tag.startswith("direct_branch_")
                    or op.tag.startswith("paired_branch_")
                    or op.tag.startswith("paired_direct_branch_")
                    else 0
                )
                return (
                    height_weight * height[i]
                    + external_scores[i]
                    + group_offset
                    + branch_bias,
                    reach[i] // 32,
                    engine_rank[op.engine],
                    group_offset,
                    -op.group if op.group >= 0 else -i,
                )
            if policy == 86:
                cohort = -1 if op.group < 0 else op.group // 4
                round_offset = 0 if op.round < 0 else ROUND_PRIORITY_OFFSETS[op.round]
                return (
                    height[i] - 8 * cohort + round_offset,
                    reach[i] // 32,
                    engine_rank[op.engine],
                    -cohort,
                    -op.group if op.group >= 0 else -i,
                )
            if 87 <= policy <= 102:
                variant = policy - 87
                cohort_size = (2, 3, 4, 5)[(variant // 4) % 4]
                penalty = (0, 4, 8, 12)[variant % 4]
                cohort = -1 if op.group < 0 else op.group // cohort_size
                return (
                    data_height[i] - penalty * cohort,
                    data_reach[i] // 16,
                    engine_rank[op.engine],
                    -cohort,
                    i,
                )
            if 103 <= policy <= 106:
                # Match the compact scheduler used by the strongest public
                # baseline: critical height follows only true data hazards;
                # zero-lag anti-dependencies affect legality but not priority.
                return (data_height[i], 0, 0, 0, i)
            if 143 <= policy <= 158:
                variant = policy - 143
                resource_variant = variant // 4
                latency_multiplier = (0, 1, 2, 4)[variant % 4]
                return (
                    resource_height[resource_variant][i]
                    + latency_multiplier * height[i],
                    data_height[i],
                    reach[i] // 32,
                    engine_rank[op.engine],
                    -op.group if op.group >= 0 else -i,
                )
            if 159 <= policy <= 174:
                variant = policy - 159
                resource_variant = variant // 4
                divisor = (4, 8, 16, 32)[variant % 4]
                distance = distance_to_load[i]
                finite = int(distance < inf)
                load_score = -distance if finite else -inf
                return (
                    ENGINE_HEIGHT_MULTIPLIERS.get(op.engine, 1) * height[i]
                    + load_score
                    + resource_height[resource_variant][i] // divisor,
                    finite,
                    reach[i] // 32,
                    engine_rank[op.engine],
                    -op.group if op.group >= 0 else -i,
                )
            if 175 <= policy <= 178:
                group_offset = 0 if op.group < 0 else BASIC_GROUP_OFFSETS[op.group]
                round_offset = 0 if op.round < 0 else BASIC_ROUND_OFFSETS[op.round]
                return (
                    height[i] // BASIC_HEIGHT_DIVISOR + group_offset + round_offset,
                    reach[i] // BASIC_REACH_DIVISOR,
                    engine_rank_sets[policy % 4][op.engine],
                    -op.group if op.group >= 0 else 0,
                    i,
                )
            if 107 <= policy <= 110:
                return (data_height[i], data_reach[i], 0, 0, i)
            if 111 <= policy <= 142:
                variant = policy - 111
                load_multiplier = (1, 2, 3, 4)[variant % 4]
                launch_penalty = (0, 1, 2, 4, 8, 12, 16, 24)[variant // 4]
                finite = int(distance_to_load[i] < inf)
                load_score = (
                    -load_multiplier * distance_to_load[i] if finite else -inf
                )
                launch = (
                    -1 if op.group < 0 else FULL_ROUND_OFFSETS[op.group]
                )
                return (
                    height[i] + load_score - launch_penalty * launch,
                    finite,
                    -launch,
                    reach[i] // 32,
                    i,
                )
            if 82 <= policy <= 85:
                group_offset = 64 if op.group < 0 else GROUP_PRIORITY_OFFSETS[op.group]
                tag_offset = TAG_PRIORITY_OFFSETS.get(op.tag, 0)
                return (
                    height[i] + group_offset + tag_offset,
                    reach[i] // 32,
                    engine_rank[op.engine],
                    group_offset,
                    -op.group if op.group >= 0 else -i,
                )
            if policy == 81:
                group_offset = 64 if op.group < 0 else GROUP_PRIORITY_OFFSETS[op.group]
                round_offset = 0 if op.round < 0 else ROUND_PRIORITY_OFFSETS[op.round]
                return (
                    height[i] + group_offset + round_offset,
                    reach[i] // 32,
                    engine_rank[op.engine],
                    group_offset,
                    -op.group if op.group >= 0 else -i,
                )
            if policy == 80:
                offset = 64 if op.group < 0 else GROUP_PRIORITY_OFFSETS[op.group]
                return (
                    height[i] + offset,
                    reach[i] // 32,
                    engine_rank[op.engine],
                    offset,
                    -op.group if op.group >= 0 else -i,
                )
            if policy >= 72:
                pipeline_penalties = (32, 40, 48, 56, 64, 72, 80, 96)
                penalty = pipeline_penalties[policy - 72]
                if op.group < 0:
                    pipeline_rank = -1
                else:
                    pipeline_rank = 2 * ((op.group % 16) // 4) + op.group // 16
                return (
                    height[i] - pipeline_rank * penalty,
                    reach[i] // 32,
                    engine_rank[op.engine],
                    -pipeline_rank,
                    -op.group if op.group >= 0 else -i,
                )
            if policy >= 48:
                tuned_penalties = (56, 60, 62, 64, 66, 68)
                penalty = tuned_penalties[(policy - 48) // 4]
                cohort = -1 if op.group < 0 else op.group // 4
                return (
                    height[i] - cohort * penalty,
                    reach[i] // 32,
                    engine_rank[op.engine],
                    -cohort,
                    -op.group if op.group >= 0 else -i,
                )
            if policy >= 32:
                variant = policy - 32
                distance = distance_to_load[i]
                finite = int(distance < inf)
                if variant < 4:
                    cohort_size = (3, 4, 5, 6)[variant]
                    cohort = -1 if op.group < 0 else op.group // cohort_size
                    return (
                        finite,
                        -distance if finite else -inf,
                        -cohort,
                        height[i],
                        engine_rank[op.engine],
                    )
                multipliers = (1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32)
                multiplier = ENGINE_LOAD_MULTIPLIERS.get(
                    op.engine, multipliers[variant - 4]
                )
                load_score = -multiplier * distance if finite else -inf
                cohort = -1 if op.group < 0 else op.group // 4
                fine_offset = (
                    0 if op.group < 0 else GROUP_FINE_OFFSETS[op.group]
                )
                round_offset = (
                    0 if op.round < 0 else ROUND_PRIORITY_OFFSETS[op.round]
                )
                return (
                    ENGINE_HEIGHT_MULTIPLIERS.get(op.engine, 1) * height[i]
                    + load_score
                    + fine_offset
                    + round_offset
                    + TAG_PRIORITY_OFFSETS.get(op.tag, 0)
                    + OP_PRIORITY_OFFSETS.get((op.tag, op.group, op.round), 0)
                    + (0 if priority_noise is None else priority_noise[i]),
                    finite,
                    -cohort,
                    reach[i] // 32,
                    engine_rank[op.engine],
                    0 if tie_scores is None else tie_scores[i],
                )
            if policy >= 16:
                penalties = (8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128)
                penalty = penalties[policy - 16]
                cohort = -1 if op.group < 0 else op.group // 4
                return (
                    height[i]
                    - cohort * penalty
                    + (0 if op.group < 0 else GROUP_FINE_OFFSETS[op.group]),
                    reach[i] // 32,
                    engine_rank[op.engine],
                    -cohort,
                    -op.group if op.group >= 0 else -i,
                )
            if policy >= 8:
                cohort_sizes = (4, 5, 6, 7, 8, 10, 12, 16)
                cohort_size = cohort_sizes[policy - 8]
                # Setup is cohort -1.  Thereafter a small group cohort is
                # allowed to run ahead into gathers while later cohorts keep
                # flow/VALU busy with shallow rounds.
                cohort = -1 if op.group < 0 else op.group // cohort_size
                return (
                    -cohort,
                    height[i],
                    reach[i] // 32,
                    engine_rank[op.engine],
                    -op.group if op.group >= 0 else -i,
                )
            # Policies 0..3 strongly honor latency; 4..7 allow cohort/order
            # locality to break near-critical ties more aggressively.
            h = height[i] if policy < 4 else height[i] // 2
            group_bias = -op.group if op.group >= 0 else 0
            if policy in (2, 6):
                group_bias = -(op.group % 4) if op.group >= 0 else 0
            id_bias = -i if policy % 2 == 0 else i
            op_offset = OP_PRIORITY_OFFSETS.get((op.tag, op.group, op.round), 0)
            direct_branch_bias = (
                DIRECT_BRANCH_PRIORITY
                if op.tag.startswith("direct_branch_")
                or op.tag.startswith("paired_branch_")
                or op.tag.startswith("paired_direct_branch_")
                else 0
            )
            return (
                h
                + op_offset
                + direct_branch_bias
                + (0 if priority_noise is None else priority_noise[i]),
                reach[i] // 32,
                engine_rank[op.engine],
                group_bias,
                id_bias,
            )

        earliest = [0] * n
        heaps: dict[str, list[tuple[tuple[int, ...], int]]] = {
            engine: [] for engine in ("alu", "valu", "load", "store", "flow")
        }

        def push_ready(i: int) -> None:
            p = priority(i)
            heapq.heappush(heaps[ops[i].engine], (tuple(-x for x in p), i))

        future: dict[int, list[int]] = defaultdict(list)
        for i, deg in enumerate(indegree):
            if deg == 0:
                push_ready(i)

        scheduled = 0
        scheduled_cycle = [-1] * n
        cycle = 0
        bundles: list[dict[str, list[tuple]]] = []
        engine_order_variants = (
            ("load", "valu", "alu", "flow", "store"),
            ("valu", "load", "alu", "flow", "store"),
            ("alu", "load", "valu", "flow", "store"),
            ("flow", "load", "valu", "alu", "store"),
        )
        engine_order = engine_order_variants[policy % 4]

        while scheduled < n:
            for i in future.pop(cycle, ()):
                push_ready(i)

            used = {engine: 0 for engine in heaps}
            bundle: dict[str, list[tuple]] = defaultdict(list)

            made_progress = True
            while made_progress:
                made_progress = False
                for engine in engine_order:
                    if used[engine] >= SLOT_LIMITS[engine]:
                        continue
                    if engine == "flow" and any(
                        ops[item[1]].tag
                        in {
                            "paired_branch_delayed_copy",
                            "direct_branch_copy",
                            "paired_direct_branch_copy",
                        }
                        and earliest[item[1]] <= cycle
                        for item in heaps["alu"]
                    ):
                        # Packetize all delayed-table copy placeholders before
                        # choosing the sole flow op.  Their target marker then
                        # becomes ready in this same cycle and must sit exactly
                        # one bundle after the indirect jump; intervening main
                        # flow instructions would be skipped at run time.
                        continue
                    heap = heaps[engine]
                    if not heap:
                        continue

                    deferred = []
                    chosen = None
                    while heap:
                        item = heapq.heappop(heap)
                        candidate = item[1]
                        if earliest[candidate] <= cycle:
                            chosen = candidate
                            break
                        deferred.append(item)
                    for item in deferred:
                        heapq.heappush(heap, item)
                    if chosen is None:
                        continue

                    op = ops[chosen]
                    bundle[engine].append(op.slot)
                    used[engine] += 1
                    scheduled += 1
                    scheduled_cycle[chosen] = cycle
                    made_progress = True

                    for child, lag in children[chosen]:
                        indegree[child] -= 1
                        ready_at = cycle + lag
                        if ready_at > earliest[child]:
                            earliest[child] = ready_at
                        if indegree[child] == 0:
                            if earliest[child] <= cycle:
                                push_ready(child)
                            else:
                                future[earliest[child]].append(child)

            if not bundle:
                # All dependency lags are at most one, so this can occur only
                # before the next real cycle.  Do not emit a zero-cost bundle.
                cycle += 1
                continue
            bundles.append(dict(bundle))
            cycle += 1

        if return_cycles:
            return bundles, scheduled_cycle
        return bundles

    @staticmethod
    def _compact_schedule(ops: list[_Op], cycles: list[int]) -> list[dict[str, list[tuple]]]:
        cycles = cycles.copy()
        for _ in range(4):
            horizon = max(cycles) + 1
            usage = [defaultdict(int) for _ in range(horizon)]
            for i, op in enumerate(ops):
                usage[cycles[i]][op.engine] += 1

            moved = 0
            for i, op in enumerate(ops):
                earliest = 0
                if op.parents:
                    earliest = max(cycles[parent] + lag for parent, lag in op.parents.items())
                current = cycles[i]
                for target in range(earliest, current):
                    if usage[target][op.engine] < SLOT_LIMITS[op.engine]:
                        usage[current][op.engine] -= 1
                        usage[target][op.engine] += 1
                        cycles[i] = target
                        moved += 1
                        break
            if moved == 0:
                break

        horizon = max(cycles) + 1
        bundles: list[dict[str, list[tuple]]] = [defaultdict(list) for _ in range(horizon)]
        for i, op in enumerate(ops):
            bundles[cycles[i]][op.engine].append(op.slot)
        return [dict(bundle) for bundle in bundles if bundle]

    def _validate_program(self) -> None:
        if self.scratch_ptr > SCRATCH_SIZE:
            raise AssertionError(f"scratch overflow: {self.scratch_ptr}")
        for pc, bundle in enumerate(self.instrs):
            if not bundle:
                raise AssertionError(f"empty bundle at pc={pc}")
            for engine, slots in bundle.items():
                if len(slots) > SLOT_LIMITS[engine]:
                    raise AssertionError(
                        f"{engine} overflow at pc={pc}: {len(slots)} > {SLOT_LIMITS[engine]}"
                    )
