import numpy as np
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISRTA Benchmark MPS Generator (airport-style instances)
-------------------------------------------------------
按论文 7.1.1 的方式生成基准算例：
- 7-day horizon
- tasks_per_day ∈ {150, 300}
- tightness ∈ {0.6, 0.9}
- skilling ∈ {0.3, 1.0}
- staff = 30, shift_len = 8h, shift starts at {06:00, 09:00, 12:00}
"""

import os, random, csv, math
from typing import Dict, List, Set, Tuple
import pulp

# ============================================================
# USER SETTINGS
# ============================================================
# 选择：生成单个自定义实例 还是 生成8个基准实例全集
MODE = "benchmark"            # "benchmark" or "single"
# MODE = "realworld"

# 单个实例参数（仅当 MODE="single" 时使用）
DAYS = 7
TASKS_PER_DAY = 300           # {150, 300}
STAFF = 30
TIGHTNESS = 0.9               # {0.6, 0.9}
SKILLING = 1.0                # {0.3, 1.0}

# 输出前缀（自动创建文件夹）
PREFIX = "D:/Documents/CPLEX/integrated staffrostering and task assignment problem/isrta_bench/ISRTA"
SEED = 1
WRITE_CSV = False
# ============================================================


# ---------- 工具函数 ----------
def intervals_overlap(a_start, a_end, b_start, b_end):
    return a_start < b_end and b_start < a_end

MIN_SHIFTS_PER_TASK = 4   # 你想要的下界 K

def count_overlaps_for_interval(day, start_min, end_min, shift_info):
    """统计在给定 day,start,end 下，有多少个班次与它重叠。"""
    cnt = 0
    for _, (dd, s_start, s_len, _) in shift_info.items():
        if dd != day:
            continue
        if intervals_overlap(start_min, end_min, s_start, s_start + s_len):
            cnt += 1
    return cnt

def choose_start_with_min_shifts(day, length, shift_info,
                                 day_start_min, day_end_min,
                                 min_shifts_per_task=MIN_SHIFTS_PER_TASK,
                                 step_min=15, rnd=None):
    """
    在 [day_start_min, day_end_min-length] 内寻找起始时间 start，
    使得该任务至少与 min_shifts_per_task 个班次重叠。
    """
    if rnd is None:
        import random as _random
        rnd = _random.Random()

    candidates = []
    best_start, best_cnt = day_start_min, -1

    for start in range(day_start_min, day_end_min - length + 1, step_min):
        end = start + length
        cnt = count_overlaps_for_interval(day, start, end, shift_info)
        if cnt > best_cnt:
            best_cnt, best_start = cnt, start
        if cnt >= min_shifts_per_task:
            candidates.append(start)

    if candidates:
        return rnd.choice(candidates)  
    else:
        return best_start


def maximal_cliques_interval_graph(intervals):
    # intervals: list of (t_id, start, end) on a line
    pts = sorted({p for _, s, e in intervals for p in (s, e)})
    for a, b in zip(pts, pts[1:]):
        mid = (a + b) // 2
        if mid != a and mid != b:
            pts.append(mid)
    pts = sorted(set(pts))
    candidates = []
    for t in pts:
        active = {tid for tid, s, e in intervals if s < e and s <= t < e}
        if active:
            candidates.append(active)
    unique, seen = [], set()
    for c in candidates:
        f = frozenset(c)
        if f not in seen:
            seen.add(f)
            unique.append(c)
    maximal = []
    for c in unique:
        if not any(c < d for d in unique):
            maximal.append(c)
    return maximal


def synthesize_instance_realworld(
    days: int = 7,
    tasks_per_day: int = 200,
    staff: int = 30,
    tightness: float = 0.9,
    skilling: float = 0.6,
    seed: int = 0,
    r1_start_hour: float = 4.0,     
    r1_end_hour: float   = 24.0,    
    r2_min_hours: float  = 5.0,     # R2: min shift length
    r3_max_hours: float  = 8.0,     # R3: max shift length
    r4_step_hours: float = 1.0,     # R4: shift start time interval (more shifts -> smaller step)
    max_shifts_per_day: int = 10,   # 每天最多使用的班次数
    min_emp_per_task: int = 10      # 每个任务至少能被这么多人干
):
    
    rnd = random.Random(seed)

    # --- 索引 ---
    D = list(range(days))
    E = list(range(staff))
    T, S = [], []

    # ============================
    # 1. 生成班次 (shifts) —— R1–R4
    # ============================
    shift_info = {}
    sid = 0
    for d in D:
        # 候选起点：4:00, 5:00, ..., < r1_end_hour
        cand_starts = list(np.arange(r1_start_hour,
                                     r1_end_hour - r2_min_hours + 1e-9,
                                     r4_step_hours))
        # 如果候选太多，只取前 max_shifts_per_day 个（可以按需要换成随机选）
        cand_starts = cand_starts[:max_shifts_per_day]

        for h in cand_starts:
            dur_h = rnd.uniform(r2_min_hours, r3_max_hours)   # [5h,8h] 之间随机
            start = int(round(h * 60))
            length = int(round(dur_h * 60))
            end = start + length
            if end > r1_end_hour * 60:
                continue  # 不跨到第二天

            shift_info[sid] = (d, start, length, 0.0)  # c_s 暂时占位
            S.append(sid)
            sid += 1

    # 每天的班次集合 S_d
    S_d = {d: [] for d in D}
    for s, (dd, _, _, _) in shift_info.items():
        S_d[dd].append(s)

    # ============================
    # 2. 生成任务 (tasks) 并按 tightness 缩放
    # ============================
    task_info = {}
    tid = 0
    for d in D:
        # 原始时长：60–180分钟，更长更容易形成大团
        # raw_durations = [rnd.randint(60, 180) for _ in range(tasks_per_day)]
        raw_durations = [rnd.randint(25, 95) for _ in range(tasks_per_day)]
        raw_sum = sum(raw_durations)
        total_shift_min_day = sum(dur for s, (dd, _, dur, _) in shift_info.items() if dd == d)
        target_sum = tightness * total_shift_min_day
        scale = (target_sum / raw_sum) if raw_sum > 0 else 1.0
        durations = [max(10, int(round(x * scale))) for x in raw_durations]

        for k in range(tasks_per_day):
            length = durations[k]
            day_start = int(r1_start_hour * 60)
            day_end   = int(r1_end_hour * 60)
            latest_start = max(day_start, day_end - length)
            start = rnd.randint(day_start, latest_start)
            end   = start + length

            #  取消任务成本：ρ_t = 10000 + variable part (依赖时长)
            cancel_cost = 10000 + 2.0 * length   # ≥ 10000，且随时长增加
            task_info[tid] = (d, start, end, cancel_cost)
            T.append(tid)
            tid += 1

    # 任务成本：对应 z_t 目标系数
    c_t = {t: float(task_info[t][3]) for t in T}

    # 3. 构造 S_t（任务-班次覆盖关系）
    S_t = {t: [] for t in T}
    for t in T:
        d_t, s_t, e_t, _ = task_info[t]
        for s in S:
            d_s, s_start, s_len, _ = shift_info[s]
            if d_s == d_t and intervals_overlap(s_t, e_t, s_start, s_start + s_len):
                S_t[t].append(s)


    cover_size = {s: 0 for s in S}
    for t, shifts in S_t.items():
        for s in shifts:
            cover_size[s] += 1

    max_cover = max(cover_size.values()) if cover_size else 1

    c_s = {}
    for s in S:
        ratio = cover_size[s] / max_cover  # ∈[0,1]
        # 映射到 [1,9]，连续型，保证严格单调
        c_s[s] = 1.0 + 8.0 * ratio + 1e-4 * rnd.random()


    # 5. 员工-任务可行性 (skilling)
    group_of_task = {t: (t % 3) for t in T}
    task_counts = [0, 0, 0]
    for t in T:
        task_counts[group_of_task[t]] += 1
    total_tasks = sum(task_counts)

    if skilling >= 0.99:
        # skilling = 1.0，全能工人
        group_of_emp = {e: 0 for e in E}
        E_t = {t: E[:] for t in T}
    else:
        # 先按任务占比分配员工到 3 个技能组
        base = [max(1, int(staff * (c / total_tasks))) for c in task_counts]
        diff = staff - sum(base)
        ratios = [
            (task_counts[g] / total_tasks if total_tasks > 0 else 0.0) - (base[g] / staff)
            for g in range(3)
        ]
        order = sorted(range(3), key=lambda g: ratios[g], reverse=True)
        for i in range(abs(diff)):
            g = order[i % 3]
            base[g] += 1 if diff > 0 else -1
            if base[g] < 1:
                base[g] = 1

        group_of_emp, idx = {}, 0
        for g in range(3):
            for _ in range(base[g]):
                if idx < len(E):
                    group_of_emp[E[idx]] = g
                    idx += 1
        while idx < len(E):
            g = max(range(3), key=lambda gg: task_counts[gg])
            group_of_emp[E[idx]] = g
            idx += 1


        if skilling <= 0.35:
            task_to_emp_groups = {0: {0}, 1: {1}, 2: {2}}
        else:  # 0.35 < skilling < 0.99，大约 0.6
            task_to_emp_groups = {
                0: {0, 1},
                1: {1, 2},
                2: {2, 0},
            }

        # 生成 E_t
        E_t = {}
        for t in T:
            g_t = group_of_task[t]
            allowed_groups = task_to_emp_groups[g_t]
            E_t[t] = [e for e in E if group_of_emp[e] in allowed_groups]

        # 再保证每个任务至少有 min_emp_per_task 个可行员工
        for t in T:
            if len(E_t[t]) < min_emp_per_task:
                # 先从“允许的技能组”里补
                allowed_groups = task_to_emp_groups[group_of_task[t]]
                same_groups = [
                    e for e in E
                    if group_of_emp[e] in allowed_groups and e not in E_t[t]
                ]
                need = min_emp_per_task - len(E_t[t])
                take = same_groups[:need]
                if len(take) < need:
                    others = [e for e in E if e not in E_t[t] and e not in take]
                    take += others[: (need - len(take))]
                E_t[t] = list(set(E_t[t] + take))

    # 6. 构造 MC_e（互斥团，放大 clique）
    MC_e = {e: [] for e in E}
    for e in E:
        intervals = []
        for t in T:
            if e in E_t[t]:
                d_t, s_t, e_t, _ = task_info[t]
                abs_start = d_t * 24 * 60 + s_t
                abs_end = d_t * 24 * 60 + e_t
                intervals.append((t, abs_start, abs_end))
        MC_e[e] = maximal_cliques_interval_graph(intervals) if intervals else []

    return E, T, D, S, E_t, S_t, S_d, MC_e, c_t, c_s, task_info, shift_info


def randomize_cliques(cliques, task_info, T,
                      p_split=0.25, p_drop=0.20, p_noise=0.10):
    new_cliques = []

    for c in cliques:
        c = set(c)

        # --- ① 大团随机拆分 ---
        if len(c) > 10 and random.random() < p_split:
            k = len(c) // 2
            c1 = set(random.sample(list(c), k))
            c2 = c - c1
            new_cliques.extend([c1, c2])
            continue

        # --- ② 随机裁剪 drop（让分布更稀疏） ---
        drop_k = int(len(c) * p_drop)
        if drop_k > 0:
            c = set(random.sample(list(c), len(c) - drop_k))

        # --- ③ 随机扩张 noise（加入一些近似冲突任务）---
        noise_candidates = []
        for t in T:               # 直接遍历任务编号
            d, s, e, _ = task_info[t]
            for tt in c:
                d2, s2, e2, _ = task_info[tt]
                if d == d2 and abs(s - s2) < 40:   # 40 分钟以内视为“时间很近的任务”
                    noise_candidates.append(t)
                    break
        if noise_candidates:
            add_k = int(len(noise_candidates) * p_noise)
            add_set = set(random.sample(noise_candidates, add_k))
            c |= add_set

        new_cliques.append(c)

    # 去除空集
    new_cliques = [c for c in new_cliques if len(c) > 1]

    return new_cliques

# ---------- 基准实例生成 ----------
def synthesize_instance_benchmark(
    days: int,
    tasks_per_day: int,
    staff: int,
    tightness: float,
    skilling: float,
    seed: int = 0,
    shift_len_min: int = 8*60,        # 8小时
    day_start_hour: int = 6,          # 任务开始最早 06:00
    day_end_hour: int = 22,           # 任务最晚结束 22:00
    # 每隔 0.35 小时起一个 8h 班，覆盖 06:00–22:00
    shift_starts_hours = tuple(np.arange(6, 22, 0.35)),
    min_shifts_per_task: int = 4      # 约束(4) 中每个任务至少 4 个可行班次
):
    """
    生成 airport-style 基准实例：
    - 班次集合 S：每天若干起始时刻的“班次原型”；多个员工可以选同一个班次
    - 紧张度 tightness：通过缩放任务时长，使每日任务总时长精确匹配
    - skilling 控制每个任务可行员工的多少：
        skilling=0.3 → 每个任务平均约 30% 员工可行
        skilling=0.6 → 每个任务平均约 60% 员工可行
        skilling=1.0 → 所有员工都可执行所有任务
    """
    rnd = random.Random(seed)

    # --- 索引集 ---
    D = list(range(days))       # days
    E = list(range(staff))      # employees
    T = []                      # tasks
    S = []                      # shifts (prototype per day)

    # ========================
    # 1. 生成班次 S, S_d
    # ========================
    shift_info = {}
    sid = 0
    for d in D:
        for h in shift_starts_hours:
            start = int(round(h * 60))
            shift_info[sid] = (d, start, shift_len_min, 0.0)  # c_s 先置0
            S.append(sid)
            sid += 1

    S_d = {d: [] for d in D}
    for s, (dd, _, _, _) in shift_info.items():
        S_d[dd].append(s)

    # 每日供给的班次时长 = staff × 8h
    daily_shift_capacity_min = staff * shift_len_min

    # ========================
    # 2. 生成任务 T, task_info
    # ========================
    task_info = {}
    tid = 0
    for d in D:
        # 先生成粗略时长（偏向30-90min），再统一缩放到目标总量
        raw_durations = [rnd.randint(25, 95) for _ in range(tasks_per_day)]
        raw_sum = sum(raw_durations)
        target_sum = tightness * daily_shift_capacity_min
        scale = target_sum / raw_sum if raw_sum > 0 else 1.0
        durations = [max(10, int(round(x * scale))) for x in raw_durations]

        for k in range(tasks_per_day):
            length = durations[k]
            day_start = int(day_start_hour * 60)
            day_end   = int(day_end_hour * 60)

            # 选一个起点，使其至少与 K 个班次重叠（不行就退而求其次）
            start = choose_start_with_min_shifts(
                day=d,
                length=length,
                shift_info=shift_info,
                day_start_min=day_start,
                day_end_min=day_end,
                min_shifts_per_task=MIN_SHIFTS_PER_TASK,
                rnd=rnd
            )
            end = start + length

            # 取消成本：>=10000
            cancel_cost = 10000 + 2.0 * length
            task_info[tid] = (d, start, end, cancel_cost)
            T.append(tid)
            tid += 1

    # z_t 的成本
    c_t = {t: float(task_info[t][3]) for t in T}

    # ========================
    # 3. 任务-班次覆盖 S_t
    # ========================
    S_t = {t: [] for t in T}
    for t in T:
        d_t, s_t, e_t, _ = task_info[t]
        for s in S:
            d_s, s_start, s_len, _ = shift_info[s]
            if d_s == d_t and intervals_overlap(s_t, e_t, s_start, s_start + s_len):
                S_t[t].append(s)

    # ========================
    # 4. 班次成本 c_s（1~9，单调递增 & 去列占优）
    # ========================
    cover_size = {s: 0 for s in S}
    for t, shifts in S_t.items():
        for s in shifts:
            cover_size[s] += 1

    max_cover = max(cover_size.values()) if cover_size else 1
    c_s = {}
    for s in S:
        ratio = cover_size[s] / max_cover  # ∈ [0,1]
        c_s[s] = 1.0 + 8.0 * ratio         # 1.0 ~ 9.0，人力成本个位数

    def prune_dominated_shifts(D, S, S_d, S_t, c_s, task_info, shift_info, eps=1e-9):
        # 计算每个 shift 的覆盖集合
        T_by_day = {d: [] for d in D}
        for t, (dd, _, _, _) in task_info.items():
            T_by_day[dd].append(t)

        cover = {}
        for s, (d, s_start, s_len, _) in shift_info.items():
            tasks = []
            for t in T_by_day[d]:
                _, ts, te, _ = task_info[t]
                if intervals_overlap(ts, te, s_start, s_start + s_len):
                    tasks.append(t)
            cover[s] = set(tasks)

        keep = set(S)
        for d in D:
            shifts = S_d[d]
            for s1 in list(shifts):
                if s1 not in keep:
                    continue
                for s2 in shifts:
                    if s1 == s2 or s2 not in keep:
                        continue
                    # 真子集 + 成本不便宜 → s1 被 s2 支配
                    if cover[s1] < cover[s2] and c_s[s1] >= c_s[s2] - eps:
                        keep.discard(s1)
                        break

        S_new = [s for s in S if s in keep]
        S_d_new = {d: [s for s in S_d[d] if s in keep] for d in D}
        S_t_new = {t: [s for s in sts if s in keep] for t, sts in S_t.items()}
        c_s_new = {s: c for s, c in c_s.items() if s in keep}
        shift_info_new = {s: shift_info[s] for s in S_new}
        return S_new, S_d_new, S_t_new, c_s_new, shift_info_new

    S, S_d, S_t, c_s, shift_info = prune_dominated_shifts(
        D, S, S_d, S_t, c_s, task_info, shift_info
    )

    # ========================
    # 5. 技能结构 E_t（关键改动：真正用 skilling）
    # ========================
    E_t = {t: [] for t in T}
    if skilling >= 0.999:
        # skilling = 1.0 → 全能工人
        for t in T:
            E_t[t] = E[:]          # 所有员工都能执行所有任务
    else:
        K = 10                     # 每个任务至少 K 个可行员工
        for t in T:
            # 先按概率 skilling 随机挑员工
            for e in E:
                if rnd.random() < skilling:
                    E_t[t].append(e)
            # 避免一个都没有
            if len(E_t[t]) == 0:
                E_t[t].append(rnd.choice(E))
            # 再强制补到至少 K 个（或所有员工用完）
            while len(E_t[t]) < K and len(E_t[t]) < len(E):
                e = rnd.choice(E)
                if e not in E_t[t]:
                    E_t[t].append(e)

    # ========================
    # 6. 重新构造 S_t, S_d（用剪枝后的 S）
    # ========================
    S_t = {t: [] for t in T}
    for t in T:
        d_t, s_t, e_t, _ = task_info[t]
        for s in S:
            d_s, s_start, s_len, _ = shift_info[s]
            if d_s == d_t and intervals_overlap(s_t, e_t, s_start, s_start + s_len):
                S_t[t].append(s)

    S_d = {d: [] for d in D}
    for s in S:
        d_s, *_ = shift_info[s]
        S_d[d_s].append(s)

    # ========================
    # 7. 构造 MC_e（互斥团，带随机扰动）
    # ========================
    MC_e = {e: [] for e in E}
    for e in E:
        intervals = []
        for t in T:
            if e in E_t[t]:
                d_t, s_t, e_t, _ = task_info[t]
                abs_start = d_t * 24 * 60 + s_t
                abs_end   = d_t * 24 * 60 + e_t
                intervals.append((t, abs_start, abs_end))
        cliques = maximal_cliques_interval_graph(intervals)
        MC_e[e] = randomize_cliques(cliques, task_info, T)

    return E, T, D, S, E_t, S_t, S_d, MC_e, c_t, c_s, task_info, shift_info


def check_column_dominance(T, S_d, task_info, shift_info, c_s):
    print("\n=== Checking Column Dominance (Shift Columns) ===")
    dominated_pairs = []

    # 同一天内部做比较
    for d, shifts in S_d.items():
        for s1 in shifts:
            d1, start1, len1, _ = shift_info[s1]
            cover1 = {t for t in T
                      if task_info[t][0] == d1 and
                         task_info[t][1] < start1 + len1 and
                         task_info[t][2] > start1}

            for s2 in shifts:
                if s1 == s2:
                    continue

                d2, start2, len2, _ = shift_info[s2]
                cover2 = {t for t in T
                          if task_info[t][0] == d2 and
                             task_info[t][1] < start2 + len2 and
                             task_info[t][2] > start2}

                # 检查列占优条件
                if cover1.issubset(cover2) and c_s[s1] >= c_s[s2]:
                    dominated_pairs.append((s1, s2, c_s[s1], c_s[s2]))

    if dominated_pairs:
        print(f"Found {len(dominated_pairs)} dominated shift-columns:")
        for s1, s2, c1, c2 in dominated_pairs[:20]:
            print(f"  shift {s1} dominated by shift {s2}   c_s={c1:.2f}>{c2:.2f}")
        if len(dominated_pairs) > 20:
            print("  ... (more omitted)")
    else:
        print("No dominated columns found.")
    print("=== End Check ===\n")


# ---------- 模型（与你原来一致） ----------
def build_model(E, T, D, S, E_t, S_t, S_d, MC_e, c_t, c_s, name="ISRTA"):
    prob = pulp.LpProblem(name, pulp.LpMinimize)
    z = {t: pulp.LpVariable(f"z_{t}", lowBound=0, upBound=1, cat="Continuous") for t in T}
    # z = {t:  pulp.LpVariable(f"z_{t}", 0, 1, cat="Binary")}
    # x_et = { (e,t):  pulp.LpVariable(..., cat="Binary") }
    # x_eds = { (e,d,s):  pulp.LpVariable(..., cat="Binary") }  # 新增三指标
    lens_St = [len(S_t[t]) for t in T]
    max_mc  = [max((len(c) for c in MC_e[e]), default=0) for e in E]
    lens_Et = [len(E_t[t]) for t in T]
    print("min |S_t| =", min(lens_St), "max |S_t| =", max(lens_St))  
    print(f"max clique per employee: mean={np.mean(max_mc):.2f}, max={max(max_mc)}")
    print(f"|E_t|: mean={np.mean(lens_Et):.2f}, max={max(lens_Et)}")


    x_et = {}
    for e in E:
        for t in T:
            v = pulp.LpVariable(f"x_{e}_{t}", cat="Binary")
            if e not in E_t[t]:
                prob += (v == 0)
            x_et[(e, t)] = v
    x_es = {(e, s): pulp.LpVariable(f"x_{e}_s{s}", cat="Binary") for e in E for s in S}
    w_ed = {(e, d): pulp.LpVariable(f"w_{e}_d{d}", cat="Binary") for e in E for d in D}
    u_e  = {e: pulp.LpVariable(f"u_{e}", cat="Binary") for e in E}
    # 目标
    prob += (
        pulp.lpSum(c_t[t] * z[t] for t in T) +
        pulp.lpSum(c_s[s] * x_es[(e, s)] for e in E for s in S)
    )
    # 任务覆盖
    for t in T:
        prob += (pulp.lpSum(x_et[(e, t)] for e in E) + z[t] >= 1)
    # 员工时间冲突（互斥团）
    for e in E:
        for mc in MC_e[e]:
            prob += (pulp.lpSum(x_et[(e, t)] for t in mc) <= 1)
    # 任务-班次链接
    for e in E:
        for t in T:
            St = S_t.get(t, [])
            if len(St) == 0:
                prob += (-x_et[(e, t)] >= 0)
            else:
                prob += (-x_et[(e, t)] + pulp.lpSum(x_es[(e, s)] for s in St) >= 0)
    # 日班次计数
    for e in E:
        for d in D:
            Sd = S_d.get(d, [])
            prob += (pulp.lpSum(x_es[(e, s)] for s in Sd) - w_ed[(e, d)] == 0)
    # 是否启用员工
    H = len(D)
    for e in E:
        prob += (H * u_e[e] - pulp.lpSum(w_ed[(e, d)] for d in D) >= 0)
    return prob


# ---------- 输出 ----------
def write_csv_instance(prefix, T, S, E, D, task_info, shift_info, E_t, S_t, S_d, MC_e):
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    with open(f"{prefix}_tasks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id","day","start_min","end_min","c_t"])
        for t in T:
            d, s, e, c = task_info[t]
            w.writerow([t, d, s, e, c])
    with open(f"{prefix}_shifts.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["shift_id","day","start_min","duration","c_s"])
        for s in S:
            d, start, dur, c = shift_info[s]
            w.writerow([s, d, start, dur, c])
    for e, cliques in MC_e.items():
        with open(f"{prefix}_cliques_e{e}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["clique_index","tasks"])
            for i, c in enumerate(cliques): w.writerow([i," ".join(map(str,sorted(c)))])


def run_instance(days, tasks_per_day, staff, tightness, skilling, idx):
    E,T,D,S,E_t,S_t,S_d,MC_e,c_t,c_s,task_info,shift_info = synthesize_instance_benchmark(
        days, tasks_per_day, staff, tightness, skilling, SEED
    )
    check_column_dominance(T, S_d, task_info, shift_info, c_s)
    prob = build_model(E,T,D,S,E_t,S_t,S_d,MC_e,c_t,c_s,f"ISRTA_{idx}")
    mps_path=f"{PREFIX}_{idx}.mps"; os.makedirs(os.path.dirname(mps_path) or ".", exist_ok=True)
    prob.writeMPS(mps_path)
    if WRITE_CSV:
        write_csv_instance(f"{PREFIX}_{idx}",T,S,E,D,task_info,shift_info,E_t,S_t,S_d,MC_e)
    print(f"[{idx}] 生成 MPS 完成 → {mps_path}")
    print(f"    |D|={len(D)}, |T|={len(T)}, |E|={len(E)}, |S|={len(S)}")


def main():
    os.makedirs(os.path.dirname(PREFIX) or ".", exist_ok=True)

    if MODE == "single":
        idx = f"C_{DAYS}_{TASKS_PER_DAY}_{TIGHTNESS}_{SKILLING}"
        E,T,D,S,E_t,S_t,S_d,MC_e,c_t,c_s,task_info,shift_info = synthesize_instance_benchmark(
            DAYS, TASKS_PER_DAY, STAFF, TIGHTNESS, SKILLING, SEED
        )
        prob = build_model(E,T,D,S,E_t,S_t,S_d,MC_e,c_t,c_s,f"ISRTA_{idx}")
        mps_path=f"{PREFIX}_{idx}.mps"
        os.makedirs(os.path.dirname(mps_path) or ".", exist_ok=True)
        prob.writeMPS(mps_path)
        print(f"[{idx}] → benchmark 模式生成完成")
        return

    elif MODE == "realworld":
        days = 7
        staff = 30
        for tau in (0.6, 0.9):
            for skill in (0.3, 0.6, 1.0):
                idx = f"C_{days}_{TASKS_PER_DAY}_{tau}_{skill}"
                E,T,D,S,E_t,S_t,S_d,MC_e,c_t,c_s,task_info,shift_info = synthesize_instance_realworld(
                    days=days, tasks_per_day=TASKS_PER_DAY, staff=staff,
                    tightness=tau, skilling=skill, seed=SEED
                )
                prob = build_model(E,T,D,S,E_t,S_t,S_d,MC_e,c_t,c_s,f"ISRTA_{idx}")
                mps_path=f"{PREFIX}_{idx}.mps"
                os.makedirs(os.path.dirname(mps_path) or ".", exist_ok=True)
                prob.writeMPS(mps_path)
                print(f"[{idx}] → real-world 模式生成完成 |D|={len(D)}, |T|={len(T)}, |E|={len(E)}, |S|={len(S)}")
        return

    else:  # benchmark 模式
        days = 7
        staff = 30
        tpd_values = np.arange(300, 300 + 1, 50, dtype=int)
        for tpd in tpd_values:
            for tau in (0.6, 0.9):
                for skill in (0.3, 0.6, 1.0):
                    idx = f"C_{days}_{tpd}_{tau}_{skill}"
                    run_instance(days, tpd, staff, tau, skill, idx)


if __name__ == "__main__":
    main()