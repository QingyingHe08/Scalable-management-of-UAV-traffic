import numpy as np
import pandas as pd

import pandas as pd


import matplotlib as mpl

# ======================================================
#    Fontsize
# ======================================================
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'


# ---------------- data source ----------------
splits = {'delivery_cq': 'data/delivery_cq-00000-of-00001-465887add76aeabc.parquet', 'delivery_hz': 'data/delivery_hz-00000-of-00001-8090c86f64781f71.parquet', 'delivery_jl': 'data/delivery_jl-00000-of-00001-a4fbefe3c368583c.parquet', 'delivery_sh': 'data/delivery_sh-00000-of-00001-ad9a4b1d79823540.parquet', 'delivery_yt': 'data/delivery_yt-00000-of-00001-cc85c1fcb1d10955.parquet'}

# ---------------- Read and combine ----------------
dfs = []
for _, rel in splits.items():
    d = pd.read_parquet("hf://datasets/Cainiao-AI/LaDe-D/" + rel)
    dfs.append(d)
df = pd.concat(dfs, ignore_index=True)

# ---------------------------
KEY = [
    'city',
    'accept_gps_lng','accept_gps_lat',
    'delivery_gps_lng','delivery_gps_lat',
    'accept_gps_time','delivery_gps_time',
    'ds'
]
missing = set(KEY) - set(df.columns)
if missing:
    raise ValueError(f"缺少必要列：{sorted(missing)}")

df = df.dropna(subset=KEY).copy()

# ---------------- Coordinates washing ----------------
for c in ['accept_gps_lng','accept_gps_lat','delivery_gps_lng','delivery_gps_lat']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.dropna(subset=['accept_gps_lng','accept_gps_lat','delivery_gps_lng','delivery_gps_lat'])
df = df[
    df['accept_gps_lng'].between(-180, 180) &
    df['delivery_gps_lng'].between(-180, 180) &
    df['accept_gps_lat'].between(-90, 90) &
    df['delivery_gps_lat'].between(-90, 90)
].copy()

# ---------------- "MM-DD HH:MM:SS" ----------------
ds_date = pd.to_datetime(df['ds'], errors='coerce')
fmt = '%m-%d %H:%M:%S'
# 去掉可能的前后空格再解析
acc_md = pd.to_datetime(df['accept_gps_time'].astype(str).str.strip(), format=fmt, errors='coerce')
pic_md = pd.to_datetime(df['delivery_gps_time'].astype(str).str.strip(),  format=fmt, errors='coerce')

def build_dt(year, md):
    return pd.to_datetime({
        'year':   year.astype('int64'),
        'month':  md.dt.month,
        'day':    md.dt.day,
        'hour':   md.dt.hour,
        'minute': md.dt.minute,
        'second': md.dt.second
    }, errors='coerce')

acc_dt = build_dt(ds_date.dt.year, acc_md)
pic_dt = build_dt(ds_date.dt.year, pic_md)

ok_time = acc_dt.notna() & pic_dt.notna()
df = df.loc[ok_time].copy()
acc_dt = acc_dt.loc[df.index]
pic_dt = pic_dt.loc[df.index]

# ---------------- Eulidean distance ----------------
k = 111.32
lat_mid = np.deg2rad((df['accept_gps_lat'] + df['delivery_gps_lat']) / 2.0)
dx = (df['delivery_gps_lng'] - df['accept_gps_lng']) * k * np.cos(lat_mid)
dy = (df['delivery_gps_lat'] - df['accept_gps_lat']) * k
df['euclid_km'] = np.hypot(dx, dy)

# ---------------- UAV/duration (mins) ----------------
DRONE_SPEED_KMPH = 45
df['drone_time_min']   = df['euclid_km'] / DRONE_SPEED_KMPH * 60.0
df['current_time_min'] = (pic_dt - acc_dt).dt.total_seconds() / 60.0

# Remove outliers
df = df[(df['current_time_min'] > 0) & (df['euclid_km'] > 1e-6)].copy()

# ---------------- improvemetns ----------------
df['improve_min'] = df['current_time_min'] - df['drone_time_min']
df['improve_pct'] = df['improve_min'] / df['current_time_min'] * 100.0
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# -----------------------------
summary_city = (
    df.groupby('city', dropna=False)
      .agg(
          n=('city','size'),
          euclid_km_mean=('euclid_km','mean'),
          drone_time_mean_min=('drone_time_min','mean'),
          current_time_mean_min=('current_time_min','mean'),
          improve_mean_min=('improve_min','mean'),
          improve_median_min=('improve_min','median'),
          improve_pct_mean=('improve_pct','mean'),
          improve_pct_median=('improve_pct','median'),
          improve_pos_share=('improve_min', lambda s: (s > 0).mean())
      )
      .reset_index()
      .sort_values(['improve_mean_min','improve_pct_mean'], ascending=False)
)

#
summary_city = summary_city.round({
    'euclid_km_mean': 3,
    'drone_time_mean_min': 2, 'current_time_mean_min': 2,
    'improve_mean_min': 2, 'improve_median_min': 2,
    'improve_pct_mean': 2, 'improve_pct_median': 2,
    'improve_pos_share': 3
})














# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ========= 样式辅助：严格贴柱顶的箭头与百分比 =========
def _ensure_ylim(ax, a, b, pad_ratio=0.12):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    top = float(np.nanmax([np.nanmax(a), np.nanmax(b)]))
    bottom = 0.0
    if not np.isfinite(top):
        top = 1.0
    ax.set_ylim(bottom, top * (1 + pad_ratio))

def add_delay_arrow_strict(ax, human_vals, uav_vals, x, width,
                           x_at="center", fontsize=14, line_lw=2.5,
                           head_px=28, head_len_ratio=0.02, head_overshoot_ratio=0.005,
                           label_offset_ratio=0.03):
    """
    “延迟/耗时”型（数值越小越好）：显示相对 human 的下降百分比（-xx.x%）
    - 红色竖线严格覆盖 human 顶 ↔ UAV 顶
    - 箭头尖端略微“越过”对端柱顶，视觉贴合
    - 文本放在 human 顶端上方，显示 -xx.x%
    """
    human_vals = np.asarray(human_vals, dtype=float)
    uav_vals   = np.asarray(uav_vals, dtype=float)

    ymin, ymax = ax.get_ylim()
    if not (np.isfinite(ymin) and np.isfinite(ymax)) or ymax <= ymin:
        _ensure_ylim(ax, human_vals, uav_vals)
    yspan   = ax.get_ylim()[1] - ax.get_ylim()[0]
    ypad    = label_offset_ratio   * yspan
    headlen = head_len_ratio       * yspan
    overs   = head_overshoot_ratio * yspan

    def xi_at(i):
        if x_at == "left":  return x[i] - width/2
        if x_at == "right": return x[i] + width/2
        return x[i]

    for i in range(len(x)):
        h = float(human_vals[i])
        u = float(uav_vals[i])
        xi = xi_at(i)

        # 黑色虚线连接两个柱顶（可选）
        ax.plot([x[i] - width/2, x[i] + width/2], [h, u],
                marker="o", color="black", linestyle="--", linewidth=1.5, zorder=3)

        # 百分比（相对human的下降）
        if np.isfinite(h) and h > 1e-12 and np.isfinite(u):
            pct = (h - u) / h * 100.0
            label = f"-{abs(pct):.1f}%"
        else:
            label = "—"
        ax.text(xi, h + ypad, label, color="red", fontsize=fontsize,
                ha="center", va="bottom", fontweight="bold", zorder=8)

        # 红色竖线：严格覆盖差值
        y0, y1 = (u, h) if u < h else (h, u)
        ax.vlines(xi, y0, y1, colors="red", linewidth=line_lw, zorder=5)

        # 箭头头：终点略微“越过”对端柱顶（关键）
        if u < h:  # 向下箭头：尾部在 u 上方，头端略低于 u
            tail_y = u + headlen
            end_y  = u - overs
        else:      # 向上箭头：尾部在 u 下方，头端略高于 u
            tail_y = u - headlen
            end_y  = u + overs

        head = FancyArrowPatch(
            (xi, tail_y), (xi, end_y),
            arrowstyle='-|>', mutation_scale=head_px,
            color='red', linewidth=0, zorder=7
        )
        head.set_clip_on(False)
        ax.add_patch(head)


# ========= 主函数：城市维度两张图，贴合高级样式 =========
def plot_city_comparison_styled(summary_city,
                                city_order="auto",
                                human_color="burlywood",
                                uav_color="skyblue",
                                alpha_val=0.45,
                                fontsize=25):
    """
    输入：
        summary_city：包含以下列：
          - 'city'
          - 'current_time_mean_min' （人工/现状平均耗时）
          - 'drone_time_mean_min'   （UAV 平均耗时）
          - 'improve_pct_mean'      （平均节省百分比）
        city_order：'auto'（按 improve_pct_mean 降序）或 城市列表
    输出：
        生成两张图：
          1) 平均耗时对比（带红线+贴柱顶箭头+相对下降百分比）
          2) 平均节省百分比（顶端标注）
    """
    df = summary_city.copy()

    # 排序
    if city_order == "auto":
        df = df.sort_values("improve_pct_mean", ascending=False).reset_index(drop=True)
    elif isinstance(city_order, (list, tuple)):
        df = df.set_index("city").reindex(city_order).reset_index()
    # 否则保持原顺序

    cities = df["city"].astype(str).tolist()
    human  = df["current_time_mean_min"].to_numpy(float)
    uav    = df["drone_time_mean_min"].to_numpy(float)
    pct    = df["improve_pct_mean"].to_numpy(float)

    x = np.arange(len(cities))
    width = 0.36

    # ---------------- 图1：平均耗时（现状 vs UAV） ----------------
    plt.rcParams.update({"font.size": fontsize})
    fig1, ax1 = plt.subplots(figsize=(12, 8))

    ax1.bar(x - width/2, human, width, label="Human courier",
            color=human_color, alpha=alpha_val)
    ax1.bar(x + width/2, uav,   width, label="UAV courier",
            color=uav_color, alpha=alpha_val)

    ax1.set_xticks(x)
    ax1.set_xticklabels(cities, rotation=0)
    ax1.set_ylabel("Average delivery time (min)", fontsize=fontsize+2)
    ax1.set_xlabel("City", fontsize=fontsize+2)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    ax1.tick_params(axis="both", labelsize=fontsize)
    ax1.margins(x=0.02)
    _ensure_ylim(ax1, human, uav, pad_ratio=0.18)
    ax1.set_ylim(0, 500)

    # 红线 + 贴柱顶箭头 + 下降百分比
    add_delay_arrow_strict(
        ax1,
        human_vals=human,
        uav_vals=uav,
        x=x, width=width,
        x_at="center",
        head_px=34,
        head_len_ratio=0.02,
        head_overshoot_ratio=0.008,
        fontsize=fontsize
    )

    ax1.legend(fontsize=fontsize, loc="upper right")
    fig1.tight_layout()

    # ---------------- 图2：平均节省百分比 ----------------
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    bars = ax2.bar(x, pct, width=0.5, color="tab:green", alpha=0.5, label="Avg. time saved (%)")

    ax2.set_xticks(x)
    ax2.set_xticklabels(cities, rotation=0)
    ax2.set_ylabel("Average time saved (%)", fontsize=fontsize+2)
    ax2.set_xlabel("City", fontsize=fontsize+2)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)
    ax2.tick_params(axis="both", labelsize=fontsize)
    ax2.margins(x=0.02)

    ymax2 = np.nanmax(pct) if np.isfinite(np.nanmax(pct)) else 10.0
    ax2.set_ylim(0, max(10.0, ymax2 * 1.25))

    # 顶端标注
    for rect, val in zip(bars, pct):
        if np.isfinite(val):
            ax2.text(rect.get_x() + rect.get_width()/2,
                     rect.get_height() * 1.01,
                     f"{val:.1f}%",
                     ha="center", va="bottom", fontsize=fontsize, color="tab:green")

    ax2.legend(fontsize=fontsize, loc="upper right")
    fig2.tight_layout()

    plt.show()


# ================= 使用示例 =================


# ========== 固定顺序并绘图 ==========
desired_order = ["Jilin", "Shanghai", "Hangzhou", "Yantai", "Chongqing"]

# 1) 复制并规范城市名
df_plot = summary_city.copy()
df_plot["city"] = (
    df_plot["city"].astype(str).str.strip()
    .replace({
        "Yanti": "Yantai",         # 常见拼写修正
        "Chungking": "Chongqing",  # 以防旧称
    })
)

# 2) 只保留目标城市（若只想画这5个）
df_plot = df_plot[df_plot["city"].isin(desired_order)].copy()

# 3) 按给定顺序排序（严格按列表顺序）
cat = pd.Categorical(df_plot["city"], categories=desired_order, ordered=True)
df_plot = df_plot.sort_values(["city"], key=lambda s: pd.Categorical(s, categories=desired_order, ordered=True))

# 4) 画图（按你模板的高级样式）
plot_city_comparison_styled(
    df_plot,
    city_order=desired_order,      # 显式传入顺序（函数里也会按此顺序重排）
    human_color="burlywood",
    uav_color="skyblue",
    alpha_val=0.45,
    fontsize=25
)
