# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.ticker import MaxNLocator

# =========================
# Load data
# =========================
path = "Topfiftypercent.csv"
# 为了直接运行测试，建议确保 CSV 文件在同一目录下
df = pd.read_csv(path)

COL_CLASS   = "trip_class"
COL_H_DELAY = "avg_human_delay_min"
COL_H_OTR   = "human_on_time_ratio"
COL_U_DELAY = "avg_uav_delay_min"
COL_U_OTR   = "uav_on_time_ratio"

# =========================
# Helpers
# =========================
def class_key(x):
    s = str(x).lower().strip()
    if "overall" in s:
        return "overall"
    if "short" in s:
        return "short"
    if "medium" in s or "med" in s:
        return "medium"
    if "long" in s:
        return "long"
    return "overall"

def to_percent(series):
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return s
    if s.dropna().max() <= 1.5:
        return s * 100
    return s

# =========================
# Preprocess
# =========================
df[COL_H_DELAY] = pd.to_numeric(df[COL_H_DELAY], errors="coerce")
df[COL_U_DELAY] = pd.to_numeric(df[COL_U_DELAY], errors="coerce")
df[COL_H_OTR]   = to_percent(df[COL_H_OTR])
df[COL_U_OTR]   = to_percent(df[COL_U_OTR])

df["_cls"] = df[COL_CLASS].apply(class_key)

df = df.dropna(subset=[COL_H_DELAY, COL_U_DELAY, COL_H_OTR, COL_U_OTR]).reset_index(drop=True)

# =========================
# Gap metrics
# =========================
df["gap_delay"] = df[COL_H_DELAY] - df[COL_U_DELAY]
df["gap_otr"]   = df[COL_U_OTR] - df[COL_H_OTR]
df["TMG"]       = np.maximum(df["gap_delay"],0) * np.maximum(df["gap_otr"],0)

'''
# =========================
# Layer mapping (修改了这里的顺序)
# =========================
present_classes = list(dict.fromkeys(df["_cls"].tolist()))

# 修改点：将 short 放在最后，这样它的 Z 轴高度最大，显示在最上方
default_order = ["overall", "long", "medium", "short"]

layer_order = [c for c in default_order if c in present_classes]
'''

# =========================
# Layer mapping
# =========================
present_classes = list(dict.fromkeys(df["_cls"].tolist()))

# 修改点：列表里的顺序代表 Z 轴从最低到最高（即视觉上的从下到上）
default_order = ["long", "overall", "medium", "short"]

layer_order = [c for c in default_order if c in present_classes]



layer_gap = 3.5

layer_map = {cls:i*layer_gap for i,cls in enumerate(layer_order)}

df["z_layer"] = df["_cls"].map(layer_map).astype(float)

# Normalize TMG
tmg = df["TMG"].values
if np.max(tmg)>np.min(tmg):
    tnorm = (tmg-np.min(tmg))/(np.max(tmg)-np.min(tmg))
else:
    tnorm = np.zeros_like(tmg)

# =========================
# Plot style
# =========================
plt.rcParams.update({
    "font.size":24,
    "axes.labelsize":28
})

trip_color={
"short":"tab:green",
"medium":"tab:orange",
"long":"tab:purple",
"overall":"tab:blue"
}

# =========================
# Figure
# =========================
fig=plt.figure(figsize=(12,8))
ax=fig.add_subplot(111,projection="3d")

# =========================
# Axis limits
# =========================
xmax=max(df[COL_H_DELAY].max(),df[COL_U_DELAY].max())
xmin=min(df[COL_H_DELAY].min(),df[COL_U_DELAY].min())

ymin=min(df[COL_H_OTR].min(),df[COL_U_OTR].min())
ymax=max(df[COL_H_OTR].max(),df[COL_U_OTR].max())

x_left=max(0,xmin-0.08)
x_right=xmax+0.30

y_low=ymin-3
y_high=ymax+3

z_low=-0.6
z_high=(len(layer_order)-1)*layer_gap+2.0

ax.set_xlim(x_left,x_right)
ax.set_ylim(y_low,y_high)
ax.set_zlim(z_low,z_high)

# =========================
# 拉宽 x y 轴
# =========================
ax.set_box_aspect((3.2,3.2,3.2))

# =========================
# Grid
# =========================
ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
ax.zaxis.set_major_locator(MaxNLocator(nbins=6))

ax.grid(True)

# =========================
# Hide tick labels
# =========================
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

# =========================
# Axis labels
# =========================
ax.set_xlabel("Average delay (min)",labelpad=22)
ax.set_ylabel("On-time ratio (%)",labelpad=22)
ax.set_zlabel("Trip class",labelpad=20)

# =========================
# Plot rectangles only
# =========================
for i,r in df.iterrows():

    cls=r["_cls"]
    c=trip_color.get(cls,"0.5")

    z=float(r["z_layer"])

    hx,hy=float(r[COL_H_DELAY]),float(r[COL_H_OTR])
    ux,uy=float(r[COL_U_DELAY]),float(r[COL_U_OTR])

    gap_d=float(r["gap_delay"])
    gap_r=float(r["gap_otr"])

    improved=(gap_d>0) and (gap_r>0)

    if improved:

        alpha=0.35+0.45*float(tnorm[i])

        verts=[[
        (ux,hy,z),
        (hx,hy,z),
        (hx,uy,z),
        (ux,uy,z)
        ]]

        poly=Poly3DCollection(
        verts,
        facecolors=c,
        edgecolors=c,
        linewidths=1.1,
        alpha=alpha
        )

        ax.add_collection3d(poly)

# =========================
# Layer guide lines
# =========================
for i,cls in enumerate(layer_order):

    z=i*layer_gap

    ax.plot(
    [x_left,x_right],
    [y_low,y_low],
    [z,z],
    color="0.65",
    lw=2,
    alpha=0.28
    )

# =========================
# View angle
# =========================
ax.view_init(elev=15,azim=-120)

# 修复了这里的 right 参数，将其改回 0.98
plt.subplots_adjust(left=0.04, right=0.98, bottom=0.04, top=0.98)

plt.show()
