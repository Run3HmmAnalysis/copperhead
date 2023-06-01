import matplotlib.pyplot as plt
import mplhep as hep

style = hep.style.CMS
style["mathtext.fontset"] = "cm"
style["mathtext.default"] = "rm"
plt.style.use(style)
import numpy as np
import pandas as pd

feat_names = {
    "dimuon_mass": "$m_{\mu\mu}$",
    "dimuon_pt": "$p_\mathrm{T}^{\mu\mu}$",
    "dimuon_pt_log": "$\log p_\mathrm{T}^{\mu\mu}$",
    "dimuon_eta": "$\eta_{\mu\mu}$",
    "dimuon_pisa_mass_res": "$\sigma_{m_{\mu\mu}}$",
    "dimuon_pisa_mass_res_rel": "$\sigma_{m_{\mu\mu}}/m_{\mu\mu}$",
    "dimuon_cos_theta_cs_pisa": r"$\cos \theta_\mathrm{CS}$",
    "dimuon_phi_cs_pisa": "$\phi_\mathrm{CS}$",
    "jet1_pt": "$p_\mathrm{T}^{j_1}$",
    "jet1_eta": "$\eta_{j_1}$",
    "jet1_phi": "$\phi_{j_1}$",
    "jet1_qgl": "$\mathrm{QGL}_{j_1}$",
    "jet2_pt": "$p_\mathrm{T}^{j_2}$",
    "jet2_eta": "$\eta_{j_2}$",
    "jet2_phi": "$\phi_{j_2}$",
    "jet2_qgl": "$\mathrm{QGL}_{j_2}$",
    "jj_mass": "$m_{jj}$",
    "jj_mass_log": "$\log m_{jj}$",
    "jj_dEta": "$\Delta\eta_{jj}$",
    "rpt": "$R_{p_\mathrm{T}}$",
    "ll_zstar_log": "$\log z^{*}$",
    "mmj_min_dEta": "$\min\Delta\eta_{\mu\mu,j}$",
    "nsoftjets5": "$N_{p_\mathrm{T}>5~\mathrm{GeV}}^\mathrm{soft}$",
    "htsoft2": "$H_\mathrm{T}^\mathrm{soft}$",
    "year": "$\mathrm{Year}$",
}

data = pd.read_pickle("plots/feat_imp.pkl")
data["feat_name"] = data.index
data["feat_name"] = data.feat_name.map(feat_names)
data = data.reindex(np.array(list(feat_names.keys()) + ["total"]))
data.loc[
    data.dnn_shuffle > data.loc["total", "significance"], "dnn_shuffle"
] = data.loc["total", "significance"]
data = data.sort_values(by="significance", ascending=False)
data = data.sort_values(by="dnn_shuffle", ascending=True)
print(data)
fig = plt.figure()
fig, ax = plt.subplots()
fig.set_size_inches(15, 9)

opts = {
    "color": "black",
    "linewidth": 2,
    "marker": "^",
    "ms": 9,
}

# f_ = lambda x: 100*(x-1.82726)/1.82726
# f = lambda x: x/100*1.82726+1.82726

# sign_improv = 100*(np.array(data["sign"])-1.82726)/1.82726

ax2 = ax.twinx()
max_sign = data.loc["total", "significance"]
count_exp = data.loc["total", "dnn_shuffle"]

# data = data[data.index!="total"]
data = data[~data.index.isin(["total", "jj_mass_log", "dimuon_pt_log"])]
bar = ax.bar(
    data.feat_name,
    data.significance,
    label="Single feature significance",
    **{"edgecolor": "black"},
)
line1 = ax.plot(
    [-0.5, len(data) - 0.5],
    [max_sign, max_sign],
    label="Total DNN significance: $1.82\sigma$",
    **{"color": "green", "linewidth": 3},
)
line2 = ax.plot(
    [-0.5, len(data) - 0.5],
    [count_exp, count_exp],
    label="Counting experiment for $m_{\mu\mu}\in[115, 135]~\mathrm{GeV}$: $0.52\sigma$",
    **{"color": "red", "linewidth": 3},
)
line3 = ax.plot(
    data.feat_name,
    data.dnn_shuffle,
    label="DNN significance if one feature is shuffled",
    **{"color": "purple", "linewidth": 3},
)
line4 = ax2.plot(
    data.feat_name,
    data.mean_grad,
    label="Mean gradient in the input layer of DNN",
    **{"color": "orange", "linewidth": 3},
)

ax.yaxis.set_ticks_position("left")

entries = line2 + line1 + line3 + [bar] + line4
labels = [e.get_label() for e in entries]

ax.legend(
    entries,
    labels,
    prop={"size": "xx-small"},
    loc="lower left",
    bbox_to_anchor=(0.5, 0.365),
)
fs = 20
ax.set_xlabel("DNN input", fontsize=fs)
ax.xaxis.set_label_coords(1, -0.15)
ax.set_ylabel("Expected significance (VBF)", fontsize=fs)
ax2.set_ylabel("Mean gradient in the input layer of DNN", fontsize=fs)
ax2.set_xlabel("")
# ax2.get_yaxis().get_offset_text().set_position((1.07, -1))
ax2.get_yaxis().offsetText.set_visible(False)
ax2.text(
    1.062,
    1.045,
    r"$\times$1e-6",
    transform=ax2.transAxes,
    horizontalalignment="right",
    verticalalignment="top",
    fontsize=fs,
)
ax.set_ylim(0, 1.95)
ax2.set_ylim(0, 1.95e-6)
# ax.tick_params(axis='both', which='major', labelsize=fs-2)
# ax2.tick_params(axis='both', which='major', labelsize=fs-2)

ax.set_xticklabels(data.feat_name, rotation=90)
# plt.xticks(rotation=90)
ax.grid(True)
hep.cms.label(ax=ax, data=False, label="Preliminary", year="", rlabel="", fontsize=16)
fig.tight_layout()
fig.savefig(f"plots/feat_imp.png")
fig.savefig(f"plots/feat_imp.pdf")
