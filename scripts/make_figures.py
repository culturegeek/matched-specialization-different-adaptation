"""Generate bilingual manuscript figures from archived confirmatory CSVs.

Plotting intervals are paired/clustered by seed, with directions averaged within
seed. These descriptive figure intervals do not replace preregistered results.
"""
from pathlib import Path
import csv
import io
import json
import zipfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager, FontProperties

ROOT = Path(__file__).resolve().parents[1]
IV = ROOT / "experiments/iv"
VCSV = ROOT / "experiments/v-a/results_confirm/cell_summary.csv"
FOLLOW = ROOT / "experiments/iv-followup/results_confirm"
FIG = ROOT / "generated/figures"
BLUE, ORANGE, GREEN = "#2166AC", "#C65E28", "#397F65"
JAFONT = None

def rows(study, name):
    with (IV / study / "results_confirm" / (name + ".csv")).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))

def groups(data, pred, field):
    out = {}
    for r in data:
        if pred(r):
            out.setdefault(int(r["seed"]), []).append(float(r[field]))
    return {k: float(np.mean(v)) for k, v in out.items()}

def estimate(g):
    a = np.array([g[k] for k in sorted(g)])
    rng = np.random.default_rng(20260903)
    boot = a[rng.integers(0, len(a), size=(5000, len(a)))].mean(1)
    lo, hi = np.quantile(boot, [.025, .975])
    return float(a.mean()), float(lo), float(hi)

def contrast(data, pred, component, first, second):
    a = groups(data, lambda r: pred(r) and r["optimizer_component"] == component and r["route"] == first, "pulse_drop")
    b = groups(data, lambda r: pred(r) and r["optimizer_component"] == component and r["route"] == second, "pulse_drop")
    assert set(a) == set(b) and len(a) == 40
    return estimate({k: a[k] - b[k] for k in a})

def style(lang):
    plt.rcParams.update({"font.family": JAFONT if lang == "ja" else "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "pdf.fonttype": 3 if lang == "ja" else 42, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "savefig.facecolor": "white"})

def save(fig, stem, lang):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}_{lang}.pdf", bbox_inches="tight", pad_inches=.06)
    plt.close(fig)

def headline(lang, b, d):
    ja = lang == "ja"
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)
    labs = ["A精密", "均衡", "B精密"] if ja else ["A precise", "Balanced", "B precise"]
    for i, acq in enumerate(["A_precise", "balanced", "B_precise"]):
        mean, lo, hi = estimate(groups(b, lambda r: r["acquisition"] == acq and r["update"] == "balanced", "final_s_high"))
        ax[0].errorbar(i, mean, yerr=[[mean-lo], [hi-mean]], fmt="o", color=[BLUE,GREEN,ORANGE][i], capsize=4, ms=7)
    ax[0].axhline(0, color=".55", lw=.8)
    ax[0].set(xticks=range(3), xticklabels=labs, ylim=(-.43,.43), ylabel="配分指数 s" if ja else "Allocation index s",
              title="a  更新重みを均等に固定" if ja else "a  Equal update weights")
    ax[0].set_xlabel("形成時の取得条件" if ja else "Acquisition during formation")
    for route, color, label in [("acquisition",BLUE,"取得形成" if ja else "Acquisition-formed"), ("update",ORANGE,"更新形成" if ja else "Update-formed")]:
        vals = [estimate(groups(d, lambda r: r["route"] == route and r["optimizer_mode"] == "carried" and r["protocol"] == "combined_switch", f)) for f in ["start_aligned_s", "pulse_end_aligned_s"]]
        means = np.array([v[0] for v in vals])
        ax[1].errorbar([0,1], means, yerr=np.array([[v[0]-v[1] for v in vals], [v[2]-v[0] for v in vals]]), color=color, marker="o", capsize=4, label=label)
    ax[1].axhline(0,color=".55",lw=.8,ls="--")
    ax[1].set(xticks=[0,1],xticklabels=["摂動前","10更新後"] if ja else ["Before pulse","After 10 updates"],ylabel="元の優位対象に揃えた s" if ja else "s aligned to original favored target", title="b  同じ摂動、異なる応答" if ja else "b  Same pulse, different response")
    ax[1].legend(loc="lower left", frameon=False)
    save(fig,"headline",lang)

def controls(lang, b, c):
    ja = lang == "ja"
    fig, axs = plt.subplots(2,2,figsize=(7.0,5.1), constrained_layout=True)
    a = axs.flat[0]
    arr = np.array([[estimate(groups(b, lambda r: r["acquisition"]==acq and r["update"]==upd,"final_s_high"))[0] for upd in ["A_heavy","balanced","B_heavy"]] for acq in ["A_precise","balanced","B_precise"]])
    a.imshow(arr,cmap="RdBu",vmin=-.45,vmax=.45,aspect="auto")
    for (i,j),v in np.ndenumerate(arr): a.text(j,i,f"{v:+.3f}",ha="center",va="center",color="white" if abs(v)>.29 else "black",fontsize=10)
    a.set(xticks=range(3),yticks=range(3),xticklabels=["A重視","均衡","B重視"] if ja else ["A-heavy","Balanced","B-heavy"],yticklabels=["A精密","均衡","B精密"] if ja else ["A precise","Balanced","B precise"],xlabel="更新配分" if ja else "Update allocation",ylabel="取得" if ja else "Acquisition",title="a  形成終点の配分指数" if ja else "a  Allocation after formation")
    for ax, acq, title in zip(list(axs.flat)[1:],["A_precise","balanced","B_precise"],["b  A精密取得","c  均衡取得","d  B精密取得"] if ja else ["b  A-precise acquisition","c  Balanced acquisition","d  B-precise acquisition"]):
        for origin,color,ls,label in [("A_formed",BLUE,"-","A形成" if ja else "A-formed"),("neutral",GREEN,":","中立" if ja else "Neutral"),("B_formed",ORANGE,"--","B形成" if ja else "B-formed")]:
            steps=sorted({int(r["step"]) for r in c if r["initial_state"]==origin and r["acquisition"]==acq})
            ys=[np.mean([float(r["s_high"]) for r in c if r["initial_state"]==origin and r["acquisition"]==acq and int(r["step"])==t]) for t in steps]
            ax.plot(steps,ys,color=color,ls=ls,label=label,lw=1.5)
        ax.axhline(0,color=".7",lw=.6)
        ax.set(xlabel="更新回数" if ja else "Updates",ylabel="配分指数 s" if ja else "Allocation index s",ylim=(-.46,.46),title=title)
        if acq=="balanced": ax.legend(frameon=False,loc="upper right")
    save(fig,"controls",lang)

def moments(lang, e, u):
    ja = lang == "ja"
    fig, axes=plt.subplots(1,2,figsize=(7.0,2.9),constrained_layout=True)
    result={}
    for ax,data,key,first,second,ylab,title in [(axes[0],e,"route","acquisition","update","取得形成 − 更新形成" if ja else "Acquisition-formed minus update-formed","a  形成経路の比較" if ja else "a  Different formation routes"),(axes[1],u,"weight_history","constant","variable","一定重み − 変動重み" if ja else "Constant minus variable weighting","b  平均更新重みを一致" if ja else "b  Equal mean update weight")]:
        result[key]={}
        for i,component in enumerate(["own","m_other","v_other","reset"]):
            mean,lo,hi=contrast(data,lambda r:r["protocol"]=="combined_switch",component,first,second)
            result[key][component]={"mean":mean,"ci95":[lo,hi]}
            ax.errorbar(i,mean,yerr=[[mean-lo],[hi-mean]],fmt="o",ms=7,capsize=4,color=ORANGE if component=="v_other" else BLUE)
        margin = .02 if key == "route" else .01
        ax.axhspan(-margin, margin, color="#E2E8EB", alpha=.65, zorder=0)
        ax.axhline(0,color=".5",lw=.8)
        ax.set(xticks=range(4),xticklabels=["保持","m交換","v交換","初期化"] if ja else ["Own","m swap","v swap","Reset"],title=title,xlabel="optimizerの操作" if ja else "Optimizer intervention",ylabel=("10更新での特化減少量の差\n"+ylab) if ja else ("Difference in specialization loss\n"+ylab))
        ax.grid(axis="y",alpha=.15)
    save(fig,"moments",lang)
    return result

def three(lang, v):
    ja=lang=="ja"
    fig,axes=plt.subplots(1,2,figsize=(7,3.2),constrained_layout=True)
    for ax,dim in zip(axes,[2,3]):
        base={int(r["seed"]):r for r in v if int(r["bottleneck"])==dim and r["acquisition"]=="balanced" and r["update"]=="balanced"}
        for target,color in zip("ABC",[BLUE,ORANGE,GREEN]):
            rs=[r for r in v if int(r["bottleneck"])==dim and r["acquisition"]==target+"_precise" and r["update"]=="balanced"]
            xy=np.array([[float(r["final_high_q_"+t])-float(base[int(r["seed"])]["final_high_q_"+t]) for t in "AB"] for r in rs])
            ax.scatter(xy[:,0],xy[:,1],color=color,s=10,alpha=.32)
            mean=xy.mean(0)
            ax.annotate("",xy=mean,xytext=(0,0),arrowprops={"arrowstyle":"->","color":color,"lw":1.6})
            ax.plot(*mean,"o",color=color,ms=6)
            ax.annotate(target,mean,xytext=(6,5),textcoords="offset points",color=color,fontweight="bold")
        ax.axhline(0,color=".75",lw=.6);ax.axvline(0,color=".75",lw=.6)
        ax.set(xlabel="Aへの配分利得" if ja else "Allocation gain for A",ylabel="Bへの配分利得" if ja else "Allocation gain for B",title=(f"共有表現の次元 {dim}" if ja else f"Bottleneck dimension {dim}"))
        ax.set_aspect("equal",adjustable="datalim")
        ax.margins(.2)
    save(fig,"three_targets",lang)

def timecourse(lang, route_effects):
    """Show what the follow-up adds: second-moment structure and response timing."""
    ja = lang == "ja"
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), constrained_layout=True)

    modes = ["own", "scalar_v_own", "v_equal_scale", "scalar_v_equal"]
    labels = (["継承Adam", "vを一値化", r"平均$\bar a_v$を一致", r"一値化＋$\bar a_v$一致"] if ja else
              ["Inherited Adam", "Scalar v", r"Equal mean $\bar a_v$", r"Scalar v + equal $\bar a_v$"])
    for i, mode in enumerate(modes):
        row = next(r for r in route_effects if r["mode"] == mode and
                   r["trajectory"] == "continued_opposition" and int(r["step"]) == 10)
        mean, lo, hi = map(float, (row["mean"], row["low"], row["high"]))
        axes[0].errorbar(i, mean, yerr=[[mean-lo], [hi-mean]], fmt="o", ms=6,
                         capsize=4, color=BLUE if mode != "scalar_v_equal" else ORANGE)
    axes[0].axhspan(-.02, .02, color="#E2E8EB", alpha=.65, zorder=0)
    axes[0].axhline(0, color=".5", lw=.8)
    axes[0].set(xticks=range(len(modes)), xticklabels=labels,
                ylabel="10更新での低下量の経路差" if ja else "Route difference in loss\nof specialization at 10 updates",
                title="a  倍率と配置を分ける" if ja else "a  Separating scale and pattern")
    axes[0].tick_params(axis="x", rotation=22)

    for mode, color, marker, label in [
        ("own", BLUE, "o", "継承Adam" if ja else "Inherited Adam"),
        ("reset", ORANGE, "s", "新規Adam" if ja else "Fresh Adam"),
        ("sgd_calibrated", GREEN, "^", "校正SGD" if ja else "Calibrated SGD"),
    ]:
        rs = sorted((r for r in route_effects if r["mode"] == mode and
                     r["trajectory"] == "continued_opposition" and int(r["step"]) > 0),
                    key=lambda r: int(r["step"]))
        x = np.array([int(r["step"]) for r in rs])
        y = np.array([float(r["mean"]) for r in rs])
        lo = np.array([float(r["low"]) for r in rs])
        hi = np.array([float(r["high"]) for r in rs])
        axes[1].plot(x, y, marker=marker, color=color, lw=1.5, ms=4, label=label)
        axes[1].fill_between(x, lo, hi, color=color, alpha=.12)
    axes[1].axhspan(-.02, .02, color="#E2E8EB", alpha=.65, zorder=0)
    axes[1].axhline(0, color=".5", lw=.8)
    axes[1].set(xscale="log", xticks=[10,20,50,100,500],
                xticklabels=[10,20,50,100,500],
                xlabel="反対条件の更新回数" if ja else "Opposing-condition updates",
                ylabel="低下量の経路差" if ja else "Route difference in loss\nof specialization",
                title="b  差が現れる時間が変わる" if ja else "b  The timing of route differences changes")
    axes[1].legend(frameon=False)
    save(fig, "timecourse", lang)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", choices=["en", "ja"], default=["en"])
    parser.add_argument("--japanese-font", type=Path, help="Optional path to a user-installed Japanese font")
    args = parser.parse_args()
    if "ja" in args.languages:
        if not args.japanese_font or not args.japanese_font.is_file():
            parser.error("Japanese output requires --japanese-font pointing to a user-installed CJK font")
        fontManager.addfont(str(args.japanese_font))
        JAFONT = FontProperties(fname=str(args.japanese_font)).get_name()
    b=rows("IV-B","cell_summary");c=rows("IV-C","timecourse");d=rows("IV-D","protocol_summary")
    e=rows("IV-E","protocol_summary");u=rows("IV-E-update-bias-control","protocol_summary")
    with VCSV.open(encoding="utf-8") as f:v=list(csv.DictReader(f))
    with (FOLLOW / "route_effects.csv").open(encoding="utf-8") as f:follow=list(csv.DictReader(f))
    for lang in args.languages:
        style(lang);headline(lang,b,d);controls(lang,b,c);result=moments(lang,e,u);three(lang,v);timecourse(lang,follow)
    (FIG/"reader_figure_audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(f"Generated {5 * len(args.languages)} vector figure PDFs in {FIG}.")
