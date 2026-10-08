import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
INPUT = R"C:\Users\Lenovo\Downloads\New folder (2)\PPG\dual_reading\index_middle\ppg_results_sg\ppg_fusion_output.csv"
INPUT = "ppg_results_sg\ppg_fusion_output.csv"
OUT = "ppg_fusion_results"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(INPUT)
t = df["Elapsed_s"].to_numpy(float)
FS = 1.0 / np.median(np.diff(t))

PAIRS = {
    "IR": ("IR1 Fusion", "IR2 Fusion"),
    "RED": ("RED1 Fusion", "RED2 Fusion")
}

def corr(a,b):
    if np.std(a)<1e-12 or np.std(b)<1e-12:
        return 0.0
    return float(np.corrcoef(a,b)[0,1])

def detect_troughs(x):
    return find_peaks(
        -x,
        distance=max(1,int(FS*60/180)),
        prominence=max(1e-9,0.15*np.std(x))
    )[0]

def get_pulses(x,trough_idx,n=100):
    P=[]; B=[]
    for a,b in zip(trough_idx[:-1],trough_idx[1:]):
        pp=(b-a)/FS
        if 60/180 <= pp <= 60/40:
            s=x[a:b+1]
            if len(s)>=5 and np.all(np.isfinite(s)):
                p=np.interp(
                    np.linspace(0,1,n),
                    np.linspace(0,1,len(s)),
                    s
                )
                p=(p-p.mean())/(p.std()+1e-12)
                P.append(p); B.append((a,b))
    return np.asarray(P),B

CAL_S = 30.0         # initial calibration period (s) used to build the template
PAIR_TOL_S = 0.15    # max trough-time difference for two beats to be "corresponding"

def pair_beats(b1,b2,tol_s=PAIR_TOL_S):
    """Pair beats of sensor 1 and sensor 2 by trough timing (one-to-one,
    nearest start-trough in time within tol_s). Returns list of (i1,i2)."""
    if len(b1)==0 or len(b2)==0:
        return []
    s1=np.array([t[a] for a,_ in b1])
    s2=np.array([t[a] for a,_ in b2])
    pairs=[]; used=set()
    for i,ts in enumerate(s1):
        order=np.argsort(np.abs(s2-ts))
        for j in order:
            if abs(s2[j]-ts)>tol_s:
                break
            if j not in used:
                used.add(int(j)); pairs.append((i,int(j)))
                break
    return pairs

def make_template(c1,c2,cal_s=CAL_S):
    x1=df[c1].to_numpy(float)
    x2=df[c2].to_numpy(float)
    p1,b1=get_pulses(x1,detect_troughs(x1))   # each pulse resampled + mean/SD normalised
    p2,b2=get_pulses(x2,detect_troughs(x2))

    # Keep only beats that lie fully inside the initial calibration period.
    t_end=t[0]+cal_s
    k1=[i for i,(a,b) in enumerate(b1) if t[b]<=t_end]
    k2=[i for i,(a,b) in enumerate(b2) if t[b]<=t_end]
    p1c,b1c=p1[k1],[b1[i] for i in k1]
    p2c,b2c=p2[k2],[b2[i] for i in k2]

    # Pair corresponding beats by timing; average each pair point-by-point.
    pairs=pair_beats(b1c,b2c)
    if len(pairs)==0:
        raise ValueError(
            f"No corresponding beat pairs for {c1}/{c2} in the first {cal_s}s; "
            "increase CAL_S or PAIR_TOL_S.")
    pair_templates=np.array([(p1c[i]+p2c[j])/2.0 for i,j in pairs])

    # Average all pair templates (arithmetic mean, not median) -> final template.
    T=pair_templates.mean(axis=0)
    return (T-T.mean())/(T.std()+1e-12)

def template_matching(c1,c2):
    x1=df[c1].to_numpy(float)
    x2=df[c2].to_numpy(float)
    tr1,tr2=detect_troughs(x1),detect_troughs(x2)
    p1,b1=get_pulses(x1,tr1)
    p2,b2=get_pulses(x2,tr2)
    T=make_template(c1,c2)

    w1=np.zeros(len(x1))
    w2=np.zeros(len(x2))

    # Paper: Wi = rho_i if rho_i >= 0, otherwise 0.
    for (a,b),p in zip(b1,p1):
        w1[a:b+1]=max(corr(p,T),0)
    for (a,b),p in zip(b2,p2):
        w2[a:b+1]=max(corr(p,T),0)

    # Fill the small gaps between pulse intervals.
    for w in (w1,w2):
        nz=np.flatnonzero(w>0)
        if len(nz):
            w[:nz[0]]=w[nz[0]]
            w[nz[-1]+1:]=w[nz[-1]]
            missing=w==0
            if missing.any():
                w[missing]=np.interp(np.flatnonzero(missing),nz,w[nz])

    fused=(w1*x1+w2*x2)/(w1+w2+1e-12)
    return fused,w1,w2,T

def ivw(c1,c2,window_s=10.0,sigma_floor=0.5):
    x1=df[c1].to_numpy(float)
    x2=df[c2].to_numpy(float)
    tr1,tr2=detect_troughs(x1),detect_troughs(x2)
    tt1,tt2=t[tr1],t[tr2]

    w1=np.zeros(len(t))
    w2=np.zeros(len(t))
    rows=[]

    # Paper: Wi = 1/sigma_i, sigma_i = SD of HR in the window.
    for ws in np.arange(t[0],t[-1],window_s):
        we=ws+window_s
        sigmas=[]

        for tt in (tt1,tt2):
            z=tt[(tt>=ws)&(tt<we)]
            pp=np.diff(z)
            pp=pp[(pp>=60/180)&(pp<=60/40)]
            hr=60/pp
            sigma=np.std(hr,ddof=1) if len(hr)>=2 else sigma_floor
            sigmas.append(max(float(sigma),sigma_floor))

        a,b=1/sigmas[0],1/sigmas[1]
        wa,wb=a/(a+b),b/(a+b)
        idx=(t>=ws)&(t<we)
        w1[idx]=wa
        w2[idx]=wb

        rows.append({
            "start_s":ws,"end_s":we,
            f"{c1}_sigma_HR":sigmas[0],
            f"{c2}_sigma_HR":sigmas[1],
            f"{c1}_weight":wa,
            f"{c2}_weight":wb
        })

    fused=w1*x1+w2*x2
    return fused,w1,w2,pd.DataFrame(rows)

def extract_features(x,method,wavelength,template):
    tr=detect_troughs(x)
    P,B=get_pulses(x,tr)
    rows=[]

    for beat,((a,b),p) in enumerate(zip(B,P),1):
        s=x[a:b+1]
        peak=a+int(np.argmax(s))
        amp=float(x[peak]-x[a])
        half=x[a]+0.5*amp
        above=np.where(s>=half)[0]
        width50=(above[-1]-above[0])/FS if len(above)>=2 else np.nan
        pp=(b-a)/FS

        rows.append({
            "method":method,
            "wavelength":wavelength,
            "beat":beat,
            "start_s":t[a],
            "peak_s":t[peak],
            "end_s":t[b],
            "PP_interval_s":pp,
            "Pulse_Rate_BPM":60/pp,
            "Pulse_Amplitude":amp,
            "Rise_Time_s":(peak-a)/FS,
            "Fall_Time_s":(b-peak)/FS,
            "Width_50pct_s":width50,
            "Pulse_Area":np.trapz(s-s.min(),dx=1/FS),
            "Template_Correlation":corr(p,template)
        })
    return pd.DataFrame(rows)

def thirty_second_features(beat_df):
    rows=[]
    edges=np.arange(t[0],t[-1]+30,30)

    for (method,wavelength),g in beat_df.groupby(["method","wavelength"]):
        for ws,we in zip(edges[:-1],edges[1:]):
            q=g[(g.start_s>=ws)&(g.start_s<we)]
            if len(q)==0:
                continue

            pp=q.PP_interval_s.to_numpy()
            hr=q.Pulse_Rate_BPM.to_numpy()

            rows.append({
                "method":method,"wavelength":wavelength,
                "start_s":ws,"end_s":we,
                "n_beats":len(q),
                "mean_PP_s":pp.mean(),
                "median_PP_s":np.median(pp),
                "SDNN_PP_s":np.std(pp,ddof=1) if len(pp)>1 else np.nan,
                "RMSSD_PP_s":np.sqrt(np.mean(np.diff(pp)**2)) if len(pp)>1 else np.nan,
                "mean_HR_BPM":hr.mean(),
                "median_HR_BPM":np.median(hr),
                "SD_HR_BPM":np.std(hr,ddof=1) if len(hr)>1 else np.nan,
                "mean_pulse_amplitude":q.Pulse_Amplitude.mean(),
                "mean_rise_time_s":q.Rise_Time_s.mean(),
                "mean_fall_time_s":q.Fall_Time_s.mean(),
                "mean_width50_s":q.Width_50pct_s.mean(),
                "mean_template_corr":q.Template_Correlation.mean()
            })
    return pd.DataFrame(rows)

# ---------- FUSION ----------
outputs={}
templates={}
all_beat=[]

for wavelength,(c1,c2) in PAIRS.items():
    tm,tmw1,tmw2,template=template_matching(c1,c2)
    iv,ivw1,ivw2,ivtab=ivw(c1,c2)

    outputs[(wavelength,"TM")]=tm
    outputs[(wavelength,"IVW")]=iv
    templates[wavelength]=template

    df[f"TM_{wavelength}_Fused"]=tm
    df[f"IVW_{wavelength}_Fused"]=iv
    df[f"TM_{wavelength}_W1"]=tmw1
    df[f"TM_{wavelength}_W2"]=tmw2
    df[f"IVW_{wavelength}_W1"]=ivw1
    df[f"IVW_{wavelength}_W2"]=ivw2

    ivtab.to_csv(
        os.path.join(OUT,f"IVW_{wavelength}_window_weights.csv"),
        index=False
    )

    all_beat.append(extract_features(tm,"TM",wavelength,template))
    all_beat.append(extract_features(iv,"IVW",wavelength,template))

df.to_csv(os.path.join(OUT,"ppg_fusion_results.csv"),index=False)

beat_df=pd.concat(all_beat,ignore_index=True)
beat_df.to_csv(os.path.join(OUT,"ppg_fused_beat_features.csv"),index=False)

window_df=thirty_second_features(beat_df)
window_df.to_csv(os.path.join(OUT,"ppg_fused_30s_features.csv"),index=False)

# ---------- COMPARISON ----------
def ten_second_hr(x):
    tr=detect_troughs(x)
    tt=t[tr]
    out=[]
    for ws in np.arange(t[0],t[-1],10):
        we=ws+10
        z=tt[(tt>=ws)&(tt<we)]
        pp=np.diff(z)
        pp=pp[(pp>=60/180)&(pp<=60/40)]
        out.append(np.median(60/pp) if len(pp) else np.nan)
    return np.asarray(out)

consensus={}
for wavelength,(c1,c2) in PAIRS.items():
    consensus[wavelength]=np.nanmedian(
        np.vstack([ten_second_hr(df[c1]),ten_second_hr(df[c2])]),
        axis=0
    )

metrics=[]
for (wavelength,method),x in outputs.items():
    b=extract_features(x,method,wavelength,templates[wavelength])
    cons=consensus[wavelength]
    h=ten_second_hr(x)
    metrics.append({
        "method":method,
        "wavelength":wavelength,
        "beats":len(b),
        "median_HR_BPM":b.Pulse_Rate_BPM.median(),
        "mean_HR_BPM":b.Pulse_Rate_BPM.mean(),
        "SD_HR_BPM":b.Pulse_Rate_BPM.std(),
        "PP_CV":b.PP_interval_s.std()/b.PP_interval_s.mean(),
        "median_template_corr":b.Template_Correlation.median(),
        "mean_template_corr":b.Template_Correlation.mean(),
        "template_corr_10th_percentile":b.Template_Correlation.quantile(.10),
        "10s_consensus_HR_MAE_BPM":np.nanmean(np.abs(h-cons))
    })

metrics=pd.DataFrame(metrics)
metrics.to_csv(os.path.join(OUT,"ppg_fusion_comparison_metrics.csv"),index=False)

# ---------- VISUALISATIONS ----------
# All figures are saved in the same output folder (OUT).
# PNG files are used so they can be directly inserted into reports.

def savefig(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, filename), dpi=300, bbox_inches="tight")
    plt.close()

# 1. Raw PPG signals and the two fused signals for each wavelength.
for wavelength,(c1,c2) in PAIRS.items():
    fig,ax=plt.subplots(figsize=(12,5))
    ax.plot(t,df[c1],label=c1,linewidth=0.8,alpha=0.75)
    ax.plot(t,df[c2],label=c2,linewidth=0.8,alpha=0.75)
    ax.plot(t,outputs[(wavelength,"TM")],
            label=f"{wavelength} TM Fused",linewidth=1.2)
    ax.plot(t,outputs[(wavelength,"IVW")],
            label=f"{wavelength} IVW Fused",linewidth=1.2)
    ax.set_title(f"{wavelength} PPG: Raw Sensors and Fused Signals")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("PPG Amplitude")
    ax.legend()
    ax.grid(True,alpha=0.3)
    savefig(f"{wavelength}_raw_and_fused.png")

# 2. Template-matching weights.
for wavelength,(c1,c2) in PAIRS.items():
    fig,ax=plt.subplots(figsize=(12,5))
    ax.plot(t,df[f"TM_{wavelength}_W1"],label=f"{c1} TM Weight",linewidth=1)
    ax.plot(t,df[f"TM_{wavelength}_W2"],label=f"{c2} TM Weight",linewidth=1)
    ax.set_title(f"{wavelength} Template-Matching Fusion Weights")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Weight")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True,alpha=0.3)
    savefig(f"{wavelength}_TM_weights.png")

# 3. Inverse-variance weights.
for wavelength,(c1,c2) in PAIRS.items():
    fig,ax=plt.subplots(figsize=(12,5))
    ax.plot(t,df[f"IVW_{wavelength}_W1"],label=f"{c1} IVW Weight",linewidth=1)
    ax.plot(t,df[f"IVW_{wavelength}_W2"],label=f"{c2} IVW Weight",linewidth=1)
    ax.set_title(f"{wavelength} Inverse-Variance Fusion Weights")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Weight")
    ax.set_ylim(0,1)
    ax.legend()
    ax.grid(True,alpha=0.3)
    savefig(f"{wavelength}_IVW_weights.png")

# 4. Heart-rate comparison for each wavelength.
for wavelength,(c1,c2) in PAIRS.items():
    series = {
        c1: ten_second_hr(df[c1]),
        c2: ten_second_hr(df[c2]),
        f"{wavelength} TM": ten_second_hr(outputs[(wavelength,"TM")]),
        f"{wavelength} IVW": ten_second_hr(outputs[(wavelength,"IVW")]),
        "Sensor Consensus": consensus[wavelength]
    }
    hr_t = np.arange(len(next(iter(series.values()))))*10 + t[0]

    fig,ax=plt.subplots(figsize=(12,5))
    for label,h in series.items():
        ax.plot(hr_t,h,label=label,marker="o",markersize=2,linewidth=1)
    ax.set_title(f"{wavelength} 10-second Heart-Rate Comparison")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Heart Rate (BPM)")
    ax.legend()
    ax.grid(True,alpha=0.3)
    savefig(f"{wavelength}_10s_HR_comparison.png")

# 5. Summary comparison of the fusion metrics.
for wavelength in PAIRS:
    m=metrics[metrics["wavelength"]==wavelength].copy()
    if not m.empty:
        labels=m["method"].tolist()
        x=np.arange(len(labels))
        width=0.25

        fig,axes=plt.subplots(1,3,figsize=(14,5))

        axes[0].bar(x-width/2,m["median_HR_BPM"],width,label="Median HR")
        axes[0].bar(x+width/2,m["mean_HR_BPM"],width,label="Mean HR")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels)
        axes[0].set_ylabel("BPM")
        axes[0].set_title("Heart Rate")
        axes[0].legend()
        axes[0].grid(True,axis="y",alpha=0.3)

        axes[1].bar(x,m["median_template_corr"],width=0.5)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels)
        axes[1].set_ylabel("Correlation")
        axes[1].set_title("Median Template Correlation")
        axes[1].grid(True,axis="y",alpha=0.3)

        axes[2].bar(x,m["10s_consensus_HR_MAE_BPM"],width=0.5)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(labels)
        axes[2].set_ylabel("MAE (BPM)")
        axes[2].set_title("10-second Consensus HR MAE")
        axes[2].grid(True,axis="y",alpha=0.3)

        fig.suptitle(f"{wavelength} Fusion Method Comparison")
        savefig(f"{wavelength}_fusion_method_comparison.png")

# 6. Combined overview of TM and IVW fused signals.
fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
for i,wavelength in enumerate(PAIRS):
    axes[i].plot(t,outputs[(wavelength,"TM")],label=f"{wavelength} TM",linewidth=1)
    axes[i].plot(t,outputs[(wavelength,"IVW")],label=f"{wavelength} IVW",linewidth=1)
    axes[i].set_ylabel("PPG Amplitude")
    axes[i].set_title(f"{wavelength} Fused PPG")
    axes[i].legend()
    axes[i].grid(True,alpha=0.3)
axes[-1].set_xlabel("Elapsed Time (s)")
savefig("all_fused_signals_overview.png")

print(metrics.to_string(index=False))
print("Saved results and visualisations in:",OUT)