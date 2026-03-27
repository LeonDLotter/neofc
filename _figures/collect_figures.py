from pathlib import Path
import shutil


# directory of this file
cd = Path(__file__).parent
# main project directory
pd = cd.parent
# plots directory
pd_plots = pd / "plots"

# dictionary linking output figure to source plot(s)
dct = {
    
    # MAIN -----------------------------------------------------------------------------------------
    "Fig1_method": [
        pd_plots / "method_overview" / "overview_ref-NET_thresh-50.png"
    ],
    "Fig2_overview": [
        pd_plots / "discover" / "mrioverview_parc-Schaefer200_measure-pearson_nodelta.pdf",
        pd_plots / "evaluate" / "posctrloverview_parc-Schaefer200_measure-pearson.pdf",
        pd_plots / "evaluate" / "posctrlsel_parc-Schaefer200_measure-pearson.pdf",
        pd_plots / "discover" / "mrisel_parc-Schaefer200_measure-pearson_run-1.pdf"
    ],
    "Fig3_meg+reprod": [
        pd_plots / "discover" / "megoverview_parc-Schaefer200_measure-aec.pdf",
        pd_plots / "comp_mri_meg" / "indivprofilebeta_parc-Schaefer200.pdf",
        pd_plots / "replicability" / "dsetcorr_parc-Schaefer200.pdf",
    ],
    "Fig4_physio+loo": [
        pd_plots / "reference" / "_reference-NET_parc-Schaefer200Subcortical.png",
        pd_plots / "physio" / "physiooverview_parc-Schaefer200.pdf",
        pd_plots / "loo" / "regionalimportancesign_parc-Schaefer200Subcortical.png",
    ],
    "Fig5_drug": [
        pd_plots / "drug" / "drugeffects_parc-Schaefer200.pdf",
    ],
    "Fig6_clinic": [
        pd_plots / "clinic" / "clinic_parc-Schaefer200.pdf",
    ],
    
}

# delete existing figures but keep manually created composite figures
# find all files with png, pdf and svg endings
f_all = [*cd.glob("*.png"), *cd.glob("*.pdf"), *cd.glob("*.svg")]
# delete als the copied files
for f in f_all:
    if "_sub" in f.name:
        f.unlink()

# copy figures
for out_fig, src_figs in dct.items():
    for i, src_fig in enumerate(src_figs, start=1):
        ext = src_fig.suffix
        shutil.copy(src_fig, cd / f"{out_fig}_sub{i}{ext}")

# special treatment: copy animation figure
shutil.copy(pd_plots / "method_overview" / "overview_ref-NET.gif", cd / "Fig1_method.gif")