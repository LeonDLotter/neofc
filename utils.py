from pathlib import Path
import pandas as pd
import numpy as np
from nispace.stats.misc import permute_groups, null_to_p, mc_correction
from pingouin import intraclass_corr
import seaborn as sn
import matplotlib as mpl
from tqdm import tqdm
import gzip
import pickle
from joblib import Parallel, delayed

from mapconn import MapConnNull

# parent of this file
wd = Path(__file__).parent


REF_GROUPS = {
    "Resting-State Networks": [
        "Visual",
        "Auditory",
        "Lateral Somatomotor",
        "Dorsal Somatomotor",
        "Ventral Attention",
        "Dorsal Attention", 
        "Cingulo-Opercular", 
        "Frontoparietal", 
        "Medial Parietal",
        "Occipital Parietal",
        "Salience",
        "Default Mode",
    ],
    "General & Metabolic": [
        "CMRglu",
        "SV2A",
        "HDAC",
        "VMAT2",
    ],
    "Glutamate & GABA": [
        "mGluR5",
        "NMDA",
        "GABAa",
        "GABAa5",
    ],
    "Noradrenaline & Acetylcholine": [
        "NET",
        "A4B2",
        "M1",
        "VAChT",
    ],
    "Dopamine": [
        "FDOPA",
        "D1",
        "D23",
        "DAT",
    ],
    "Serotonin": [
        "5HT1a",
        "5HT1b",
        "5HT2a",
        "5HT4",
        "5HT6",
        "5HTT",
    ],
    "Opioids & Endocannabinoids": [
        "MOR",
        "KOR",
        "CB1",
    ]
}
REF_GROUPS_PET = {k: v for k, v in REF_GROUPS.items() if k != "Resting-State Networks"}

REF_NAMES_ALL = sum(REF_GROUPS.values(), [])
REF_NAMES_RSN = REF_GROUPS["Resting-State Networks"]
REF_NAMES_PET = [r for r in REF_NAMES_ALL if r not in REF_NAMES_RSN]

REF_GROUPNAMES_ALL = list(REF_GROUPS.keys())
REF_GROUPNAMES_PET = REF_GROUPNAMES_ALL[1:]
REF_GROUPS_COLORS = sn.palettes.color_palette(np.array(sn.color_palette("tab10"))[[0,2,4,1,9,8,6]])
REF_COLORS_BY_MAP = {
    m: REF_GROUPS_COLORS[ sum([i if m in v else 0 for i, v in enumerate(REF_GROUPS.values())]) ] 
    for m in REF_NAMES_ALL
}

REF_MATH_NAMES = {
    '5HT1a': '$5\\text{-}HT_{1A}$',
    '5HT1b': '$5\\text{-}HT_{1B}$',
    '5HT2a': '$5\\text{-}HT_{2A}$',
    '5HT4': '$5\\text{-}HT_{4}$',
    '5HT6': '$5\\text{-}HT_{6}$',
    '5HTT': '$5\\text{-}HTT$',
    'A4B2': '$\\alpha_4\\beta_2$',
    'CB1': '$CB1$',
    'CMRglu': '$CMR_{glu}$',
    'D1': '$D_1$',
    'D23': '$D_{2/3}$',
    'DAT': '$DAT$',
    'FDOPA': '$FDOPA$',
    'GABAa': '$GABA_A$',
    'GABAa5': '$GABA_{A5}$',
    'HDAC': '$HDAC$',
    'KOR': '$KOR$',
    'M1': '$M_1$',
    'mGluR5': '$mGluR5$',
    'MOR': '$MOR$',
    'NET': '$NET$',
    'NMDA': '$NMDA$',
    'SV2A': '$SV2A$',
    'VAChT': '$VAChT$',
    'VMAT2': '$VMAT2$', 
    'ADRA': '$ADRA$',
    "t1t2": "$T1/T2$",
    "saaxis": "$SA\\text{-}Axis$",
    "gm": "$GM$",
    "csf": "$CSF$",
    "veins": "$Veins$",
    "arteries": "$Arteries$", 
    "histogradient1": "$HistoGradient1$",
    "histogradient2": "$HistoGradient2$",
    "microgradient1": "$MicroGradient1$",
    "microgradient2": "$MicroGradient2$",
    "funcgradient1": "$FuncGradient1$",
    "funcgradient2": "$FuncGradient2$",
    "NET-t1t2": "$NET \sim T1/T2$", 
    "NET-saaxis": "$NET \sim SA\\text{-}Axis$",
    "NET-gm": "$NET \sim GM$",
    "NET-csf": "$NET \sim CSF$",
    "NET-veins": "$NET \sim Veins$",
    "NET-arteries": "$NET \sim Arteries$", 
    "NET-histogradient1": "$NET \sim HistoGradient1$",
    "NET-histogradient2": "$NET \sim HistoGradient2$",
    "NET-microgradient1": "$NET \sim MicroGradient1$",
    "NET-microgradient2": "$NET \sim MicroGradient2$",
    "NET-funcgradient1": "$NET \sim FuncGradient1$",
    "NET-funcgradient2": "$NET \sim FuncGradient2$",
    "NET-Visual": "$NET \sim Visual$",
    "NET-Auditory": "$NET \sim Auditory$",
    "NET-LateralSomatomotor": "$NET \sim LateralSomatomotor$",
    "NET-DorsalSomatomotor": "$NET \sim DorsalSomatomotor$",
    "NET-VentralAttention": "$NET \sim VentralAttention$",
    "NET-DorsalAttention": "$NET \sim DorsalAttention$",
    "NET-CinguloOpercular": "$NET \sim CinguloOpercular$",
    "NET-FrontoParietal": "$NET \sim FrontoParietal$",
    "NET-ParietalMedial": "$NET \sim ParietalMedial$",
    "NET-ParietalOccipital": "$NET \sim ParietalOccipital$",
    "NET-Salience": "$NET \sim Salience$",
    "NET-DefaultMode": "$NET \sim DefaultMode$",
}

PARCS_ALL = [
    "Schaefer100", 
    "Schaefer100Subcortical",
    "Schaefer200",
    "Schaefer200Subcortical",
    "Schaefer400", 
    "Schaefer400Subcortical",
]
PARC_DEFAULT = "Schaefer200"
PARCS_CX = ["Schaefer100", "Schaefer200", "Schaefer400"]

MEASURES_ALL = ["pearson", "pearsoncut", "shrunkprec"]
MEASURE_DEFAULT = "pearson"

MEG_FQBANDS = ["delta", "theta", "alpha", "beta", "lgamma", "hgamma"]
MEG_MEASURES_ALL = ["aec", "aecorth"]

MEASURES_NICE = {
    "pearson": "Pearson",
    "pearsoncut": "Pearson (cut)",
    "shrunkprec": "ShrunkPrec",
    "aec": "AEC",
    "aecorth": "AECorth",
    "pui": "Pupil unrest index",
    "lf_s": "Low-frequency power\n(sympathetic-weighted)",
    "lf_r": "Low-frequency power\n(respiratory/cardiovagal)",
    "hf_p": "High-frequency power\n(parasympathetic-weighted)",
    "hrv_rmssd": "RMSSD",
    "hrv_madnn": "MADNN",
}
      
      
def get_ref_data(ref="pet", parcs=PARCS_ALL, standardized=True, null=False):
    """
    Get reference data for a given parcellation.
    """
    
    # working directory
    wd = Path.cwd()
    
    # parc and level
    parcs = [parcs] if isinstance(parcs, str) else parcs  
      
    # load data
    data_ref = {}
    for i, parc in enumerate(parcs):  
        
        # load data
        if i == 0: print(f"Loading parcellated {ref} data, standardized={standardized}, null={null}")
        if not null:
            data_ref[parc] = pd.read_csv(
                wd / "data_deriv" / "reference" / ref / f"reference_dset-{ref}_parc-{parc}{'_z' if standardized else ''}.csv", 
                index_col=0
            )
        else:
            with gzip.open(
                wd / "data_deriv" / "reference" / ref / f"reference_dset-{ref}_parc-{parc}_nulls.pkl.gz", "rb") as f:
                data_ref[parc] = pickle.load(f)
        
    if len(data_ref) == 1:
        data_ref = data_ref[list(data_ref.keys())[0]]
    
    # return
    return data_ref


def get_dist_mat(parcs=PARCS_ALL):
    """
    Get distance matrix for a given parcellation.
    """
    
    # working directory
    wd = Path.cwd()
    
    # parc and level
    parcs = [parcs] if isinstance(parcs, str) else parcs

    # load data
    dist_mat = {}
    for parc in parcs:
        dist_mat[parc] = pd.read_csv(
            wd / "parcellation" / f"parc-{parc}.distmat.csv", 
            index_col=0
        )
    if len(dist_mat) == 1:
        dist_mat = dist_mat[list(dist_mat.keys())[0]]
                
    # return
    return dist_mat

def meff_li_ji(R):
    return np.minimum(np.linalg.eigvalsh(R), 1.0).sum()
def bonferroni(p, factor):
    return np.minimum(p * factor, 1.0)
def sidak(p, factor):
    return 1 - (1 - p) ** factor


def get_stats(mfc_dict, levels=["parc", "measure"], stat="auc2", save_path=None, overwrite=False, 
              recalculate_dist_stats=False, get_nulls_stats=True, drop_delta=True):
    if save_path is not None:
        save_path = Path(save_path)
        save_path_indiv = save_path.with_name(save_path.name.replace(".csv", "individual.csv"))
        save_path_group = save_path.with_name(save_path.name.replace(".csv", "group.csv"))
        save_path_indiv_nulls = save_path.with_name(save_path.name.replace(".csv", "individual_nulls.csv"))
        save_path_group_nulls = save_path.with_name(save_path.name.replace(".csv", "group_nulls.csv"))
        if save_path_indiv.exists() and save_path_group.exists() and not overwrite:
            stats_individual = pd.read_csv(save_path_indiv, index_col=[0,1,2,3,4], header=[0])
            stats_group = pd.read_csv(save_path_group, index_col=[0,1,2,3], header=[0])
            if get_nulls_stats:
                stats_indiv_nulls = pd.read_csv(save_path_indiv_nulls, index_col=[0,1,2,3,4], header=[0])
                stats_group_nulls = pd.read_csv(save_path_group_nulls, index_col=[0,1,2,3], header=[0])
                return stats_group, stats_individual, stats_indiv_nulls, stats_group_nulls
            else:
                return stats_group, stats_individual
    
    # function to get stats
    def get_stats_from_mapconn(MapConnNull_instance, stat):
        
        if recalculate_dist_stats:
            if isinstance(MapConnNull_instance, MapConnNull):
                print("Recalculating null stat distributions")
                MapConnNull_instance._mapconn_null_stats_dist_group = None
                MapConnNull_instance._mapconn_null_stats_dist_indiv = None
                MapConnNull_instance._mapconn_delta_null_stats_dist_group = None
                MapConnNull_instance._mapconn_delta_null_stats_dist_indiv = None
                
        group = MapConnNull_instance.get_summary(stats=stat, level="group", reduce_index=False) \
            .droplevel("curve_stat")
        indiv = MapConnNull_instance.get_summary(stats=stat, level="individual", reduce_index=False) \
            .droplevel("curve_stat")
        if drop_delta:
            group = group.query("metric!='delta'")    
            indiv = indiv.query("metric!='delta'")
            
        if get_nulls_stats:
            try: 
                indiv_nulls = pd.concat(
                    [MapConnNull_instance.get_null_stats(stats=stat, multilevel_index=True),
                     MapConnNull_instance.get_null_stats(stats=stat, multilevel_index=True, inverted=True)],
                    axis=0, keys=["original", "inverted"], names=["metric", "null", "id"]
                )
            except:
                indiv_nulls = (
                    MapConnNull_instance.get_null_stats(stats=stat, multilevel_index=True)
                    .assign(metric="original").reset_index().set_index(["metric", "null", "id"])
                )
            group_nulls = indiv_nulls.groupby(["metric", "null"]).mean()
            if drop_delta:
                indiv_nulls = indiv_nulls.query("metric!='delta'")    
                group_nulls = group_nulls.query("metric!='delta'")
        else:
            indiv_nulls, group_nulls = None, None
        return indiv, group, indiv_nulls, group_nulls
    
    def str_to_int(x):
        try:
            return int(x)
        except:
            return x
        
    def get_pmeff(mfc, stats_group):
        # calculate
        meff = meff_li_ji(R=mfc.get_map_data().T.corr())
        for metric in stats_group.index.get_level_values("metric").unique():
            try:
                stats_group.loc[(metric, "pmeff"), :] = sidak(stats_group.loc[(metric, "p"), :], meff)
            except KeyError:
                continue
            
        # sort index
        idc = stats_group.index
        idc_sorted = []
        for metric in idc.get_level_values("metric").unique():
            for idx in [i for i in idc if i[0] == metric and not i[1].startswith("null")]:
                idc_sorted.append(idx)
            for idx in [i for i in idc if i[0] == metric and i[1].startswith("null")]:
                idc_sorted.append(idx)
        stats_group = stats_group.loc[idc_sorted]
        return stats_group
        
    stats_indiv, stats_group, stats_indiv_nulls, stats_group_nulls = {}, {}, {}, {}
    
    ## iterate over levels (parc & measure always exist, most of the time there's one level more)
    if len(levels) > 4:
        raise ValueError(f"more than 4 levels are not supported")
    
    # iterate level 0: measure 
    print(f"Iterating level {levels[0]} (should be measure)")
    for lev0 in mfc_dict.keys():
        
        # iterate level 1: connections
        print(f"Iterating level {levels[1]} (should be connections)")
        for lev1 in mfc_dict[lev0].keys():
            
            # if measure is a dict, iterate over next level
            if isinstance(mfc_dict[lev0][lev1], dict):
                if len(levels) == 2:
                    raise ValueError(f"Dict has at least 2 levels, but only 2 levels are provided")
                
                # iterate flexible level 2
                print(f"Iterating level {levels[2]} (should be run, ses, or treat?)")
                for lev2 in mfc_dict[lev0][lev1].keys():
                    
                    # check if level 2 is a dict
                    if isinstance(mfc_dict[lev0][lev1][lev2], dict):
                        if len(levels) == 3:
                            raise ValueError(f"Dict has at least 3 levels, but only 3 levels are provided")
                        
                        # iterate flexible level 3
                        print(f"Iterating level {levels[3]} (should be run, ses, or treat?)")
                        for lev3 in mfc_dict[lev0][lev1][lev2].keys():
                            
                            # check if level 3 is not a dict
                            if isinstance(mfc_dict[lev0][lev1][lev2][lev3], dict):
                                raise ValueError(f"Only 4 levels are supported, but level 4 is a dict")
                            
                            # key
                            k = (lev0, lev1, lev2, lev3)
                            k = tuple(map(str_to_int, k))
                            # get stats
                            print(f"Getting statistics for: {k}")
                            mfc = mfc_dict[lev0][lev1][lev2][lev3]
                            stats_indiv[k], stats_group[k], stats_indiv_nulls[k], stats_group_nulls[k] = \
                                get_stats_from_mapconn(
                                    MapConnNull_instance=mfc, 
                                    stat=stat,
                                )
                            # add meff pvalue
                            stats_group[k] = get_pmeff(mfc, stats_group[k])
                    
                    # if not a dict, get stats
                    else:
                        # key
                        k = (lev0, lev1, lev2)
                        k = tuple(map(str_to_int, k))
                        # get stats
                        print(f"Getting statistics for: {k}")
                        mfc = mfc_dict[lev0][lev1][lev2]
                        stats_indiv[k], stats_group[k], stats_indiv_nulls[k], stats_group_nulls[k] = \
                            get_stats_from_mapconn(
                                MapConnNull_instance=mfc, 
                                stat=stat,
                        )
                        # add meff pvalue
                        stats_group[k] = get_pmeff(mfc, stats_group[k])    
                    
            # if not a dict, get stats
            else:
                # key
                k = (lev0, lev1)
                k = tuple(map(str_to_int, k))
                # get stats
                print(f"Getting statistics for: {k}")
                mfc = mfc_dict[lev0][lev1]
                stats_indiv[k], stats_group[k], stats_indiv_nulls[k], stats_group_nulls[k] = \
                    get_stats_from_mapconn(
                        MapConnNull_instance=mfc, 
                        stat=stat,
                )
                # add meff pvalue
                stats_group[k] = get_pmeff(mfc, stats_group[k])
                    
    # concat
    stats_group = pd.concat(stats_group, names=levels+["metric", "variable"]).astype(np.float32)
    stats_indiv = pd.concat(stats_indiv, names=levels+["metric", "variable", "id"]).astype(np.float32)
    if get_nulls_stats:
        stats_indiv_nulls = pd.concat(stats_indiv_nulls, names=levels+["metric", "null", "id"]).astype(np.float16)
        stats_group_nulls = pd.concat(stats_group_nulls, names=levels+["metric", "null"]).astype(np.float16)
    
    # save
    if save_path is not None:
        stats_individual.to_csv(save_path_indiv)
        stats_group.to_csv(save_path_group)
        if get_nulls_stats:
            stats_indiv_nulls.to_csv(save_path_indiv_nulls)
            stats_group_nulls.to_csv(save_path_group_nulls)

    if get_nulls_stats:
        return stats_group, stats_indiv, stats_indiv_nulls, stats_group_nulls
    else:
        return stats_group, stats_indiv
    
    
def load_sac_gc(sample, parcs=["Schaefer200", "Schaefer200Subcortical"], index_special=["run"], suffix=""):
    if isinstance(parcs, str):
        parcs = [parcs]
    if isinstance(index_special, str):
        index_special = [index_special]
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix
    fd = wd / "data_deriv" / "connectomes" / sample
    fn = f"parc-%s_sac{suffix}.csv"
    df = pd.concat([
        pd.read_csv(fd / (fn % parc)).assign(parc=parc).set_index(["parc", "measure", *index_special, "sub"])
        for parc in parcs
    ], axis=0)
    return df


def load_pickled_mapconn(sample, parcs, dset=None, windowed=False):
    if isinstance(parcs, str):
        parcs = [parcs]
        
    fd = wd / "data_deriv" / "mapconn_pickled" / sample
    fn = f"parc-%s{'_dset-' + dset if dset else ''}{'_window' if windowed else ''}_mapconn.pkl.gz"
    
    mapconn_objects = {}
    for parc in tqdm(parcs, desc="Loading pickled MapConn objects"):
        with gzip.open(fd / (fn % parc), "rb") as f:
            mapconn_objects[parc] = pickle.load(f)
    
    return mapconn_objects


def load_neofc_stats(sample, 
                       parcs=["Schaefer200", "Schaefer200Subcortical"], 
                       stats=["auc", "poly2"], 
                       dset=None, 
                       index_special=["run"],
                       level=["group", "individual"],
                       nulls=False,
                       windowed=False):
    if isinstance(parcs, str):
        parcs = [parcs]
    if isinstance(stats, str):
        stats = [stats]
    if isinstance(level, str):
        level = [level]
    if isinstance(index_special, str):
        index_special = [index_special]
        
    fd = wd / "results" / "neofc" / sample
    fn = {
        "group": f"parc-%s{'_dset-' + dset if dset else ''}{'_window' if windowed else ''}_stat-%s_group{'_nulls' if nulls else ''}.csv.gz",
        "individual": f"parc-%s{'_dset-' + dset if dset else ''}{'_window' if windowed else ''}_stat-%s_individual{'_nulls' if nulls else ''}.csv.gz"
    }
    
    out = ()
    for l in level:
        print(f"Loading {l} stats: {str(fd / fn[l]) % ('...', '...')}")
        df = (
            pd.concat([
                pd.read_csv(fd / (fn[l] % (parc, stat))).assign(parc=parc, stat=stat) 
                for parc in parcs 
                for stat in stats
            ])
            #.replace({"observed": "original", "inverse": "inverted"}) # not necessary after adjustement in mapconn
            .set_index(["parc", "measure", "connections"] + index_special + ["metric", "stat"] + (["variable"] if not nulls else ["null"]))
        )
        if l == "individual":
            df = df.set_index("id", append=True)
        out += (df,)
    
    return out[0] if len(out) == 1 else out


def calc_wcv(a, b):
    a = np.array(a).squeeze()
    b = np.array(b).squeeze()
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")
    means = (a + b) / 2
    diffs = np.abs(a - b) # absolute differences
    # Within-subject variances
    s2 = diffs**2 / 2
    # Individual CVs
    cv = np.sqrt(s2) / means
    # Within-subject coefficient of variation
    wcv = np.sqrt(np.mean(cv**2))
    return wcv


def calc_retest(stats_individual, save_path=None, overwrite=False, n_perm=1000, n_jobs=-1):
    if save_path is not None:
        if save_path.exists() and not overwrite:
            retest_stats = pd.read_csv(save_path, index_col=[0,1,2,3,4,5], header=[0,1])
            return retest_stats
    na = slice(None)
    
    # check df index names: should be ["parc", "measure", "run"]
    df_index_names = stats_individual.index.names
    if not all(col in df_index_names for col in ["parc", "measure", "run", "metric"]):
        raise ValueError("stats_individual must have index levels with names 'parc', 'measure', 'run', 'metric'.")
    
    # check df column names: should only be "map"
    if isinstance(stats_individual.columns, pd.MultiIndex):
        raise ValueError("stats_individual must have a 1D column index for reference maps.")
    maps = stats_individual.columns.to_list()
    
    # # init dict to store stats
    # retest_stats = {}
    
    # iterate over conditions
    index_to_iterate = (
        stats_individual.index.to_frame()[["parc", "measure", "connections", "metric", "stat"]]
        .drop_duplicates()
    )
    # for _, (parc, measure, connections, metric, stat) in tqdm(
    #     index_to_iterate.iterrows(),
    #     desc="Calculating retest stats",
    #     total=index_to_iterate.shape[0]
    # ):
        
    # parallelization function
    def par_fun(parc, measure, connections, metric, stat):
        
        # empty dict
        dct = {}
        
        # iterate over maps
        for m in maps:
        
            # get data
            df = (
                stats_individual
                .loc[(parc, measure, connections, na, metric, stat, na), m]
                .reset_index(drop=False)
                .rename(columns={m: "value"})
            )
            subs_bothses = [sub for sub in df["id"].unique() if df.query("id==@sub").shape[0]==2]
            df = df.loc[df["id"].isin(subs_bothses)]
            #display(df)
            
            # check if all values are NaN
            if df.value.isna().all():
                continue
            
            # generate permuted indices
            sessions_perm = permute_groups(
                groups=df.run.values, 
                subjects=df.id.values, 
                paired=True, strategy="proportional", seed=42, n_perm=n_perm
            )
            
            # calculate ICC
            icc = intraclass_corr(data=df, targets="id", raters="run", ratings="value")
            
            # calculate WCV
            wcv = calc_wcv(df.query("run==1")["value"], df.query("run==2")["value"])
            arr = df["value"].values
            wcv_null = [
                calc_wcv(arr[ses_perm==1], arr[ses_perm==2])
                for ses_perm in sessions_perm
            ]
            
            # to joint df
            tmp = pd.concat(
                [icc.query("Type == 'ICC2'")[["ICC", "F", "df1", "df2", "pval", "CI95%"]].reset_index(drop=True),
                icc.query("Type == 'ICC2k'")[["ICC", "F", "df1", "df2", "pval", "CI95%"]].reset_index(drop=True),
                icc.query("Type == 'ICC3'")[["ICC", "F", "df1", "df2", "pval", "CI95%"]].reset_index(drop=True),
                icc.query("Type == 'ICC3k'")[["ICC", "F", "df1", "df2", "pval", "CI95%"]].reset_index(drop=True)],
                keys=["ICC(2,1)", "ICC(2,k)", "ICC(3,1)", "ICC(3,k)"], 
                axis=1,
            )
            tmp[("CV", "WCV")] = wcv
            tmp[("CV", "pval")] = null_to_p(wcv, wcv_null, tail="lower")
            
            # store in dict
            dct[parc, measure, connections, metric, stat, m] = tmp
        return dct
            
    # run (returns list of dicts)
    dcts = Parallel(n_jobs=n_jobs)(
        delayed(par_fun)(parc, measure, connections, metric, stat)
        for parc, measure, connections, metric, stat 
        in tqdm(index_to_iterate.itertuples(index=False), total=index_to_iterate.shape[0])    
    )
    
    # concatenate into one dict
    dcts = {k: v for d in dcts for k, v in d.items()}
    
    # to pandas df        
    retest_stats = pd.concat(dcts, names=["parc", "measure", "connections", "metric", "stat", "map"]).droplevel(-1)
    
    # save
    if save_path is not None:
        retest_stats.to_csv(save_path)
        
    return retest_stats


def calc_delta_permutation(stats_individual, n_perm=1000, save_path=None, overwrite=False):
    if save_path is not None:
        save_path = Path(save_path)
        if save_path.exists() and not overwrite:
            return pd.read_csv(save_path, index_col=[0,1,2,3,4])
    
    # get index to iterate
    index_to_iterate = (
        stats_individual.index.to_frame()[["parc", "measure", "connections", "stat"]]
        .drop_duplicates().to_numpy()
    )
    
    # iterate
    dict_results = {}
    for parc, measure, connections, stat in index_to_iterate:
        #print(parc, measure, connections, stat)
        
        # get data
        orig = stats_individual.loc[(parc, measure, connections, "original", stat, slice(None))]
        inv = stats_individual.loc[(parc, measure, connections, "inverted", stat, slice(None))]
        delta = stats_individual.loc[(parc, measure, connections, "delta", stat, slice(None))]

        # concatenated array
        data_concat = np.concatenate([orig, inv], axis=0)

        # generate permuted indices
        subjects = np.array(orig.index.to_list() * 2)
        indexer_true = np.array([0] * orig.shape[0] + [1] * orig.shape[0])
        indexer_perm = permute_groups(
            groups=indexer_true, 
            subjects=subjects, 
            paired=True, strategy="proportional", seed=42, n_perm=n_perm
        )
        
        # calculate delta
        delta_mean = delta.mean(axis=0)
        delta_null_mean = np.zeros((n_perm, delta_mean.shape[0]))
        for i in range(n_perm):
            delta_null_mean[i, :] = ( data_concat[indexer_perm[i]==0] - data_concat[indexer_perm[i]==1] ).mean(axis=0)
        
        # results
        p_values = [null_to_p(delta_mean.loc[m], delta_null_mean[:, i_m], tail="two") 
                    for i_m, m in enumerate(delta_mean.index)]
        meff = meff_li_ji(R=get_ref_data("pet", parc).T.corr())
        print(meff)
        df_results = pd.DataFrame({
            "delta": delta_mean,
            "p": p_values,
            "pmeff": sidak(np.array(p_values), meff),
            "null_mean": delta_null_mean.mean(axis=0),
            "null_std": delta_null_mean.std(axis=0),
            "null_min": delta_null_mean.min(axis=0),
            "null_1%": np.percentile(delta_null_mean, 1, axis=0),
            "null_10%": np.percentile(delta_null_mean, 10, axis=0),
            "null_25%": np.percentile(delta_null_mean, 25, axis=0),
            "null_50%": np.percentile(delta_null_mean, 50, axis=0),
            "null_75%": np.percentile(delta_null_mean, 75, axis=0),
            "null_90%": np.percentile(delta_null_mean, 90, axis=0),
            "null_99%": np.percentile(delta_null_mean, 99, axis=0),
            "null_max": delta_null_mean.max(axis=0),
        }).T
        dict_results[parc, measure, connections, stat] = df_results
    
    # concatenate
    df_results = pd.concat(dict_results, axis=0, names=["parc", "measure", "connections", "stat", "variable"])
    
    # save
    if save_path is not None:
        df_results.to_csv(save_path)
    
    return df_results


def generate_indices(n=256, power=2):
    # Generate indices with more samples at the lower extreme
    indices = np.linspace(0, 1, n)  # Create a linear space between 0 and 1
    indices = indices**power
    indices = (indices * (n - 1))  # Scale to the range 0 to n
    return indices

def merge_cmaps(cmap1, cmap2, center="k", N=256, N_cmap=512, power=2):
    cmap1 = sn.color_palette(cmap1, n_colors=N_cmap)
    cmap2 = sn.color_palette(cmap2, n_colors=N_cmap)
    if power:
        idc = generate_indices(N_cmap, power=power).round(0).astype(int)
    else:
        idc = np.arange(N_cmap)
    colors = np.array(cmap1)[idc].tolist() + [center] + np.array(cmap2)[N_cmap-1-idc[::-1]].tolist()
    return mpl.colors.LinearSegmentedColormap.from_list(name="", colors=colors, N=N)

