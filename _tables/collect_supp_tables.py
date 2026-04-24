from pathlib import Path
import shutil
import math
import re
import pandas as pd

# directory of this file
cd = Path(__file__).parent
# main project directory
pdir = cd.parent
# deriv directory
pdir_deriv = pdir / "data_deriv"
# results directory
pdir_results = pdir / "results"

# dictionary linking output table to source tables
dct = {
    "Tab. S1 - PET reference": {
        "path": pdir_deriv / "reference" / "pet" / "metadata.csv",
        "legend": """
            Metadata for the 25 PET nuclear imaging maps used as neurobiological reference atlases.\\n
            group: neurotransmitter system or functional category; target: receptor or transporter target; tracer: radioligand used; 
            metric: quantitative PET metric (e.g., BPnd, SUVR, CMRglu); sample_size: total number of subjects; 
            n_female: number of female subjects; age_mean: mean age in years; age_sd: standard deviation of age; 
            age_min: minimum age; age_max: maximum age; publication: first-author and year of primary publication; 
            doi: DOI(s) of associated publications; license: data license.
        """
    },
    "Tab. S2 - Sample data": {
        "path": pdir_deriv / "pheno" / "sample_characteristics.csv", 
        "legend": """
            Sample characteristics across all cohorts included in this study.\\n
            Variable: sample characteristic; n: number of subjects; gender: sex distribution (F=female, M=male; * all-male sample); 
            age: mean ± standard deviation in years, * data unknown and taken from source publications; mean_fd: mean framewise displacement in mm; 
            gc: global connectivity (mean across all parcel pairs); bmi: body mass index; sestot: socioeconomic status total score; 
            panss_total/pos/neg: PANSS total, positive, and negative subscale scores; 
            apd_exp_months: antipsychotic exposure duration in months; apd_chlor_equiv: antipsychotic dose in chlorpromazine equivalents; 
            nih_totalcogcomp_unadjusted: NIH Toolbox unadjusted total cognitive composite score.\\n
            Cohort abbreviations: HCP-YA: Human Connectome Project Young Adult; YRSP: Yale Resting-state Pupilometry Study; HCP-EP: HCP Early Psychosis; 
            MPH: methylphenidate; SCZ: psychosis patients.
        """
    },
    "Tab. S3 - NEOFC MRI RSN": {
        "path": {
            (parc, stat): pdir_results / "neofc" / "hcp_ya_mri" / f"parc-{parc}_dset-rsn_stat-{stat}_group.csv.gz"
            for parc in ["Schaefer100", "Schaefer200", "Schaefer400"]
            for stat in ["auc", "poly2"]
        },
        "index_levels": ["parc", "stat"],
        "legend": """
            Group-level NEOFC statistics for resting-state network (RSN) reference atlases, rsfMRI, across parcellations and statistics.\\n
            parc: parcellation scheme (Schaefer100/200/400); stat: NEOFC statistic (auc: AUC score, poly2: second-degree polynomial fit coefficient);
            measure: FC measure; connections: connection subset (all: all connections); run: fMRI run number (1 or 2);
            metric: NEOFC test direction (original: AUC+, inverted: AUC−);
            variable: summary statistic (mean: group mean; mean_rz: robust z-score against null; mean_z: z-score against null; 
            std/min/max: standard deviation, minimum, maximum; p: p-value; pz: p-value derived from fitting a normal distribution to the null values (not used); 
            pmeff: Meff-corrected p-value; null_*: null distribution percentiles and descriptives).
            Columns represent RSN reference atlases.
        """
    },
    "Tab. S4 - NEOFC MRI PET": {
        "path": {
            (parc, stat): pdir_results / "neofc" / "hcp_ya_mri" / f"parc-{parc}_dset-pet_stat-{stat}_group.csv.gz"
            for parc in ["Schaefer100", "Schaefer100Subcortical", 
                         "Schaefer200", "Schaefer200Subcortical", 
                         "Schaefer400", "Schaefer400Subcortical"]
            for stat in ["auc", "poly2"]
        },
        "index_levels": ["parc", "stat"],
        "legend": """
            Group-level NEOFC statistics for nuclear imaging reference atlases, rsfMRI, across parcellations and statistics, including subcortical regions.\\n
            parc: parcellation scheme (Schaefer100/200/400, ± Subcortical); stat: NEOFC statistic (auc: AUC score, poly2: second-degree polynomial fit coefficient);
            measure: FC measure; connections: connection subset (all: all connections, nointer: excluding inter-hemispheric connections); run: fMRI run number (1 or 2);
            metric: NEOFC test direction (original: AUC+, inverted: AUC−, delta: AUC+ − AUC−);
            variable: summary statistic (as in Tab. S3).
            Columns represent nuclear imaging reference atlases.
        """
    },
    "Tab. S5 - Meta-analysis": {
        "path": pdir_results / "replicability" / "neofc_mri_metap.csv",
        "legend": """
            Weighted Stouffer meta-analysis results for NEOFC AUC scores across six rsfMRI cohorts.\\n
            parc: parcellation scheme (Schaefer200, Schaefer200Subcortical); metric: NEOFC test direction (original: AUC+, inverted: AUC−);
            map: nuclear imaging reference atlas; Z: weighted Stouffer Z-statistic; p: meta-analytic p-value; pmeff: Meff-corrected p-value.
        """
    },
    "Tab. S6 - Test-retest": {
        "path": pdir_results / "neofc" / "hcp_ya_mri" / "parcs-ALL_dset-pet_retest.csv",
        "legend": """
            Test-retest reliability of NEOFC scores between fMRI run 1 and run 2 (HCP-YA).\\n
            parc: parcellation scheme; measure: FC measure; connections: connection subset (all: all connections, nointer: excluding inter-hemispheric);
            metric: NEOFC test direction (original: AUC+, inverted: AUC−, delta: AUC+ − AUC−); stat: NEOFC statistic (auc, poly2); map: nuclear imaging reference atlas.\\n
            ICC(2,1)/ICC(2,k)/ICC(3,1)/ICC(3,k): intraclass correlation coefficients (single/average measures, two-way random/mixed models);
            ICC: ICC value; F: F-statistic; df1/df2: degrees of freedom; pval: p-value; CI95%: 95% confidence interval.
            WCV: within-subject coefficient of variation; pval: p-value for WCV.
        """
    },
    "Tab. S7 - Reproducibility (ICC)": {
        "path": pdir_results / "replicability" / "neofc_mri_replic_icc.csv",
        "legend": """
            ICC-based reproducibility of group-level NEOFC AUC profiles across cohorts.\\n
            datasets: cohort combination included (All: all six cohorts, Without MEG: excluding HCP-YA MEG); parc: parcellation scheme;
            metric: NEOFC test direction (original: AUC+, inverted: AUC−); Type: ICC type; Description: ICC model description;
            ICC: ICC value; F: F-statistic; df1/df2: degrees of freedom; pval: p-value; CI95%: 95% confidence interval.
        """
    },
    "Tab. S8 - Reproducibility (rho)": {
        "path": pdir_results / "replicability" / "neofc_mri_replic_corr.csv",
        "legend": """
            Pairwise Spearman correlations of group-level NEOFC AUC profiles (robust z-scores) across cohorts.\\n
            parc: parcellation scheme; metric: NEOFC test direction (original: AUC+, inverted: AUC−);
            cohort1/cohort2: cohort pair being compared; Spearman's rho: Spearman correlation coefficient.
        """
    },
    "Tab. S9 - NEOFC MEG": {
        "path": {
            parc: pdir_results / "neofc" / "hcp_ya_meg" / f"parc-{parc}_stat-auc_group.csv.gz"
            for parc in ["Schaefer100", "Schaefer200", "Schaefer400"]
        },
        "index_levels": ["parc"],
        "legend": """
            Group-level NEOFC AUC statistics for nuclear imaging reference atlases, MEG, across parcellations.\\n
            parc: parcellation scheme (Schaefer100/200/400); measure: MEG FC measure (aec: amplitude envelope correlation, aecorth: orthogonalized AEC);
            connections: connection subset (all: all connections); fqband: MEG frequency band;
            metric: NEOFC test direction (original: AUC+, inverted: AUC−);
            variable: summary statistic (as in Tab. S3).
            Columns represent nuclear imaging reference atlases.
        """
    },
    "Tab. S10 - MRI-MEG profiles": {
        "path": pdir_results / "comp_mri_meg" / "comp-profile_level-group_parc-Schaefer200.csv",
        "legend": """
            Pairwise Spearman correlations between group-level NEOFC AUC profiles (robust z-scores) across MRI and MEG modalities and frequency bands.\\n
            A/B: profile being compared, encoded as modality-fqband_metric (e.g., mri_original: MRI AUC+, meg-beta_inverted: MEG beta-band AUC−);
            rho: Spearman correlation coefficient; p: p-value.
        """
    },
    "Tab. S11 - MRI-MEG per atlas": {
        "path": pdir_results / "comp_mri_meg" / "comp-subjects_parc-Schaefer200.csv",
        "legend": """
            Per-atlas Spearman correlations between individual MRI and MEG NEOFC AUC scores across subjects.\\n
            metric: NEOFC test direction (original: AUC+, inverted: AUC−); map: nuclear imaging reference atlas;
            fqband: MEG frequency band (delta, theta, alpha, beta, lgamma, hgamma);
            rho: Spearman correlation coefficient; p: p-value.
        """
    },
    "Tab. S12 - NEOFC (lower pct)": {
        "path": {
            (parc, pct): pdir_results / "neofc" / "hcp_ya_mri" / f"parc-{parc}_dset-petaucthresh_stat-auc<={pct}_group.csv.gz"
            for parc in ["Schaefer200", "Schaefer200Subcortical"]
            for pct in range(5, 100, 5)
        },
        "index_levels": ["parc", "pct"],
        "legend": """
            Sensitivity analysis assessing whether AUC effects are driven by the highest-density atlas regions.
            parc: parcellation scheme (Schaefer200, Schaefer200Subcortical); pct: maximum percentile threshold included in AUC calculation (5–90);
            all other columns as in Tab. S4.
        """
    },
    "Tab. S13 - NEOFC NET-covariates": {
        "path": pdir_results / "neofc" / "hcp_ya_mri" / f"parc-Schaefer200_dset-cov_stat-auc_group.csv.gz",
        "legend": """
            Sensitivity analysis testing whether the NET AUC+ effect is driven by spatially confounding factors (Schaefer200 parcellation).
            Columns represent covariate maps (t1t2: T1/T2 ratio; saaxis: sensory-association axis; gm: gray matter probability; csf: cerebrospinal fluid probability;
            veins/arteries: cerebrovascular probability maps; histogradient1/2: BigBrain histological gradient maps;
            microgradient1/2: microstructural gradient maps; funcgradient1/2: functional gradient maps)
            and NET NEOFC results after regressing each covariate (NET-[covariate]) as well as the unregressed baseline (NET);
            all other columns as in Tab. S4.
        """
    },
    "Tab. S14 - NET associations": {
        "path": pdir_results / "physio" / "physio_stats.csv",
        "legend": """
            Linear mixed model (LMM) results for associations between NET AUC+ scores and physiological variables (HCP-YA and YRSP cohorts).\\n
            dataset: cohort; variable: physiological outcome (hrv_rmssd: HRV root mean square of successive differences;
            hrv_madnn: HRV median absolute deviation of NN intervals; pui: pupil unrest index;
            lf_s/_r: low-frequency power of the pupil signal, frequency sympathetic/respiration-weighted; hf_p: high-frequency power, parasympathetic-weighted;
            parc: parcellation scheme; n_subjects: number of subjects; n_observations: number of observations;
            converged: LMM convergence status; df_model/df_resid: model and residual degrees of freedom;
            beta: standardized LMM beta coefficient for NET AUC+; p: p-value.
        """
    },
    "Tab. S15 - Drug challenges": {
        "path": {
            drug: pdir_results / "drug" / f"comp_drug_{drug}.csv"
            for drug in ["mph", "risp", "ketamine", "midazolam", "ketamine_midazolam"]
        },
        "index_levels": ["drug"],
        "legend": """
            Linear mixed model (LMM) Wald χ² test results for drug challenge effects on NEOFC AUC scores separately for the 25 nuclear imaging reference maps.\\n
            drug: drug challenge (mph: methylphenidate, risp: risperidone, ketamine, midazolam, ketamine_midazolam: combined ketamine/midazolam session comparison);
            parc: parcellation scheme (Schaefer200, Schaefer200Subcortical); metric: NEOFC test direction (original: AUC+, inverted: AUC−);
            map: nuclear imaging reference atlas; effect: LMM effect tested (treat: treatment effect, session: session effect, treat*session: treatment × session interaction);
            converged: LMM convergence status; df_model/df_resid: model and residual degrees of freedom;
            beta: standardized LMM beta coefficient; chi2: Wald χ² statistic; p: p-value; pmeff: Meff-corrected p-value.
        """
    },
    "Tab. S16 - Clinic covariates": {
        "path": pdir_results / "clinic" / "comp_confounds.tsv",
        "legend": """
            Type II ANCOVA results for group differences (CTRL vs. PSY) in potential confounding variables, covarying for site.\\n
            parc: parcellation scheme; dv: dependent variable (mean_fd: mean framewise displacement; gc: global connectivity; sa_lambda: sensory-association axis loading);
            n: number of subjects; df_model/df_resid: model and residual degrees of freedom;
            F: F-statistic; p: p-value; np2: partial eta-squared; beta: standardized beta coefficient for group; covariates: covariates included; posthoc: post-hoc comparison result.
        """
    },
    "Tab. S17 - Clinic comparisons": {
        "path": pdir_results / "clinic" / "comp_mapconn.tsv",
        "legend": """
            Type II ANCOVA results for group differences in NEOFC AUC scores separately for 25 nuclear imaging reference maps (site-harmonized data).\\n
            parc: parcellation scheme; metric: NEOFC test direction (original: AUC+, inverted: AUC−);
            sample: subject sample included (all: full sample, off_med: excluding subjects on current antipsychotics);
            between: grouping variable (dx: CTRL vs. PSY diagnosis; dx_apd_current: diagnosis stratified by current antipsychotic dose; 
            dx_apd_lifetime: diagnosis stratified by lifetime antipsychotic use);
            dv: nuclear imaging reference atlas; n: number of subjects; df_model/df_resid: model and residual degrees of freedom;
            F: F-statistic; p: p-value; pmeff: Meff-corrected p-value; beta: standardized beta coefficient for group;
            np2: partial eta-squared; posthoc: Tukey HSD post-hoc comparison result; covariates: covariates included.
        """
    },
    "Tab. S18 - Clinic associations": {
        "path": pdir_results / "clinic" / "assoc_mapconn.tsv",
        "legend": """
            Spearman and partial Spearman correlation results for associations between NEOFC AUC scores and clinical variables in the psychosis group.\\n
            parc: parcellation scheme; metric: NEOFC test direction (original: AUC+, inverted: AUC−);
            dv: clinical outcome (panss_total/pos/neg: PANSS total, positive, and negative subscales; apd_chlor_equiv: antipsychotic dose in chlorpromazine equivalents;
            apd_exp_months: antipsychotic exposure duration in months; nih_totalcogcomp_unadjusted: NIH Toolbox unadjusted total cognitive composite score);
            n: number of subjects; map: nuclear imaging reference atlas;
            rho: Spearman correlation coefficient; p: p-value; pmeff: Meff-corrected p-value;
            rho_partial: partial Spearman correlation coefficient; p_partial: partial correlation p-value; pmeff_partial: Meff-corrected partial p-value;
            covariates: covariates included in partial correlation.
        """
    },
}

# layouts
header_format = {
    "bold": True,
    "text_wrap": True,
    "bottom": 1
}

# create excel file with tables as sheets
with pd.ExcelWriter(cd / "supplementary_tables.xlsx", engine='xlsxwriter') as writer:
    
    # add formats
    workbook = writer.book
    fmt = workbook.add_format(header_format)
    
    # function to load table
    load_table = lambda fp: pd.read_csv(fp) if ".csv" in str(fp) else pd.read_table(fp)
    
    for table_name, source in dct.items():
        
        # load data
        # is a dictionary of file paths (e.g. for neofc tables)
        if isinstance(source["path"], dict):
            assert "index_levels" in source, f"Source for table '{table_name}' must specify 'index_levels' when 'path' is a dictionary."
            df = pd.concat(
                {k: load_table(fp) for k, fp in source["path"].items()}, 
                axis=0, 
                names=source["index_levels"]
            ).droplevel(-1).reset_index()
        # is a single file path
        else:
            df = load_table(source["path"])
        
        # write into excel file
        df.to_excel(writer, sheet_name=table_name, index=False)
        
        # apply formatting
        worksheet = writer.sheets[table_name]
        
        # adjust formatting
        for col_num, value in enumerate(df.columns):
            
            # header formatting
            worksheet.write(0, col_num, value, fmt)         
               
            # column width:
            # set column width based on max length of values in column
            if not "col_width" in source:
                source["col_width"] = "auto"
            if source["col_width"] == "auto":
                max_len = max(df[value].astype(str).map(len).max(), len(value))
                max_len = min(max_len, 30)  # set a maximum width to avoid excessively wide columns
                worksheet.set_column(col_num, col_num, max_len + 1)
            elif source["col_width"] == "header":
                # set column width based on header length only
                width = len(str(value)) + 1
                worksheet.set_column(col_num, col_num, width)
            elif isinstance(source["col_width"], (int, float)):
                # set column width based on preset value
                worksheet.set_column(col_num, col_num, source["col_width"])
        
        # add legend if present
        if "legend" in source:
            legend_row = len(df) + 2  # second row after data

            # legend layout settings
            legend_body = re.sub(r' *\n *', '\n', re.sub(r'[ \t\n]+', ' ', source["legend"]).strip().replace('\\n', '\n'))
            box_width = 700
            font_size = 11
            char_px = 7
            chars_per_line = max(20, int((box_width - 16) / char_px))

            # estimate wrapped line count from explicit line breaks and line length
            wrapped_lines = sum(
                max(1, math.ceil(len(line) / chars_per_line))
                for line in legend_body.splitlines() or [""]
            )
            title_height = 22
            body_height = max(22, wrapped_lines * 16 + 6)
            total_height = title_height + body_height

            # main bordered textbox (contains body only)
            worksheet.insert_textbox(legend_row, 0, f"\n{legend_body}", {
                "width": box_width,
                "height": total_height,
                "font": {"size": font_size, "bold": False},
                "align": {"vertical": "top", "horizontal": "left"},
                "fill": {"color": "#FFFFFF"},
                "line": {"color": "#000000", "width": 1.0}
            })

            # overlay title to make only table name bold
            worksheet.insert_textbox(legend_row, 0, f"{table_name}", {
                "width": box_width,
                "height": title_height,
                "font": {"size": font_size, "bold": True},
                "align": {"vertical": "top", "horizontal": "left"},
                "fill": {"color": "#FFFFFF"},
                "line": {"none": True}
            })