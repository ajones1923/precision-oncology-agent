"""Precision Oncology knowledge graph — targets, therapies, resistance, pathways, biomarkers.

Provides structured domain knowledge for augmenting RAG queries with
curated oncology intelligence. Each helper function returns formatted
context strings that are injected into LLM prompts alongside retrieved
evidence from Milvus collections.

Follows the same pattern as:
  cart_intelligence_agent/src/knowledge.py

Author: Adam Jones
Date: February 2026
"""

from typing import Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════
# 1. ACTIONABLE TARGETS (~40 genes)
# ═══════════════════════════════════════════════════════════════════════

ACTIONABLE_TARGETS: Dict[str, Dict] = {
    "BRAF": {
        "gene": "BRAF",
        "full_name": "B-Raf Proto-Oncogene",
        "cancer_types": ["melanoma", "NSCLC", "colorectal", "thyroid", "hairy cell leukemia"],
        "key_variants": ["V600E", "V600K", "V600D", "V600R", "class II", "class III"],
        "targeted_therapies": ["vemurafenib", "dabrafenib", "encorafenib"],
        "combination_therapies": ["dabrafenib + trametinib", "encorafenib + binimetinib", "encorafenib + cetuximab"],
        "resistance_mutations": ["MEK1/2 mutations", "NRAS activation", "BRAF amplification", "MAP2K1 C121S"],
        "pathway": "MAPK",
        "evidence_level": "A",
        "description": "BRAF V600E is the most common BRAF mutation, found in ~50% of melanomas. "
                       "FDA-approved targeted therapies include BRAF+MEK inhibitor combinations. "
                       "In CRC, BRAF V600E requires triplet therapy (BRAF+MEK+EGFR inhibition).",
    },
    "EGFR": {
        "gene": "EGFR",
        "full_name": "Epidermal Growth Factor Receptor",
        "cancer_types": ["NSCLC", "head and neck", "colorectal", "glioblastoma"],
        "key_variants": ["L858R", "exon 19 deletion", "T790M", "C797S", "exon 20 insertion", "S768I", "L861Q", "G719X"],
        "targeted_therapies": ["osimertinib", "erlotinib", "gefitinib", "afatinib", "dacomitinib", "amivantamab"],
        "combination_therapies": ["osimertinib + chemotherapy", "amivantamab + lazertinib"],
        "resistance_mutations": ["T790M", "C797S", "MET amplification", "HER2 amplification",
                                  "small cell transformation", "BRAF V600E", "PIK3CA mutations"],
        "pathway": "MAPK",
        "evidence_level": "A",
        "description": "EGFR mutations occur in ~15-20% of NSCLC (higher in never-smokers, Asian populations). "
                       "Osimertinib is standard first-line for EGFR-mutant NSCLC. T790M is the most common "
                       "acquired resistance mechanism to 1st/2nd-gen TKIs.",
    },
    "ALK": {
        "gene": "ALK",
        "full_name": "Anaplastic Lymphoma Kinase",
        "cancer_types": ["NSCLC", "anaplastic large cell lymphoma", "neuroblastoma"],
        "key_variants": ["EML4-ALK fusion", "ALK amplification", "F1174L", "R1275Q"],
        "targeted_therapies": ["alectinib", "lorlatinib", "brigatinib", "ceritinib", "crizotinib"],
        "combination_therapies": [],
        "resistance_mutations": ["G1202R", "L1196M", "I1171T/N/S", "compound mutations",
                                  "MET amplification", "EGFR activation"],
        "pathway": "MAPK/PI3K",
        "evidence_level": "A",
        "description": "ALK fusions occur in ~5% of NSCLC. Alectinib is preferred first-line. "
                       "Lorlatinib is effective against most single ALK resistance mutations "
                       "including G1202R. Sequential ALK TKI therapy can extend disease control.",
    },
    "ROS1": {
        "gene": "ROS1",
        "full_name": "ROS Proto-Oncogene 1",
        "cancer_types": ["NSCLC", "cholangiocarcinoma", "glioblastoma"],
        "key_variants": ["CD74-ROS1 fusion", "SLC34A2-ROS1 fusion", "SDC4-ROS1 fusion"],
        "targeted_therapies": ["crizotinib", "entrectinib", "lorlatinib", "repotrectinib"],
        "combination_therapies": [],
        "resistance_mutations": ["G2032R", "D2033N", "L2026M"],
        "pathway": "MAPK/PI3K",
        "evidence_level": "A",
        "description": "ROS1 fusions occur in ~1-2% of NSCLC. Entrectinib and crizotinib are "
                       "first-line options. Repotrectinib shows activity against G2032R resistance.",
    },
    "KRAS": {
        "gene": "KRAS",
        "full_name": "KRAS Proto-Oncogene GTPase",
        "cancer_types": ["NSCLC", "colorectal", "pancreatic"],
        "key_variants": ["G12C", "G12D", "G12V", "G13D", "Q61H"],
        "targeted_therapies": ["sotorasib", "adagrasib"],
        "combination_therapies": ["sotorasib + panitumumab", "adagrasib + cetuximab"],
        "resistance_mutations": ["Y96D", "R68S", "H95D/Q/R", "secondary KRAS mutations",
                                  "MET amplification", "EGFR activation", "RAS-MAPK reactivation"],
        "pathway": "MAPK",
        "evidence_level": "A",
        "description": "KRAS G12C occurs in ~13% of NSCLC and ~3% of CRC. Sotorasib and adagrasib "
                       "are FDA-approved for KRAS G12C-mutant NSCLC. Other KRAS alleles (G12D) are "
                       "targets of emerging therapies including MRTX1133.",
    },
    "HER2": {
        "gene": "ERBB2",
        "full_name": "Human Epidermal Growth Factor Receptor 2",
        "cancer_types": ["breast", "gastric", "NSCLC", "colorectal", "endometrial"],
        "key_variants": ["amplification", "overexpression", "S310F", "exon 20 insertions", "V777L"],
        "targeted_therapies": ["trastuzumab", "pertuzumab", "trastuzumab deruxtecan", "tucatinib",
                                "trastuzumab emtansine", "margetuximab"],
        "combination_therapies": ["trastuzumab + pertuzumab + docetaxel", "tucatinib + trastuzumab + capecitabine"],
        "resistance_mutations": ["PIK3CA mutations", "PTEN loss", "HER2 truncation (p95)", "NRG1 activation"],
        "pathway": "PI3K/MAPK",
        "evidence_level": "A",
        "description": "HER2 amplification defines a breast cancer subtype (~20%). Trastuzumab deruxtecan "
                       "(T-DXd/Enhertu) has shown activity across HER2-amplified and HER2-low cancers. "
                       "HER2 mutations in NSCLC are targetable with T-DXd.",
    },
    "NTRK": {
        "gene": "NTRK1/2/3",
        "full_name": "Neurotrophic Tyrosine Receptor Kinase",
        "cancer_types": ["tissue-agnostic", "secretory breast", "infantile fibrosarcoma", "thyroid",
                          "salivary gland", "NSCLC", "CRC"],
        "key_variants": ["ETV6-NTRK3 fusion", "TPM3-NTRK1 fusion", "LMNA-NTRK1 fusion"],
        "targeted_therapies": ["larotrectinib", "entrectinib", "repotrectinib", "selitrectinib"],
        "combination_therapies": [],
        "resistance_mutations": ["G595R (NTRK1)", "G623R (NTRK3)", "F589L", "xDFG mutations"],
        "pathway": "MAPK/PI3K",
        "evidence_level": "A",
        "description": "NTRK fusions are rare (<1% pan-cancer) but highly actionable. Larotrectinib "
                       "and entrectinib have tissue-agnostic FDA approvals. ORR >75% across tumor types.",
    },
    "RET": {
        "gene": "RET",
        "full_name": "Rearranged during Transfection",
        "cancer_types": ["NSCLC", "medullary thyroid", "papillary thyroid"],
        "key_variants": ["KIF5B-RET fusion", "CCDC6-RET fusion", "M918T", "C634R"],
        "targeted_therapies": ["selpercatinib", "pralsetinib"],
        "combination_therapies": [],
        "resistance_mutations": ["G810R/S/C", "V804M", "Y806C/H"],
        "pathway": "MAPK/PI3K",
        "evidence_level": "A",
        "description": "RET fusions occur in ~1-2% of NSCLC and RET mutations in >90% of medullary "
                       "thyroid cancer. Selpercatinib is the preferred selective RET inhibitor.",
    },
    "MET": {
        "gene": "MET",
        "full_name": "MET Proto-Oncogene (Hepatocyte Growth Factor Receptor)",
        "cancer_types": ["NSCLC", "renal cell", "hepatocellular", "gastric"],
        "key_variants": ["exon 14 skipping", "amplification", "Y1003 mutations"],
        "targeted_therapies": ["capmatinib", "tepotinib", "crizotinib"],
        "combination_therapies": ["osimertinib + savolitinib"],
        "resistance_mutations": ["D1228N/V", "Y1230H/C", "KRAS amplification"],
        "pathway": "MAPK/PI3K",
        "evidence_level": "A",
        "description": "MET exon 14 skipping occurs in ~3-4% of NSCLC. MET amplification is a key "
                       "resistance mechanism to EGFR TKIs. Capmatinib and tepotinib are FDA-approved.",
    },
    "FGFR": {
        "gene": "FGFR1/2/3",
        "full_name": "Fibroblast Growth Factor Receptor",
        "cancer_types": ["bladder/urothelial", "cholangiocarcinoma", "endometrial", "gastric"],
        "key_variants": ["FGFR2 fusions", "FGFR3 S249C", "FGFR3 R248C", "FGFR2 amplification"],
        "targeted_therapies": ["erdafitinib", "futibatinib", "pemigatinib", "infigratinib"],
        "combination_therapies": [],
        "resistance_mutations": ["V565I (gatekeeper)", "N550H/K", "FGFR2 V564F"],
        "pathway": "MAPK/PI3K",
        "evidence_level": "A",
        "description": "FGFR alterations are common in bladder cancer (~20%) and cholangiocarcinoma "
                       "(~15% FGFR2 fusions). Erdafitinib is approved for FGFR-altered urothelial cancer.",
    },
    "PIK3CA": {
        "gene": "PIK3CA",
        "full_name": "Phosphatidylinositol-4,5-Bisphosphate 3-Kinase Catalytic Subunit Alpha",
        "cancer_types": ["breast", "endometrial", "cervical", "head and neck", "colorectal"],
        "key_variants": ["H1047R", "E545K", "E542K", "C420R", "N345K"],
        "targeted_therapies": ["alpelisib", "inavolisib"],
        "combination_therapies": ["alpelisib + fulvestrant", "inavolisib + palbociclib + fulvestrant"],
        "resistance_mutations": ["PTEN loss", "AKT1 E17K", "mTOR mutations"],
        "pathway": "PI3K/AKT/mTOR",
        "evidence_level": "A",
        "description": "PIK3CA mutations occur in ~40% of HR+/HER2- breast cancer. Alpelisib + fulvestrant "
                       "is approved for PIK3CA-mutant advanced breast cancer after progression on endocrine therapy.",
    },
    "IDH1": {
        "gene": "IDH1",
        "full_name": "Isocitrate Dehydrogenase 1",
        "cancer_types": ["glioma", "AML", "cholangiocarcinoma"],
        "key_variants": ["R132H", "R132C", "R132G", "R132S"],
        "targeted_therapies": ["ivosidenib", "vorasidenib"],
        "combination_therapies": ["ivosidenib + azacitidine"],
        "resistance_mutations": ["second-site IDH1 mutations", "RTK pathway activation"],
        "pathway": "metabolic",
        "evidence_level": "A",
        "description": "IDH1 R132H is the most common mutation. Ivosidenib is approved for IDH1-mutant "
                       "AML and cholangiocarcinoma. Vorasidenib (brain-penetrant dual IDH1/2) approved for "
                       "low-grade glioma.",
    },
    "IDH2": {
        "gene": "IDH2",
        "full_name": "Isocitrate Dehydrogenase 2",
        "cancer_types": ["AML", "angioimmunoblastic T-cell lymphoma"],
        "key_variants": ["R140Q", "R172K"],
        "targeted_therapies": ["enasidenib"],
        "combination_therapies": ["enasidenib + azacitidine"],
        "resistance_mutations": ["second-site IDH2 mutations", "RAS pathway activation"],
        "pathway": "metabolic",
        "evidence_level": "A",
        "description": "IDH2 mutations occur in ~10-15% of AML. Enasidenib is approved for relapsed/refractory "
                       "IDH2-mutant AML. R140Q is more common than R172K.",
    },
    "BRCA1": {
        "gene": "BRCA1",
        "full_name": "BRCA1 DNA Repair Associated",
        "cancer_types": ["breast", "ovarian", "prostate", "pancreatic"],
        "key_variants": ["various truncating/missense mutations", "185delAG", "5382insC",
                          "large rearrangements", "promoter methylation (somatic)"],
        "targeted_therapies": ["olaparib", "rucaparib", "niraparib", "talazoparib"],
        "combination_therapies": ["olaparib + bevacizumab", "talazoparib + enzalutamide"],
        "resistance_mutations": ["reversion mutations", "53BP1 loss", "BRCA1 promoter demethylation",
                                  "RAD51 upregulation", "drug efflux (ABCB1)"],
        "pathway": "DDR",
        "evidence_level": "A",
        "description": "BRCA1 mutations confer homologous recombination deficiency (HRD), creating synthetic "
                       "lethality with PARP inhibitors. Germline testing recommended for breast, ovarian, "
                       "prostate, and pancreatic cancer.",
    },
    "BRCA2": {
        "gene": "BRCA2",
        "full_name": "BRCA2 DNA Repair Associated",
        "cancer_types": ["breast", "ovarian", "prostate", "pancreatic"],
        "key_variants": ["various truncating/missense mutations", "6174delT"],
        "targeted_therapies": ["olaparib", "rucaparib", "niraparib", "talazoparib"],
        "combination_therapies": ["olaparib + abiraterone"],
        "resistance_mutations": ["reversion mutations", "RAD51C/D mutations", "drug efflux"],
        "pathway": "DDR",
        "evidence_level": "A",
        "description": "BRCA2 mutations create HRD and PARP inhibitor sensitivity. More common than BRCA1 "
                       "in prostate and pancreatic cancers. Platinum sensitivity often correlates.",
    },
    "TP53": {
        "gene": "TP53",
        "full_name": "Tumor Protein P53",
        "cancer_types": ["pan-cancer"],
        "key_variants": ["R175H", "R248W", "R273H", "G245S", "R249S", "Y220C", "hotspot missense"],
        "targeted_therapies": [],
        "combination_therapies": [],
        "resistance_mutations": [],
        "pathway": "cell_cycle/apoptosis",
        "evidence_level": "D",
        "description": "TP53 is the most frequently mutated gene in cancer (~50% of all cancers). "
                       "Currently no direct targeted therapies are FDA-approved, though Y220C-targeting "
                       "compounds are in development. TP53 mutations are prognostic in most cancer types.",
    },
    "PTEN": {
        "gene": "PTEN",
        "full_name": "Phosphatase and Tensin Homolog",
        "cancer_types": ["prostate", "endometrial", "glioblastoma", "breast"],
        "key_variants": ["deletion", "truncating mutations", "promoter methylation"],
        "targeted_therapies": [],
        "combination_therapies": [],
        "resistance_mutations": [],
        "pathway": "PI3K/AKT/mTOR",
        "evidence_level": "C",
        "description": "PTEN loss activates PI3K/AKT signaling. No direct PTEN-restoring therapies exist, "
                       "but PTEN loss may predict response to PI3K/AKT/mTOR inhibitors. Common in prostate "
                       "and endometrial cancers.",
    },
    "CDKN2A": {
        "gene": "CDKN2A",
        "full_name": "Cyclin Dependent Kinase Inhibitor 2A (p16/p14ARF)",
        "cancer_types": ["melanoma", "NSCLC", "pancreatic", "mesothelioma", "glioma"],
        "key_variants": ["homozygous deletion", "promoter methylation", "missense mutations"],
        "targeted_therapies": ["palbociclib", "ribociclib", "abemaciclib"],
        "combination_therapies": ["CDK4/6i + endocrine therapy"],
        "resistance_mutations": ["RB1 loss", "CDK6 amplification", "CCNE1 amplification"],
        "pathway": "cell_cycle",
        "evidence_level": "B",
        "description": "CDKN2A loss removes the p16-mediated brake on CDK4/6, making tumors potentially "
                       "sensitive to CDK4/6 inhibitors. Homozygous deletion is a poor prognostic marker "
                       "in mesothelioma and glioma.",
    },
    "STK11": {
        "gene": "STK11",
        "full_name": "Serine/Threonine Kinase 11 (LKB1)",
        "cancer_types": ["NSCLC", "cervical"],
        "key_variants": ["truncating mutations", "deletions"],
        "targeted_therapies": [],
        "combination_therapies": [],
        "resistance_mutations": [],
        "pathway": "AMPK/mTOR",
        "evidence_level": "C",
        "description": "STK11/LKB1 loss in NSCLC (~15-20%) is associated with primary resistance to "
                       "PD-1/PD-L1 immunotherapy, especially in KRAS co-mutated tumors. No direct targeted "
                       "therapy, but important for treatment selection.",
    },
    "ESR1": {
        "gene": "ESR1",
        "full_name": "Estrogen Receptor 1",
        "cancer_types": ["breast"],
        "key_variants": ["Y537S", "D538G", "E380Q", "L536R", "ligand-binding domain mutations"],
        "targeted_therapies": ["elacestrant", "fulvestrant"],
        "combination_therapies": ["elacestrant monotherapy"],
        "resistance_mutations": ["compound ESR1 mutations", "RB1 loss", "MYC amplification"],
        "pathway": "estrogen_signaling",
        "evidence_level": "A",
        "description": "ESR1 mutations (~30-40% in metastatic HR+ breast cancer) confer resistance to "
                       "aromatase inhibitors. Elacestrant (oral SERD) is FDA-approved for ESR1-mutant "
                       "advanced breast cancer after prior endocrine therapy + CDK4/6i.",
    },
    "APC": {
        "gene": "APC",
        "full_name": "Adenomatous Polyposis Coli",
        "cancer_types": ["colorectal"],
        "key_variants": ["truncating mutations", "R1450*", "mutation cluster region"],
        "targeted_therapies": [],
        "combination_therapies": [],
        "resistance_mutations": [],
        "pathway": "WNT",
        "evidence_level": "D",
        "description": "APC mutations initiate >80% of colorectal cancers through WNT pathway activation. "
                       "No direct targeted therapies, but foundational for CRC pathogenesis understanding. "
                       "Germline APC mutations cause familial adenomatous polyposis (FAP).",
    },
    "PALB2": {
        "gene": "PALB2",
        "full_name": "Partner and Localizer of BRCA2",
        "cancer_types": ["breast", "pancreatic", "ovarian"],
        "key_variants": ["truncating mutations"],
        "targeted_therapies": ["olaparib", "rucaparib"],
        "combination_therapies": [],
        "resistance_mutations": ["reversion mutations"],
        "pathway": "DDR",
        "evidence_level": "B",
        "description": "PALB2 mutations confer HRD similar to BRCA1/2. Classified as a high-penetrance "
                       "breast cancer gene. PARP inhibitor sensitivity demonstrated in clinical trials.",
    },
    "ATM": {
        "gene": "ATM",
        "full_name": "Ataxia Telangiectasia Mutated",
        "cancer_types": ["prostate", "pancreatic", "breast", "CLL"],
        "key_variants": ["truncating mutations", "missense variants (VUS common)"],
        "targeted_therapies": ["olaparib"],
        "combination_therapies": [],
        "resistance_mutations": [],
        "pathway": "DDR",
        "evidence_level": "C",
        "description": "ATM loss impairs DNA damage response. Evidence for PARP inhibitor benefit is less "
                       "robust than BRCA1/2 — primarily studied in mCRPC (PROfound trial ATM subgroup).",
    },
    "POLE": {
        "gene": "POLE",
        "full_name": "DNA Polymerase Epsilon Catalytic Subunit",
        "cancer_types": ["endometrial", "colorectal"],
        "key_variants": ["P286R", "V411L", "S459F", "exonuclease domain mutations"],
        "targeted_therapies": [],
        "combination_therapies": [],
        "resistance_mutations": [],
        "pathway": "DDR",
        "evidence_level": "B",
        "description": "POLE exonuclease domain mutations create ultramutated tumors (>100 mut/Mb) with "
                       "excellent prognosis and high immunotherapy response rates. Important for treatment "
                       "de-escalation decisions in endometrial cancer.",
    },
    "ARID1A": {
        "gene": "ARID1A",
        "full_name": "AT-Rich Interaction Domain 1A",
        "cancer_types": ["ovarian (clear cell)", "gastric", "endometrial", "cholangiocarcinoma"],
        "key_variants": ["truncating mutations", "frameshifts"],
        "targeted_therapies": [],
        "combination_therapies": [],
        "resistance_mutations": [],
        "pathway": "chromatin_remodeling",
        "evidence_level": "C",
        "description": "ARID1A is a SWI/SNF chromatin remodeling subunit mutated in ~50% of ovarian clear "
                       "cell carcinoma. Loss may confer sensitivity to EZH2 inhibitors and ATR inhibitors. "
                       "Emerging therapeutic target.",
    },
    "MSI_H": {
        "gene": "MSI-H/dMMR",
        "full_name": "Microsatellite Instability High / Mismatch Repair Deficient",
        "cancer_types": ["tissue-agnostic", "colorectal", "endometrial", "gastric"],
        "key_variants": ["MLH1 promoter methylation", "MLH1/MSH2/MSH6/PMS2 loss"],
        "targeted_therapies": ["pembrolizumab", "nivolumab", "dostarlimab"],
        "combination_therapies": ["nivolumab + ipilimumab"],
        "resistance_mutations": ["B2M loss", "JAK1/2 mutations", "STK11 co-mutation"],
        "pathway": "DDR/immune",
        "evidence_level": "A",
        "description": "MSI-H/dMMR is a tissue-agnostic biomarker for immunotherapy (pembrolizumab, "
                       "dostarlimab). KEYNOTE-177 established first-line pembrolizumab for MSI-H CRC. "
                       "CheckMate-8HW showed nivo+ipi superiority.",
    },
}


# ═══════════════════════════════════════════════════════════════════════
# 2. THERAPY MAP (~30 drugs)
# ═══════════════════════════════════════════════════════════════════════

THERAPY_MAP: Dict[str, Dict] = {
    "osimertinib": {
        "drug_name": "osimertinib",
        "brand_name": "Tagrisso",
        "category": "targeted",
        "targets": ["EGFR"],
        "approved_indications": ["EGFR-mutant NSCLC (1L)", "EGFR T790M+ NSCLC (2L+)", "adjuvant EGFR+ NSCLC"],
        "mechanism": "3rd-generation EGFR TKI, irreversible binding to C797, mutant-selective",
        "key_trials": ["FLAURA", "FLAURA2", "AURA3", "ADAURA"],
        "resistance_mechanisms": ["C797S", "MET amplification", "small cell transformation", "HER2 amp"],
        "evidence_level": "A",
    },
    "pembrolizumab": {
        "drug_name": "pembrolizumab",
        "brand_name": "Keytruda",
        "category": "immunotherapy",
        "targets": ["PD-1"],
        "approved_indications": ["NSCLC (PD-L1≥50% 1L mono, PD-L1≥1% + chemo)", "melanoma",
                                  "MSI-H/dMMR (tissue-agnostic)", "TMB-H (≥10 mut/Mb)",
                                  "head and neck", "cervical", "gastric", "TNBC", "RCC"],
        "mechanism": "Anti-PD-1 monoclonal antibody, blocks PD-1/PD-L1 interaction to restore T-cell activity",
        "key_trials": ["KEYNOTE-024", "KEYNOTE-189", "KEYNOTE-158", "KEYNOTE-177", "KEYNOTE-826"],
        "resistance_mechanisms": ["B2M loss", "JAK1/2 loss", "STK11 co-mutation", "WNT/β-catenin activation"],
        "evidence_level": "A",
    },
    "nivolumab": {
        "drug_name": "nivolumab",
        "brand_name": "Opdivo",
        "category": "immunotherapy",
        "targets": ["PD-1"],
        "approved_indications": ["melanoma", "NSCLC", "RCC", "Hodgkin lymphoma", "head and neck",
                                  "urothelial", "MSI-H CRC", "HCC", "gastric"],
        "mechanism": "Anti-PD-1 monoclonal antibody",
        "key_trials": ["CheckMate-227", "CheckMate-9LA", "CheckMate-274", "CheckMate-8HW"],
        "resistance_mechanisms": ["B2M loss", "JAK1/2 mutations", "PTEN loss"],
        "evidence_level": "A",
    },
    "vemurafenib": {
        "drug_name": "vemurafenib",
        "brand_name": "Zelboraf",
        "category": "targeted",
        "targets": ["BRAF V600"],
        "approved_indications": ["BRAF V600E melanoma", "BRAF V600E Erdheim-Chester disease"],
        "mechanism": "Selective BRAF V600 kinase inhibitor",
        "key_trials": ["BRIM-3"],
        "resistance_mechanisms": ["MEK1/2 mutations", "NRAS activation", "BRAF amplification", "COT overexpression"],
        "evidence_level": "A",
    },
    "dabrafenib": {
        "drug_name": "dabrafenib",
        "brand_name": "Tafinlar",
        "category": "targeted",
        "targets": ["BRAF V600"],
        "approved_indications": ["BRAF V600E/K melanoma", "BRAF V600E NSCLC", "BRAF V600E ATC"],
        "mechanism": "Selective BRAF kinase inhibitor, typically combined with trametinib (MEK inhibitor)",
        "key_trials": ["COMBI-d", "COMBI-v", "BRF113928"],
        "resistance_mechanisms": ["MEK mutations", "RAS activation", "BRAF amplification"],
        "evidence_level": "A",
    },
    "sotorasib": {
        "drug_name": "sotorasib",
        "brand_name": "Lumakras",
        "category": "targeted",
        "targets": ["KRAS G12C"],
        "approved_indications": ["KRAS G12C NSCLC (2L+)"],
        "mechanism": "Irreversible, selective KRAS G12C inhibitor — locks KRAS in inactive GDP-bound state",
        "key_trials": ["CodeBreaK 100", "CodeBreaK 200"],
        "resistance_mechanisms": ["Y96D", "R68S", "H95 mutations", "MET amplification",
                                   "secondary KRAS mutations", "RAS-MAPK reactivation"],
        "evidence_level": "A",
    },
    "adagrasib": {
        "drug_name": "adagrasib",
        "brand_name": "Krazati",
        "category": "targeted",
        "targets": ["KRAS G12C"],
        "approved_indications": ["KRAS G12C NSCLC (2L+)"],
        "mechanism": "Irreversible KRAS G12C inhibitor with long half-life and CNS penetration",
        "key_trials": ["KRYSTAL-1", "KRYSTAL-7"],
        "resistance_mechanisms": ["secondary KRAS mutations", "MET amplification", "MAPK reactivation"],
        "evidence_level": "A",
    },
    "larotrectinib": {
        "drug_name": "larotrectinib",
        "brand_name": "Vitrakvi",
        "category": "targeted",
        "targets": ["NTRK1/2/3"],
        "approved_indications": ["NTRK fusion-positive solid tumors (tissue-agnostic)"],
        "mechanism": "Selective pan-TRK inhibitor",
        "key_trials": ["NAVIGATE", "SCOUT", "LOXO-TRK-14001"],
        "resistance_mechanisms": ["solvent front mutations (G595R, G623R)", "xDFG mutations"],
        "evidence_level": "A",
    },
    "entrectinib": {
        "drug_name": "entrectinib",
        "brand_name": "Rozlytrek",
        "category": "targeted",
        "targets": ["NTRK1/2/3", "ROS1", "ALK"],
        "approved_indications": ["NTRK fusion-positive solid tumors", "ROS1+ NSCLC"],
        "mechanism": "Pan-TRK, ROS1, and ALK inhibitor with CNS penetration",
        "key_trials": ["STARTRK-2", "STARTRK-1"],
        "resistance_mechanisms": ["solvent front mutations", "xDFG mutations", "G2032R (ROS1)"],
        "evidence_level": "A",
    },
    "selpercatinib": {
        "drug_name": "selpercatinib",
        "brand_name": "Retevmo",
        "category": "targeted",
        "targets": ["RET"],
        "approved_indications": ["RET fusion-positive NSCLC", "RET-mutant medullary thyroid",
                                  "RET fusion-positive thyroid (age≥12)"],
        "mechanism": "Highly selective RET kinase inhibitor designed to minimize off-target VEGFR activity",
        "key_trials": ["LIBRETTO-001", "LIBRETTO-431"],
        "resistance_mechanisms": ["G810R/S/C", "V804M", "Y806C/H"],
        "evidence_level": "A",
    },
    "capmatinib": {
        "drug_name": "capmatinib",
        "brand_name": "Tabrecta",
        "category": "targeted",
        "targets": ["MET"],
        "approved_indications": ["MET exon 14 skipping NSCLC"],
        "mechanism": "Selective MET kinase inhibitor",
        "key_trials": ["GEOMETRY mono-1"],
        "resistance_mechanisms": ["D1228N/V", "Y1230H/C", "KRAS amplification"],
        "evidence_level": "A",
    },
    "erdafitinib": {
        "drug_name": "erdafitinib",
        "brand_name": "Balversa",
        "category": "targeted",
        "targets": ["FGFR1/2/3/4"],
        "approved_indications": ["FGFR-altered urothelial carcinoma"],
        "mechanism": "Pan-FGFR tyrosine kinase inhibitor",
        "key_trials": ["THOR"],
        "resistance_mechanisms": ["gatekeeper mutations (V565I)", "FGFR2 secondary mutations"],
        "evidence_level": "A",
    },
    "alpelisib": {
        "drug_name": "alpelisib",
        "brand_name": "Piqray",
        "category": "targeted",
        "targets": ["PI3K alpha (PIK3CA)"],
        "approved_indications": ["PIK3CA-mutant HR+/HER2- advanced breast cancer (with fulvestrant)"],
        "mechanism": "PI3Kα-selective inhibitor",
        "key_trials": ["SOLAR-1"],
        "resistance_mechanisms": ["PTEN loss", "AKT1 E17K", "mTOR mutations"],
        "evidence_level": "A",
    },
    "olaparib": {
        "drug_name": "olaparib",
        "brand_name": "Lynparza",
        "category": "targeted",
        "targets": ["PARP1/2"],
        "approved_indications": ["BRCA-mutant breast cancer", "BRCA-mutant ovarian cancer",
                                  "BRCA-mutant prostate cancer", "BRCA-mutant pancreatic cancer",
                                  "HRD+ ovarian cancer (with bevacizumab)"],
        "mechanism": "PARP1/2 inhibitor — traps PARP on DNA, synthetic lethality with HRD",
        "key_trials": ["OlympiAD", "SOLO-1", "SOLO-2", "PROfound", "POLO", "PAOLA-1"],
        "resistance_mechanisms": ["BRCA reversion mutations", "53BP1 loss", "RAD51 upregulation",
                                   "drug efflux (ABCB1)"],
        "evidence_level": "A",
    },
    "ivosidenib": {
        "drug_name": "ivosidenib",
        "brand_name": "Tibsovo",
        "category": "targeted",
        "targets": ["IDH1"],
        "approved_indications": ["IDH1-mutant AML (newly diagnosed, R/R)",
                                  "IDH1-mutant cholangiocarcinoma"],
        "mechanism": "IDH1 inhibitor — blocks mutant IDH1 enzyme, reduces 2-HG oncometabolite",
        "key_trials": ["ClarIDHy", "AGILE"],
        "resistance_mechanisms": ["second-site IDH1 mutations", "RTK pathway activation"],
        "evidence_level": "A",
    },
    "trastuzumab_deruxtecan": {
        "drug_name": "trastuzumab deruxtecan",
        "brand_name": "Enhertu",
        "category": "targeted",
        "targets": ["HER2"],
        "approved_indications": ["HER2+ metastatic breast cancer (2L+)", "HER2-low breast cancer",
                                  "HER2+ gastric cancer", "HER2-mutant NSCLC"],
        "mechanism": "HER2-directed antibody-drug conjugate (ADC) with topoisomerase I inhibitor payload; "
                     "bystander effect enables activity in HER2-low tumors",
        "key_trials": ["DESTINY-Breast03", "DESTINY-Breast04", "DESTINY-Lung01", "DESTINY-Gastric01"],
        "resistance_mechanisms": ["HER2 downregulation", "drug efflux", "SLX4-dependent DNA repair"],
        "evidence_level": "A",
    },
    "palbociclib": {
        "drug_name": "palbociclib",
        "brand_name": "Ibrance",
        "category": "targeted",
        "targets": ["CDK4/6"],
        "approved_indications": ["HR+/HER2- advanced breast cancer (with endocrine therapy)"],
        "mechanism": "Selective CDK4/6 inhibitor — blocks RB phosphorylation, induces G1 arrest",
        "key_trials": ["PALOMA-2", "PALOMA-3"],
        "resistance_mechanisms": ["RB1 loss", "CDK6 amplification", "CCNE1 amplification",
                                   "FGFR pathway activation"],
        "evidence_level": "A",
    },
    "lorlatinib": {
        "drug_name": "lorlatinib",
        "brand_name": "Lorbrena",
        "category": "targeted",
        "targets": ["ALK", "ROS1"],
        "approved_indications": ["ALK+ NSCLC (1L and 2L+)"],
        "mechanism": "3rd-generation ALK TKI — macrocyclic structure with CNS penetration, "
                     "active against most ALK resistance mutations including G1202R",
        "key_trials": ["CROWN"],
        "resistance_mechanisms": ["compound ALK mutations", "ALK-independent bypass pathways"],
        "evidence_level": "A",
    },
    "alectinib": {
        "drug_name": "alectinib",
        "brand_name": "Alecensa",
        "category": "targeted",
        "targets": ["ALK"],
        "approved_indications": ["ALK+ NSCLC (1L and 2L+)"],
        "mechanism": "2nd-generation ALK inhibitor with CNS penetration",
        "key_trials": ["ALEX", "J-ALEX", "ALUR"],
        "resistance_mechanisms": ["G1202R", "I1171T/N/S", "V1180L", "compound mutations"],
        "evidence_level": "A",
    },
    "dostarlimab": {
        "drug_name": "dostarlimab",
        "brand_name": "Jemperli",
        "category": "immunotherapy",
        "targets": ["PD-1"],
        "approved_indications": ["dMMR recurrent/advanced endometrial cancer",
                                  "dMMR recurrent/advanced solid tumors"],
        "mechanism": "Anti-PD-1 monoclonal antibody",
        "key_trials": ["GARNET"],
        "resistance_mechanisms": ["B2M loss", "JAK1/2 mutations"],
        "evidence_level": "A",
    },
}


# ═══════════════════════════════════════════════════════════════════════
# 3. RESISTANCE MAP (~20 entries)
# ═══════════════════════════════════════════════════════════════════════

RESISTANCE_MAP: Dict[str, List[Dict]] = {
    "EGFR TKIs (1st/2nd gen)": [
        {"mutation": "T790M", "gene": "EGFR", "frequency": "~50-60%",
         "bypass": None, "next_line": ["osimertinib"]},
        {"mutation": "MET amplification", "gene": "MET", "frequency": "~5-10%",
         "bypass": "PI3K/AKT", "next_line": ["osimertinib + MET inhibitor"]},
        {"mutation": "HER2 amplification", "gene": "ERBB2", "frequency": "~5%",
         "bypass": "HER2 signaling", "next_line": ["T-DXd"]},
        {"mutation": "Small cell transformation", "gene": "RB1/TP53 loss", "frequency": "~3-10%",
         "bypass": "lineage switch", "next_line": ["platinum + etoposide"]},
    ],
    "osimertinib": [
        {"mutation": "C797S", "gene": "EGFR", "frequency": "~10-15%",
         "bypass": None, "next_line": ["combination EGFR strategies", "amivantamab + lazertinib"]},
        {"mutation": "MET amplification", "gene": "MET", "frequency": "~15-20%",
         "bypass": "PI3K/AKT", "next_line": ["osimertinib + savolitinib"]},
        {"mutation": "BRAF V600E", "gene": "BRAF", "frequency": "~3%",
         "bypass": "MAPK", "next_line": ["BRAF+MEK inhibitors"]},
    ],
    "BRAF V600 inhibitors": [
        {"mutation": "MEK1/2 mutations", "gene": "MAP2K1/2", "frequency": "~5-10%",
         "bypass": "MAPK reactivation", "next_line": ["MEK inhibitor switch"]},
        {"mutation": "NRAS mutations", "gene": "NRAS", "frequency": "~10-20%",
         "bypass": "MAPK reactivation", "next_line": ["immunotherapy"]},
        {"mutation": "BRAF amplification", "gene": "BRAF", "frequency": "~10%",
         "bypass": "MAPK overdrive", "next_line": ["dose escalation", "immunotherapy"]},
    ],
    "ALK TKIs": [
        {"mutation": "G1202R", "gene": "ALK", "frequency": "~20-30% (post-crizotinib)",
         "bypass": None, "next_line": ["lorlatinib"]},
        {"mutation": "compound mutations", "gene": "ALK", "frequency": "~10-15% (post-lorlatinib)",
         "bypass": None, "next_line": ["chemotherapy", "clinical trials"]},
        {"mutation": "MET amplification", "gene": "MET", "frequency": "~5%",
         "bypass": "PI3K", "next_line": ["ALK TKI + MET inhibitor"]},
    ],
    "KRAS G12C inhibitors": [
        {"mutation": "Y96D", "gene": "KRAS", "frequency": "~5-10%",
         "bypass": None, "next_line": ["next-gen KRAS G12C inhibitors"]},
        {"mutation": "Secondary KRAS mutations", "gene": "KRAS", "frequency": "~10%",
         "bypass": "RAS-MAPK", "next_line": ["SOS1 inhibitors", "SHP2 inhibitors"]},
        {"mutation": "MET amplification", "gene": "MET", "frequency": "~10%",
         "bypass": "RTK bypass", "next_line": ["KRAS G12Ci + MET inhibitor"]},
    ],
    "PARP inhibitors": [
        {"mutation": "BRCA reversion mutations", "gene": "BRCA1/2", "frequency": "~20-30%",
         "bypass": "HR restoration", "next_line": ["platinum rechallenge", "immunotherapy"]},
        {"mutation": "53BP1/RIF1 loss", "gene": "TP53BP1", "frequency": "~5%",
         "bypass": "NHEJ restoration", "next_line": ["immunotherapy"]},
        {"mutation": "Drug efflux (ABCB1)", "gene": "ABCB1", "frequency": "~5%",
         "bypass": "drug clearance", "next_line": ["switch PARP inhibitor"]},
    ],
    "anti-PD-1/PD-L1": [
        {"mutation": "B2M loss", "gene": "B2M", "frequency": "~5-10%",
         "bypass": "MHC-I loss", "next_line": ["CTLA-4 combination", "cellular therapy"]},
        {"mutation": "JAK1/2 mutations", "gene": "JAK1/JAK2", "frequency": "~5%",
         "bypass": "IFN-gamma signaling loss", "next_line": ["cellular therapy"]},
        {"mutation": "STK11 co-mutation", "gene": "STK11", "frequency": "~15-20% (NSCLC)",
         "bypass": "cold microenvironment", "next_line": ["combination strategies"]},
    ],
    "CDK4/6 inhibitors": [
        {"mutation": "RB1 loss", "gene": "RB1", "frequency": "~5-10%",
         "bypass": "target loss", "next_line": ["chemotherapy", "PI3K inhibitors"]},
        {"mutation": "CCNE1 amplification", "gene": "CCNE1", "frequency": "~10%",
         "bypass": "CDK2-driven G1/S", "next_line": ["chemotherapy"]},
        {"mutation": "CDK6 amplification", "gene": "CDK6", "frequency": "~5%",
         "bypass": "target amplification", "next_line": ["dose escalation"]},
    ],
}


# ═══════════════════════════════════════════════════════════════════════
# 4. PATHWAY MAP (~10 pathways)
# ═══════════════════════════════════════════════════════════════════════

PATHWAY_MAP: Dict[str, Dict] = {
    "MAPK": {
        "pathway_name": "MAPK (RAS-RAF-MEK-ERK)",
        "key_genes": ["KRAS", "NRAS", "HRAS", "BRAF", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3", "NF1"],
        "therapeutic_targets": ["BRAF V600 (vemurafenib/dabrafenib)", "MEK1/2 (trametinib/cobimetinib)",
                                 "KRAS G12C (sotorasib/adagrasib)", "ERK1/2 (ulixertinib)"],
        "cross_talk": ["PI3K/AKT (feedback activation upon MAPK inhibition)", "WNT", "cell cycle (cyclin D1)"],
        "clinical_relevance": "Most frequently altered oncogenic pathway. MAPK reactivation is the dominant "
                              "resistance mechanism to targeted therapies in this pathway.",
    },
    "PI3K_AKT_mTOR": {
        "pathway_name": "PI3K/AKT/mTOR",
        "key_genes": ["PIK3CA", "PIK3R1", "PTEN", "AKT1", "AKT2", "MTOR", "TSC1", "TSC2", "STK11"],
        "therapeutic_targets": ["PI3Kα (alpelisib)", "AKT (capivasertib)", "mTOR (everolimus/temsirolimus)",
                                 "dual PI3K/mTOR (future)"],
        "cross_talk": ["MAPK (reciprocal feedback)", "estrogen receptor signaling", "cell cycle"],
        "clinical_relevance": "Activated in >50% of cancers through PIK3CA mutation, PTEN loss, or AKT activation. "
                              "Critical for treatment resistance across tumor types.",
    },
    "DDR": {
        "pathway_name": "DNA Damage Response",
        "key_genes": ["BRCA1", "BRCA2", "ATM", "ATR", "CHEK1", "CHEK2", "PALB2", "RAD51",
                       "POLE", "MSH2", "MLH1", "MSH6", "PMS2"],
        "therapeutic_targets": ["PARP1/2 (olaparib/rucaparib/niraparib)", "ATR (berzosertib)",
                                 "WEE1 (adavosertib)", "CHK1 (prexasertib)"],
        "cross_talk": ["cell cycle checkpoints", "immune response (cGAS-STING)", "apoptosis"],
        "clinical_relevance": "HRD creates synthetic lethality with PARP inhibitors. MSI-H/dMMR predicts "
                              "immunotherapy response. Foundation of precision oncology biomarker-guided therapy.",
    },
    "cell_cycle": {
        "pathway_name": "Cell Cycle (CDK-RB-E2F)",
        "key_genes": ["CDK4", "CDK6", "CDKN2A", "CCND1", "CCNE1", "RB1", "CDK2", "TP53", "MDM2"],
        "therapeutic_targets": ["CDK4/6 (palbociclib/ribociclib/abemaciclib)", "CDK2 (emerging)",
                                 "MDM2 (nutlin derivatives)"],
        "cross_talk": ["estrogen receptor (ER drives CCND1)", "MAPK (drives CCND1)",
                       "PI3K/AKT (drives cell cycle entry)"],
        "clinical_relevance": "CDK4/6 inhibitors transformed HR+ breast cancer treatment. RB1 loss is a "
                              "key resistance mechanism. CCNE1 amplification predicts CDK4/6i resistance.",
    },
    "WNT": {
        "pathway_name": "WNT/β-catenin",
        "key_genes": ["APC", "CTNNB1", "AXIN1", "AXIN2", "RNF43", "RSPO2", "RSPO3"],
        "therapeutic_targets": ["porcupine inhibitors (emerging)", "tankyrase inhibitors (emerging)"],
        "cross_talk": ["MAPK", "PI3K", "Notch"],
        "clinical_relevance": "WNT activation (APC loss) initiates >80% of CRC. β-catenin activation "
                              "may confer immunotherapy resistance. Limited druggable targets currently.",
    },
    "JAK_STAT": {
        "pathway_name": "JAK-STAT Signaling",
        "key_genes": ["JAK1", "JAK2", "JAK3", "STAT3", "STAT5", "SOCS1"],
        "therapeutic_targets": ["JAK1/2 (ruxolitinib)", "JAK2 (fedratinib)", "STAT3 (emerging)"],
        "cross_talk": ["immune response", "MAPK", "PI3K"],
        "clinical_relevance": "JAK2 V617F defines myeloproliferative neoplasms. JAK1/2 loss-of-function "
                              "mutations confer immunotherapy resistance through impaired IFN-gamma signaling.",
    },
    "apoptosis": {
        "pathway_name": "Apoptosis (BCL-2 Family)",
        "key_genes": ["BCL2", "BCL2L1", "MCL1", "BAX", "BAK1", "BIM", "TP53"],
        "therapeutic_targets": ["BCL-2 (venetoclax)", "MCL-1 (emerging)", "BCL-XL (emerging)"],
        "cross_talk": ["TP53/MDM2", "cell cycle", "DDR"],
        "clinical_relevance": "Venetoclax + azacitidine is standard for unfit AML. BCL-2 dependence in CLL "
                              "makes venetoclax-based regimens highly effective.",
    },
    "angiogenesis": {
        "pathway_name": "Angiogenesis (VEGF/VEGFR)",
        "key_genes": ["VEGFA", "KDR", "FLT1", "FLT4", "PDGFRA", "PDGFRB"],
        "therapeutic_targets": ["VEGF-A (bevacizumab)", "VEGFR (sunitinib/pazopanib/cabozantinib/lenvatinib)",
                                 "multi-kinase (sorafenib)"],
        "cross_talk": ["HIF pathway", "MAPK", "PI3K"],
        "clinical_relevance": "Anti-angiogenics are backbone therapy in RCC, HCC, CRC, ovarian, and cervical cancer. "
                              "Combination with immunotherapy (lenvatinib+pembrolizumab) is paradigm-shifting.",
    },
    "Notch": {
        "pathway_name": "Notch Signaling",
        "key_genes": ["NOTCH1", "NOTCH2", "NOTCH3", "NOTCH4", "DLL3", "JAG1"],
        "therapeutic_targets": ["DLL3 (tarlatamab — BiTE)", "gamma-secretase inhibitors (emerging)"],
        "cross_talk": ["WNT", "Hedgehog", "MYC"],
        "clinical_relevance": "NOTCH1 mutations occur in T-ALL, CLL. DLL3 is targetable in SCLC with "
                              "tarlatamab (bispecific T-cell engager). Notch is context-dependent — oncogenic "
                              "in some cancers, tumor-suppressive in others.",
    },
    "Hedgehog": {
        "pathway_name": "Hedgehog (HH) Signaling",
        "key_genes": ["PTCH1", "SMO", "GLI1", "GLI2", "SUFU"],
        "therapeutic_targets": ["SMO (vismodegib/sonidegib)"],
        "cross_talk": ["WNT", "Notch", "PI3K"],
        "clinical_relevance": "Vismodegib and sonidegib are approved for advanced basal cell carcinoma. "
                              "PTCH1 loss-of-function mutations activate pathway constitutively.",
    },
}


# ═══════════════════════════════════════════════════════════════════════
# 5. BIOMARKER PANELS (~15 entries)
# ═══════════════════════════════════════════════════════════════════════

BIOMARKER_PANELS: Dict[str, Dict] = {
    "TMB-H": {
        "name": "Tumor Mutational Burden — High",
        "type": "predictive",
        "testing_method": "NGS panel (F1CDx, MSK-IMPACT) or WES",
        "clinical_cutoff": "≥10 mutations/Mb",
        "predictive_for": ["pembrolizumab (tissue-agnostic)"],
        "cancer_types": ["tissue-agnostic"],
        "evidence_level": "A",
        "description": "TMB-H (≥10 mut/Mb) is an FDA-approved tissue-agnostic biomarker for pembrolizumab. "
                       "Higher TMB generally correlates with neoantigen load and immunotherapy benefit.",
    },
    "MSI-H": {
        "name": "Microsatellite Instability — High / dMMR",
        "type": "predictive",
        "testing_method": "IHC (MMR proteins), PCR (microsatellite markers), NGS",
        "clinical_cutoff": "Loss of MLH1/MSH2/MSH6/PMS2 by IHC or MSI by PCR/NGS",
        "predictive_for": ["pembrolizumab", "nivolumab", "dostarlimab", "nivolumab + ipilimumab"],
        "cancer_types": ["tissue-agnostic", "colorectal", "endometrial", "gastric"],
        "evidence_level": "A",
        "description": "MSI-H/dMMR is the strongest predictive biomarker for immunotherapy response. "
                       "First tissue-agnostic FDA approval. Present in ~15% of CRC, ~25% of endometrial.",
    },
    "PD-L1": {
        "name": "PD-L1 Expression (TPS/CPS)",
        "type": "predictive",
        "testing_method": "IHC (22C3, 28-8, SP142, SP263 antibodies)",
        "clinical_cutoff": "TPS≥50% (1L NSCLC mono), TPS≥1% (1L NSCLC + chemo), CPS≥1-10 (various)",
        "predictive_for": ["pembrolizumab", "atezolizumab", "durvalumab"],
        "cancer_types": ["NSCLC", "head and neck", "gastric", "cervical", "TNBC", "urothelial"],
        "evidence_level": "A",
        "description": "PD-L1 TPS≥50% identifies NSCLC patients who benefit from pembrolizumab monotherapy. "
                       "Scoring systems (TPS vs CPS) and cutoffs vary by indication and assay.",
    },
    "HRD": {
        "name": "Homologous Recombination Deficiency",
        "type": "predictive",
        "testing_method": "Genomic scar assays (Myriad myChoice HRD, F1CDx LOH)",
        "clinical_cutoff": "myChoice HRD score ≥42, F1CDx LOH ≥16%",
        "predictive_for": ["PARP inhibitors (olaparib, niraparib)"],
        "cancer_types": ["ovarian", "breast", "prostate", "pancreatic"],
        "evidence_level": "A",
        "description": "HRD status predicts PARP inhibitor benefit beyond BRCA mutations. Composite scores "
                       "incorporate LOH, TAI, and LST. Most validated in ovarian cancer (PAOLA-1, PRIMA).",
    },
    "BRCA_status": {
        "name": "BRCA1/2 Mutation Status",
        "type": "predictive",
        "testing_method": "NGS panel, germline and somatic testing",
        "clinical_cutoff": "Pathogenic/likely pathogenic variants",
        "predictive_for": ["PARP inhibitors", "platinum chemotherapy"],
        "cancer_types": ["breast", "ovarian", "prostate", "pancreatic"],
        "evidence_level": "A",
        "description": "BRCA1/2 mutations are the strongest predictors of PARP inhibitor response. "
                       "Both germline and somatic mutations confer benefit. Platinum sensitivity often correlates.",
    },
    "EGFR_mutation": {
        "name": "EGFR Activating Mutations",
        "type": "predictive",
        "testing_method": "NGS, PCR (cobas), liquid biopsy (ctDNA)",
        "clinical_cutoff": "Sensitizing mutations (L858R, exon 19 del, others)",
        "predictive_for": ["osimertinib", "erlotinib", "gefitinib", "afatinib"],
        "cancer_types": ["NSCLC"],
        "evidence_level": "A",
        "description": "EGFR mutations are the most common actionable target in NSCLC (~15-20% overall, "
                       "~50% in Asian never-smokers). Testing at diagnosis is mandatory per NCCN guidelines.",
    },
    "ALK_fusion": {
        "name": "ALK Rearrangement",
        "type": "predictive",
        "testing_method": "FISH, IHC (D5F3), NGS",
        "clinical_cutoff": "Presence of ALK fusion",
        "predictive_for": ["alectinib", "lorlatinib", "brigatinib", "crizotinib"],
        "cancer_types": ["NSCLC"],
        "evidence_level": "A",
        "description": "ALK fusions occur in ~5% of NSCLC, typically in younger never/light-smokers. "
                       "5-year OS >60% with sequential ALK TKI therapy.",
    },
    "ctDNA_MRD": {
        "name": "Circulating Tumor DNA / Minimal Residual Disease",
        "type": "monitoring",
        "testing_method": "Liquid biopsy (Guardant, Signatera, FoundationOne Liquid CDx)",
        "clinical_cutoff": "ctDNA detected vs not detected (binary); variant allele frequency",
        "predictive_for": ["recurrence risk", "treatment response monitoring"],
        "cancer_types": ["CRC", "breast", "NSCLC", "bladder"],
        "evidence_level": "B",
        "description": "ctDNA MRD detection post-surgery identifies patients at high recurrence risk. "
                       "ctDNA clearance during treatment correlates with response. Actively studied for "
                       "therapy escalation/de-escalation decisions (CIRCULATE, DYNAMIC trials).",
    },
    "tumor_fraction": {
        "name": "Tumor Fraction (ctDNA)",
        "type": "monitoring",
        "testing_method": "Liquid biopsy, whole-genome sequencing of cfDNA",
        "clinical_cutoff": "Variable; typically >1% for reliable variant calling",
        "predictive_for": ["treatment response", "tumor burden estimation"],
        "cancer_types": ["pan-cancer"],
        "evidence_level": "C",
        "description": "Tumor fraction measures the proportion of cfDNA derived from tumor cells. "
                       "Low tumor fraction can lead to false negatives in liquid biopsy testing.",
    },
    "BRAF_V600": {
        "name": "BRAF V600 Mutation",
        "type": "predictive",
        "testing_method": "NGS, PCR (cobas), IHC (VE1)",
        "clinical_cutoff": "BRAF V600E/K/D/R detected",
        "predictive_for": ["dabrafenib+trametinib", "vemurafenib+cobimetinib",
                            "encorafenib+binimetinib", "encorafenib+cetuximab (CRC)"],
        "cancer_types": ["melanoma", "NSCLC", "CRC", "thyroid", "hairy cell leukemia"],
        "evidence_level": "A",
        "description": "BRAF V600E is highly actionable across tumor types. Treatment approach differs: "
                       "melanoma uses BRAF+MEK, CRC requires BRAF+MEK+anti-EGFR triplet.",
    },
    "KRAS_G12C": {
        "name": "KRAS G12C Mutation",
        "type": "predictive",
        "testing_method": "NGS, PCR",
        "clinical_cutoff": "KRAS G12C detected",
        "predictive_for": ["sotorasib", "adagrasib"],
        "cancer_types": ["NSCLC", "CRC"],
        "evidence_level": "A",
        "description": "KRAS G12C occurs in ~13% of NSCLC and ~3% of CRC. First directly targeted "
                       "KRAS mutations in clinical practice. sotorasib and adagrasib FDA-approved for NSCLC.",
    },
    "RET_fusion": {
        "name": "RET Fusion/Mutation",
        "type": "predictive",
        "testing_method": "NGS (DNA and RNA-based), FISH",
        "clinical_cutoff": "RET fusion or activating mutation detected",
        "predictive_for": ["selpercatinib", "pralsetinib"],
        "cancer_types": ["NSCLC", "thyroid"],
        "evidence_level": "A",
        "description": "RET fusions occur in ~1-2% of NSCLC. RNA-based NGS recommended as some fusions "
                       "may be missed by DNA-only testing.",
    },
    "NTRK_fusion": {
        "name": "NTRK Fusion",
        "type": "predictive",
        "testing_method": "NGS (RNA-based preferred), IHC (screening), FISH",
        "clinical_cutoff": "NTRK1/2/3 fusion detected",
        "predictive_for": ["larotrectinib", "entrectinib"],
        "cancer_types": ["tissue-agnostic"],
        "evidence_level": "A",
        "description": "NTRK fusions are rare (<1% pan-cancer) but highly actionable with ORR >75%. "
                       "Tissue-agnostic FDA approval. RNA-based testing increases sensitivity.",
    },
    "HER2_amp": {
        "name": "HER2 Amplification/Overexpression",
        "type": "predictive",
        "testing_method": "IHC (0/1+/2+/3+), FISH, NGS",
        "clinical_cutoff": "IHC 3+ or IHC 2+/FISH-amplified; HER2-low = IHC 1+ or IHC 2+/FISH-negative",
        "predictive_for": ["trastuzumab", "pertuzumab", "T-DXd", "tucatinib"],
        "cancer_types": ["breast", "gastric", "CRC", "NSCLC", "endometrial"],
        "evidence_level": "A",
        "description": "HER2 testing defines a major breast cancer subtype (~20%). The HER2-low paradigm "
                       "(IHC 1+/2+, FISH-negative) emerged with T-DXd (DESTINY-Breast04).",
    },
}


# ═══════════════════════════════════════════════════════════════════════
# 6. ENTITY ALIASES
# ═══════════════════════════════════════════════════════════════════════

ENTITY_ALIASES: Dict[str, str] = {
    # Drug brand → generic
    "keytruda": "pembrolizumab", "opdivo": "nivolumab", "tagrisso": "osimertinib",
    "zelboraf": "vemurafenib", "tafinlar": "dabrafenib", "mekinist": "trametinib",
    "lumakras": "sotorasib", "krazati": "adagrasib", "vitrakvi": "larotrectinib",
    "rozlytrek": "entrectinib", "retevmo": "selpercatinib", "gavreto": "pralsetinib",
    "tabrecta": "capmatinib", "tepmetko": "tepotinib", "balversa": "erdafitinib",
    "piqray": "alpelisib", "lynparza": "olaparib", "rubraca": "rucaparib",
    "zejula": "niraparib", "talzenna": "talazoparib", "tibsovo": "ivosidenib",
    "idhifa": "enasidenib", "enhertu": "trastuzumab_deruxtecan", "t-dxd": "trastuzumab_deruxtecan",
    "ibrance": "palbociclib", "kisqali": "ribociclib", "verzenio": "abemaciclib",
    "lorbrena": "lorlatinib", "alecensa": "alectinib", "xalkori": "crizotinib",
    "jemperli": "dostarlimab",
    # Gene aliases
    "her2": "HER2", "erbb2": "HER2", "egfrviii": "EGFR", "lkb1": "STK11",
    "p16": "CDKN2A", "p14arf": "CDKN2A", "ntrk": "NTRK",
    # Cancer type aliases
    "non-small cell": "NSCLC", "non small cell": "NSCLC", "lung adenocarcinoma": "NSCLC",
    "lung squamous": "NSCLC", "small cell lung": "SCLC", "triple negative": "TNBC",
    "crc": "colorectal", "colon cancer": "colorectal", "rectal cancer": "colorectal",
    "rcc": "renal cell", "hcc": "hepatocellular",
    # Biomarker aliases
    "msi high": "MSI-H", "msi-high": "MSI-H", "microsatellite instability": "MSI-H",
    "mmr deficient": "MSI-H", "dmmr": "MSI-H", "tmb high": "TMB-H", "tmb-high": "TMB-H",
    "pd-l1": "PD-L1", "pdl1": "PD-L1", "hrd": "HRD",
}


# ═══════════════════════════════════════════════════════════════════════
# 7. HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════


def get_target_context(gene: str) -> str:
    """Return formatted knowledge context for an actionable target gene."""
    target = ACTIONABLE_TARGETS.get(gene.upper()) or ACTIONABLE_TARGETS.get(gene)
    if not target:
        alias = ENTITY_ALIASES.get(gene.lower(), "").upper()
        target = ACTIONABLE_TARGETS.get(alias)
    if not target:
        return ""

    lines = [
        f"Target: {target['gene']} ({target['full_name']})",
        f"  Cancer types: {', '.join(target['cancer_types'])}",
        f"  Key variants: {', '.join(target['key_variants'][:6])}",
        f"  Targeted therapies: {', '.join(target['targeted_therapies'][:5]) or 'None approved'}",
        f"  Resistance: {', '.join(target['resistance_mutations'][:4]) or 'Unknown'}",
        f"  Pathway: {target['pathway']}",
        f"  Evidence level: {target['evidence_level']}",
        f"  {target['description']}",
    ]
    return "\n".join(lines)


def get_therapy_context(drug: str) -> str:
    """Return formatted knowledge context for a therapy."""
    key = drug.lower().replace(" ", "_").replace("-", "_")
    therapy = THERAPY_MAP.get(key)
    if not therapy:
        resolved = ENTITY_ALIASES.get(drug.lower(), "")
        therapy = THERAPY_MAP.get(resolved)
    if not therapy:
        return ""

    lines = [
        f"Therapy: {therapy['drug_name']} ({therapy['brand_name']})",
        f"  Category: {therapy['category']}",
        f"  Targets: {', '.join(therapy['targets'])}",
        f"  Approved indications: {'; '.join(therapy['approved_indications'][:4])}",
        f"  Mechanism: {therapy['mechanism']}",
        f"  Key trials: {', '.join(therapy['key_trials'][:4])}",
        f"  Resistance: {', '.join(therapy['resistance_mechanisms'][:3])}",
    ]
    return "\n".join(lines)


def get_resistance_context(drug: str) -> str:
    """Return resistance mechanisms for a therapy class."""
    mechanisms = RESISTANCE_MAP.get(drug)
    if not mechanisms:
        for key in RESISTANCE_MAP:
            if drug.lower() in key.lower():
                mechanisms = RESISTANCE_MAP[key]
                break
    if not mechanisms:
        return ""

    lines = [f"Resistance mechanisms for {drug}:"]
    for m in mechanisms:
        line = f"  - {m['mutation']} ({m['gene']}): ~{m['frequency']}"
        if m.get("bypass"):
            line += f" | bypass: {m['bypass']}"
        if m.get("next_line"):
            line += f" → {', '.join(m['next_line'])}"
        lines.append(line)
    return "\n".join(lines)


def get_pathway_context(pathway: str) -> str:
    """Return formatted context for an oncogenic pathway."""
    pw = PATHWAY_MAP.get(pathway.upper()) or PATHWAY_MAP.get(pathway)
    if not pw:
        for key, val in PATHWAY_MAP.items():
            if pathway.lower() in key.lower() or pathway.lower() in val["pathway_name"].lower():
                pw = val
                break
    if not pw:
        return ""

    lines = [
        f"Pathway: {pw['pathway_name']}",
        f"  Key genes: {', '.join(pw['key_genes'][:8])}",
        f"  Therapeutic targets: {'; '.join(pw['therapeutic_targets'][:4])}",
        f"  Cross-talk: {', '.join(pw['cross_talk'][:3])}",
        f"  {pw['clinical_relevance']}",
    ]
    return "\n".join(lines)


def get_biomarker_context(biomarker: str) -> str:
    """Return formatted context for a biomarker panel."""
    bm = BIOMARKER_PANELS.get(biomarker)
    if not bm:
        alias = ENTITY_ALIASES.get(biomarker.lower(), "")
        bm = BIOMARKER_PANELS.get(alias)
    if not bm:
        for key, val in BIOMARKER_PANELS.items():
            if biomarker.lower() in key.lower() or biomarker.lower() in val["name"].lower():
                bm = val
                break
    if not bm:
        return ""

    lines = [
        f"Biomarker: {bm['name']}",
        f"  Type: {bm['type']}",
        f"  Testing: {bm['testing_method']}",
        f"  Cutoff: {bm['clinical_cutoff']}",
        f"  Predictive for: {', '.join(bm['predictive_for'][:3])}",
        f"  Cancer types: {', '.join(bm['cancer_types'][:4])}",
        f"  Evidence level: {bm['evidence_level']}",
        f"  {bm['description']}",
    ]
    return "\n".join(lines)


def resolve_comparison_entity(raw: str) -> Optional[Dict]:
    """Resolve a raw entity string to a canonical entity for comparison.

    Searches targets, therapies, pathways, and biomarkers in that order.

    Returns:
        Dict with 'canonical', 'type', 'target' (gene if applicable), or None.
    """
    normalized = raw.strip()
    upper = normalized.upper()

    # Check entity aliases first
    alias = ENTITY_ALIASES.get(normalized.lower())
    if alias:
        normalized = alias
        upper = normalized.upper()

    # Check targets
    if upper in ACTIONABLE_TARGETS:
        t = ACTIONABLE_TARGETS[upper]
        return {"canonical": t["gene"], "type": "target", "target": t["gene"], "data": t}

    # Check therapies
    key = normalized.lower().replace(" ", "_").replace("-", "_")
    if key in THERAPY_MAP:
        t = THERAPY_MAP[key]
        return {"canonical": t["drug_name"], "type": "therapy", "target": None, "data": t}

    # Check pathways
    for pw_key, pw in PATHWAY_MAP.items():
        if upper == pw_key or normalized.lower() in pw["pathway_name"].lower():
            return {"canonical": pw["pathway_name"], "type": "pathway", "target": None, "data": pw}

    # Check biomarkers
    for bm_key, bm in BIOMARKER_PANELS.items():
        if upper == bm_key.upper() or normalized.lower() in bm["name"].lower():
            return {"canonical": bm["name"], "type": "biomarker", "target": None, "data": bm}

    return None


def get_comparison_context(entity_a: Dict, entity_b: Dict) -> str:
    """Generate formatted comparison context between two entities."""
    lines = [f"Comparison: {entity_a['canonical']} vs {entity_b['canonical']}"]

    if entity_a["type"] == "target" and entity_b["type"] == "target":
        a, b = entity_a["data"], entity_b["data"]
        lines.extend([
            f"\n{a['gene']}:",
            f"  Cancer types: {', '.join(a['cancer_types'][:4])}",
            f"  Therapies: {', '.join(a['targeted_therapies'][:3])}",
            f"  Pathway: {a['pathway']}",
            f"\n{b['gene']}:",
            f"  Cancer types: {', '.join(b['cancer_types'][:4])}",
            f"  Therapies: {', '.join(b['targeted_therapies'][:3])}",
            f"  Pathway: {b['pathway']}",
        ])
    elif entity_a["type"] == "therapy" and entity_b["type"] == "therapy":
        a, b = entity_a["data"], entity_b["data"]
        lines.extend([
            f"\n{a['drug_name']} ({a['brand_name']}):",
            f"  Targets: {', '.join(a['targets'])}",
            f"  Category: {a['category']}",
            f"  Indications: {'; '.join(a['approved_indications'][:3])}",
            f"\n{b['drug_name']} ({b['brand_name']}):",
            f"  Targets: {', '.join(b['targets'])}",
            f"  Category: {b['category']}",
            f"  Indications: {'; '.join(b['approved_indications'][:3])}",
        ])
    else:
        for entity in [entity_a, entity_b]:
            if entity["type"] == "target":
                lines.append(f"\n{get_target_context(entity['canonical'])}")
            elif entity["type"] == "therapy":
                lines.append(f"\n{get_therapy_context(entity['canonical'])}")
            elif entity["type"] == "pathway":
                lines.append(f"\n{get_pathway_context(entity['canonical'])}")
            elif entity["type"] == "biomarker":
                lines.append(f"\n{get_biomarker_context(entity['canonical'])}")

    return "\n".join(lines)
