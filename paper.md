---
title: 'MatchMyJob: An Open-Source O*NET-SOC Autocoder Using Fine-Tuned Sentence Embeddings and Field-Weighted Retrieval'
tags:
  - Python
  - natural language processing
  - occupational coding
  - sentence embeddings
  - workforce research
  - O*NET
authors:
  - name: Ankit Chandrakumar Katre
    orcid: 0009-0009-1406-4906
    corresponding: true
    affiliation: 1
  - name: Justin Wiegand
    orcid: 0000-0001-8151-385X
    affiliation: 2
affiliations:
  - name: San Diego State University, USA
    index: 1
  - name: Department of Management, Fowler College of Business, San Diego State University, USA
    index: 2
date: 12 October 2026
bibliography: paper.bib
---

# Summary

`MatchMyJob` is an open-source Python system that maps free-text survey job titles and descriptions to occupational codes from the O\*NET-SOC 2019 taxonomy (1,016 occupations). It fine-tunes a Sentence-BERT-style embedding model [@Reimers2019] (`BAAI/bge-small-en-v1.5`, @Xiao2023) with Multiple Negatives Ranking (MNR) loss [@Henderson2017] on 4,206 labeled pairs. It then scores each query against six O\*NET fields (official title, alternate titles, importance-weighted tasks, description, technology skills, and tools used) by weighted cosine similarity. The field weights are learned by differential evolution [@Storn1997] instead of being set by hand, which makes the scoring interpretable: official title receives 39.4% of the weight, alternate titles 29.0%, and tasks 20.2%. The system returns the top three occupations with confidence tiers (High $\geq$ 65%, Moderate 50--64.9%, Low < 50%) and is available as a Streamlit web application at <https://matchmyjob.streamlit.app>. The repository includes a pytest suite and continuous integration.

# Statement of need

Survey research in epidemiology, labor economics, psychology, and public health often requires every respondent's free-text job to be assigned a Standard Occupational Classification (SOC) code. Trained coders usually do this by hand, which is slow, costly, and hard to reproduce. Manual coding is also less consistent than is often assumed: a review reported agreement between 44% and 89% when different coders coded the same responses [@Mannetje2003], and agreement between trained coders falls from $\kappa = 0.84$ at the 2-digit level to moderate levels at the 4-digit level [@Schmitz2016].

Existing automated tools are closed-source (SOCcer [@Russ2016; @Russ2023], NIOCCS [@Schmitz2016], and the commercial O\*NET-SOC AutoCoder [@AutoCoder2020]), target non-U.S. taxonomies (`occupationMeasurement` for German KldB/ISCO [@Simson2023]; OccCANINE for HISCO [@Dahl2024]), or require interactive input from respondents. `MatchMyJob` targets researchers who need to code thousands of responses into the full O\*NET-SOC 2019 taxonomy without proprietary software or respondent interaction. Its confidence tiers support a hybrid workflow in which high-confidence matches are accepted and lower-confidence records go to expert review.

# State of the field

Rule-based and dictionary systems such as NIOCCS agree well with investigators at coarse levels but degrade at finer ones ($\kappa = 0.84$ at 2 digits, 0.56--0.70 at 4 digits; @Schmitz2016). An independent evaluation on two epidemiological cohorts found the same pattern for NIOCCS and for SOCcer v1 [@BucknerPetty2019]. Stacked ensembles such as SOCcer v2 improved agreement with expert codes to 73% at 2 digits and 50% at 6 digits [@Russ2023] but remain closed-source. A comparison of seven machine-learning algorithms found the best matched human coders about 56% of the time on German survey data [@Schierholz2021].

The O\*NET-SOC AutoCoder, developed by R.M. Wilson Consulting and licensed to state workforce agencies, targets the same taxonomy as `MatchMyJob`. It is closed-source and dictionary-driven, batch access is licensed per agency, and its accuracy figures are vendor-published [@AutoCoder2020]. Recent transformer-based work includes `occupationMeasurement`, an interactive R toolbox for German classifications [@Simson2023]; OccCANINE, a character-level transformer reaching 96% accuracy on historical HISCO codes with 15.8 million training pairs [@Dahl2024]; OPERAS, which exceeds human coders on ISCO-88 [@Hodzic2023]; and a BERT-ensemble classifier for UK SOC and O\*NET-SOC [@RonyPatman2025].

We built `MatchMyJob` rather than extend these tools for three reasons. First, no open-source system targets the full 1,016-code O\*NET-SOC 2019 taxonomy used in U.S. workforce research; SOCcer covers SOC-2010 and is closed-source. Second, `MatchMyJob` learns explicit per-field weights across all six O\*NET descriptor fields, an interpretable decomposition absent from systems that encode all occupational text as one vector, such as JobBERT [@Decorte2021] and OccCANINE. Third, the repository contains the knowledge-base construction, fine-tuning, weight-optimization, and evaluation code together with the model, learned weights, vector caches, and per-row evaluation output.

# Software design

`MatchMyJob` separates an offline build phase from online inference (\autoref{fig:architecture}).

![System architecture. The offline phase builds the knowledge base, fine-tunes the model, caches the field embeddings, and learns the field weights; the deployed application then loads the cached embeddings and weights, encodes each query, and ranks occupations.\label{fig:architecture}](architecture.png)

**Offline build.** Four scripts run in sequence. `master_data_creation.py` builds a knowledge base of 1,016 occupations from six O\*NET 30.1 source files, with task text weighted by O\*NET importance ratings. `finetune_model.py` fine-tunes `bge-small-en-v1.5` with Multiple Negatives Ranking loss (batch size 64, 3 epochs) on 4,206 pairs, built from 5,260 combined training rows after excluding 1,054 rows whose SOC code is outside the knowledge base; it holds out 20% of the labeled data for testing. `vectorize_kb.py` encodes the six knowledge-base fields, splitting long fields into 800-character chunks and mean-pooling them, and saves the tensors under `vectors/` in a cache named after the model (`kb_vectors_matchmyjob-finetuned.pt`). `optimize_weights.py` runs SciPy's differential evolution over the six weights, maximizing cosine-only top-1 accuracy on the training rows (bounded to $[0,1]$, normalized to sum to one; `maxiter=300`, `popsize=12`, seed 42). Evolution was chosen over gradient-based search because ranking is not differentiable. The learned weights are stored in `data/processed/optimal_weights.csv`: title 39.4%, alternate titles 29.0%, tasks 20.2%, description 10.2%, technology skills 0.6%, tools 0.6%. Skills and tools, which started at 15% and 5%, were driven below 1% each, which plausibly reflects that respondents write short, title-like text rather than task narratives.

**Evaluated scoring.** Accuracy is measured with `match_engine_eval.py`, which prefixes each query (job title plus description) with the BGE retrieval instruction, encodes it, and scores every occupation $o$ as

$$
\text{Score}(q,o) = \sum_{i=1}^{6} w_i \cos\!\big(\mathbf{e}_q,\mathbf{e}^{(i)}_o\big) + \gamma\,\widehat{\text{BM25}}(q,o) + \delta\,\big|T(q)\cap T(o)\big|,
$$

where $w_i$ are the learned field weights, $\widehat{\text{BM25}}$ is a BM25 keyword score over the occupation title, alternate titles, and description, normalized by its maximum over occupations, $T(\cdot)$ are the stemmed, stop-word-filtered word sets of the query title and the official occupation title, $\gamma = 0.03$, and $\delta = 0.05$. The two lexical terms act as tie-breakers between occupations whose embeddings are nearly identical.

**Deployed application.** The Streamlit application (`app.py`) offers single-title lookup and batch CSV/Excel upload. It loads the cached knowledge-base vectors and applies the same fine-tuned model, prefix, learned weights, and lexical terms (BM25 and the title-word boost, implemented once in `scoring.py`) as the evaluation, so it returns the same rankings as the evaluated pipeline. The scripted batch path, `match_engine.py`, uses the same scoring.

**Confidence tiers.** The top score, expressed as a percentage, is labeled High ($\geq 65$), Moderate (50--64.9), or Low (< 50). These tiers are heuristic score bands, not calibrated probabilities.

**Results.** On a held-out test set of 661 records, the evaluated pipeline achieves 41.8% top-1 accuracy (276/661), 60.1% top-3 accuracy (397/661), and 73.4% major-group (2-digit SOC) accuracy (485/661). The 95% binomial interval on top-1 accuracy is about $\pm 3.8$ percentage points. The summary and per-row predictions are in `data/output/`. We do not claim state-of-the-art accuracy; OccCANINE [@Dahl2024], for instance, is far more accurate on a different task and taxonomy. The contribution is an open, interpretable, reproducible pipeline for O\*NET-SOC 2019, since the embedding recipe itself (BGE with Multiple Negatives Ranking loss) is standard [@Xiao2023; @Reimers2019; @Henderson2017]. A 292-record subset of the held-out test split, restricted to records whose job title never appears in the training data, is included in `data/test/`. The full human-coded training and test spreadsheets are not redistributed in the repository and are available from the authors on request, so fine-tuning, weight optimization, and the full evaluation cannot be re-run from the repository alone; prediction and the application run without them.

# Research impact statement

`MatchMyJob` addresses a recurring processing task in job-centered survey research: matching participant-reported job titles to O\*NET-SOC codes so that O\*NET variables can be merged with job-title data. This task has repeatedly required substantial human coding. For example, @Wiegand2021 collected job titles from 3,806 employed MTurk respondents and used two independent raters to match each title to an O\*NET occupation. @Evanoff2014 had two raters independently assign SOC codes to all jobs reported across repeated surveys of 711 workers. @Hoff2021 had all job titles coded twice across two samples totaling 1,775 respondents. @Hotze2024 had three raters match job titles across two samples totaling 1,376 respondents and multiple time points. @Schwartz2025 had two coders match 1,350 job titles. At larger scale, survey data analyzed by @Basbug2025 contained 52,727 job application titles that coders assigned to SOC occupations.

The scale of this manual work is consequential. An experiment with five expert coders measured a median manual coding time of 60.6 seconds per job description [@Langezaal2025]. On that benchmark, the two coding passes for the 3,806 titles in the @Wiegand2021 dataset represent about 128 person-hours, before rater training or resolving disagreements. At \$29.81 per hour, the May 2025 U.S. median wage for social science research assistants [@BLS2026], this is roughly \$3,800 in direct coding labor.

`MatchMyJob`'s pipeline is released under an MIT license, so occupational assignments and model behavior can be inspected and reproduced, rather than relying on a hosted classifier whose model and training are unavailable. Its value is not limited to reproducing a single human-coded reference, because human coding is itself imperfect, as coder disagreements in several of these studies illustrate. Research also shows that even when automated and manual methods produce different detailed SOC codes, the linked O\*NET values can remain strongly concordant [@BucknerPetty2019]. `MatchMyJob` returns ranked candidates and confidence tiers so researchers can accept straightforward cases and direct ambiguous titles to human review, shifting effort from exhaustive duplicate lookup toward cases that need judgment. The same coding task appears in vocational psychology, personality research, occupational epidemiology, human resource management, and labor-market research, from studies with hundreds of respondents to datasets with more than 50,000 titles. `MatchMyJob` turns a recurring, study-specific manual procedure into a reusable and auditable workflow.

# AI usage disclosure

Claude Code and Claude (Anthropic) were used for code completion and refactoring during development and to help draft, edit, and fact-check this manuscript. The authors reviewed and revised all AI-assisted content, made the core design decisions, and take responsibility for the submission. All reported results were produced by running the code in the repository.

# References
