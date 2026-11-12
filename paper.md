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
date: 10 November 2026
bibliography: paper.bib
---

# Summary

`MatchMyJob` is an open-source Python system that maps free-text survey job titles and descriptions to occupational codes from the O\*NET-SOC 2019 taxonomy (1,016 occupations) of the Occupational Information Network [O\*NET, @NCOD2026]. It fine-tunes a Sentence-BERT-style embedding model [@Reimers2019], `BAAI/bge-small-en-v1.5` [@Xiao2023], with Multiple Negatives Ranking (MNR) loss [@Henderson2017] on 4,206 labeled pairs. It then scores each query against six O\*NET fields (official title, alternate titles, importance-weighted tasks, description, technology skills, and tools used) by weighted cosine similarity. The field weights are learned by differential evolution [@Storn1997] instead of being set by hand, which makes the scoring interpretable: official title receives 39.4% of the weight, alternate titles 29.0%, and tasks 20.2%. The system returns the top three occupations with confidence tiers (High $\geq$ 65%, Moderate 50--64.9%, Low < 50%) and is available as a Streamlit web application at <https://matchmyjob.streamlit.app>. The repository includes a pytest suite and continuous integration.

# Statement of need

Survey research in organizational behavior, industrial-organizational psychology, labor economics, epidemiology, and public health routinely requires every respondent's free-text, self-reported job title to be assigned a Standard Occupational Classification (SOC) code. This allows researchers to append Bureau of Labor Statistics and O\*NET variables to their datasets and test hypotheses with these variables. The O\*NET content model spans abilities, skills, interests, work activities, and work context [@NCOD2026], supporting studies of, for example, physical exposures [@Evanoff2014] and interest--job fit [@Wiegand2021]. Trained raters usually assign these codes manually, which is slow, costly, and hard to reproduce. Manual coding is also less consistent than is often assumed: a review reported agreement between 44% and 89% when different coders coded the same responses [@Mannetje2003], and agreement between trained coders falls from $\kappa = 0.84$ at the 2-digit level to $\kappa = 0.58$ at the 4-digit level [@Schmitz2016].

Existing automated tools have limitations. Some are closed-source: SOCcer [Standardized Occupation Coding for Computer-assisted Epidemiological Research, @Russ2016; @Russ2023], NIOCCS [the National Institute for Occupational Safety and Health Industry and Occupation Computerized Coding System, @Schmitz2016], and the commercial O\*NET-SOC AutoCoder [@AutoCoder2020]. Others target non-U.S. taxonomies: `occupationMeasurement` for the German Klassifikation der Berufe (KldB) and the International Standard Classification of Occupations [ISCO, @Simson2023], and OccCANINE for the Historical International Classification of Occupations [HISCO, @Dahl2024]. Still others require interactive input from respondents. `MatchMyJob` targets researchers who need to code hundreds or thousands of self-reported job titles to O\*NET occupations without proprietary software or respondent interaction.

# State of the field

Rule-based and dictionary systems such as NIOCCS agree well with investigators at coarse levels but degrade at finer ones [$\kappa = 0.84$ at 2 digits, 0.56--0.70 at 4 digits, @Schmitz2016]. An independent evaluation on two epidemiological cohorts found the same pattern for NIOCCS and for SOCcer v1 [@BucknerPetty2019]. Stacked ensembles such as SOCcer v2 improved agreement with expert codes to 73% at 2 digits and 50% at 6 digits [@Russ2023] but remain closed-source. A comparison of seven machine-learning algorithms found the best matched human coders about 56% of the time on German survey data [@Schierholz2021].

The O\*NET-SOC AutoCoder, developed by R.M. Wilson Consulting and licensed to state workforce agencies, targets the same taxonomy as `MatchMyJob`. It is closed-source and dictionary-driven, and its accuracy figures are vendor-published [@AutoCoder2020]. Recent transformer-based work includes OccCANINE, a character-level transformer reaching 96% accuracy on historical HISCO codes with 15.8 million training pairs [@Dahl2024]; OPERAS, which exceeds human coders on ISCO-88 [@Hodzic2023]; and a BERT-ensemble classifier for UK SOC and O\*NET-SOC [@RonyPatman2025].

We built `MatchMyJob` rather than extend these tools for three reasons. First, no open-source system targets the full 1,016-code O\*NET-SOC 2019 taxonomy used in U.S. workforce research; SOCcer covers SOC-2010. Second, `MatchMyJob` learns explicit per-field weights across all six O\*NET descriptor fields, an interpretable decomposition absent from systems that encode all occupational text as one vector, such as JobBERT [@Decorte2021] and OccCANINE. Third, the `MatchMyJob` repository contains the knowledge-base construction, fine-tuning, weight-optimization, and evaluation code together with the model, weights, and vector caches. Taken together, these choices make `MatchMyJob` a reusable foundation for O*NET-SOC coding that other researchers can inspect, reproduce, and extend.

# Software design

`MatchMyJob` separates an offline build phase from online inference (\autoref{fig:architecture}).

![System architecture. The offline phase builds the knowledge base, fine-tunes the model, caches the field embeddings, and learns the field weights. The deployed application then loads the cached embeddings and weights, encodes each query, and ranks occupations.\label{fig:architecture}](architecture.png)

**Offline build.** Four scripts were run in sequence. The `master_data_creation.py` script built a knowledge base of 1,016 occupations from six O\*NET 30.1 source files, with task text weighted by O\*NET importance ratings. The `finetune_model.py` script fine-tuned `bge-small-en-v1.5` with Multiple Negatives Ranking loss (batch size 64, 3 epochs) on 4,206 pairs, built from 5,259 combined training rows after excluding 1,055 rows whose SOC code is outside the knowledge base. It held out 20% of the labeled data for testing. The `vectorize_kb.py` script encoded the six knowledge-base fields and saved the tensors under `vectors/` in a cache named after the model (`kb_vectors_matchmyjob-finetuned.pt`). The `optimize_weights.py` script ran SciPy's differential evolution over the six weights, maximizing cosine-only top-1 accuracy on the training rows (bounded to $[0,1]$, normalized to sum to one; `maxiter=300`, `popsize=12`, seed 42). The learned weights were stored in `data/processed/optimal_weights.csv`: title 39.4%, alternate titles 29.0%, tasks 20.2%, description 10.2%, technology skills 0.6%, tools 0.6%. Skills and tools fell below 1% each, plausibly because respondents write short, title-like text.

**Evaluated scoring.** Accuracy is measured with `match_engine_eval.py`, which prefixes each query (job title plus description) with the BGE retrieval instruction, encodes it, and scores every occupation $o$ as

$$
\text{Score}(q,o) = \sum_{i=1}^{6} w_i \cos\!\big(\mathbf{e}_q,\mathbf{e}^{(i)}_o\big) + \gamma\,\widehat{\text{BM25}}(q,o) + \delta\,\big|T(q)\cap T(o)\big|,
$$

where $w_i$ are the learned field weights, $\widehat{\text{BM25}}$ is a BM25 keyword score over the occupation title, alternate titles, and description, normalized by its maximum over occupations, $T(\cdot)$ are the stemmed, stop-word-filtered word sets of the query title and the official occupation title, $\gamma = 0.03$, and $\delta = 0.05$. The two lexical terms act as tie-breakers between occupations whose embeddings are nearly identical.

**Deployed application.** The Streamlit application (`app.py`) offers single job-title lookup and batch processing via CSV/Excel file upload. It loads the cached knowledge-base vectors and uses the same model, prefix, weights, and lexical terms (implemented once in `scoring.py`) as the evaluation, so it returns the same rankings. The batch script `match_engine.py` uses the same scoring.

**Confidence tiers.** The score of the top-ranked occupation, expressed as a percentage, is labeled High ($\geq 65$), Moderate (50--64.9), or Low (< 50). These are heuristic tiers based on review by the author team. The underlying scores are also provided directly in the application output.

**Results.** On a held-out test set of 661 records, the evaluated pipeline achieves 41.8% top-1 accuracy (276/661), 60.1% top-3 accuracy (397/661), and 73.4% major-group (2-digit SOC) accuracy (485/661). The 95% binomial interval on top-1 accuracy is $\pm 3.8$ percentage points. Our software provides an open, interpretable, and reproducible pipeline for O\*NET-SOC 2019, built on a standard embedding recipe [@Xiao2023; @Reimers2019; @Henderson2017]. The repository includes the model, learned weights, vector caches, a de-identified training dataset, and a held-out test dataset, so the full pipeline—fine-tuning, weight optimization, prediction, and the application—can be reproduced.

# Research impact statement

`MatchMyJob` addresses a recurring processing task in job-centered survey research: matching participant-reported job titles to O\*NET-SOC codes so that O\*NET variables can be merged with job-title data. This task has repeatedly required substantial human coding. For example, @Wiegand2021 had two independent raters match the job titles of 3,806 employed MTurk respondents to O\*NET occupations; @Evanoff2014 had two raters independently code all jobs reported across repeated surveys of 711 workers; @Hoff2021 had all job titles coded twice across two samples totaling 1,775 respondents; @Hotze2024 had three raters match job titles across two samples totaling 1,376 respondents; and @Schwartz2025 had two coders match 1,350 job titles. At a larger scale, @Basbug2025 analyzed 52,727 job application titles that coders assigned to SOC occupations. The author team is also using `MatchMyJob` to match reported job titles to O\*NET-SOC codes, so that O\*NET vocational interest profiles can be linked to the jobs in forthcoming research.

An experiment with five expert coders measured a median manual coding time of 60.6 seconds per job description [@Langezaal2025]. On that benchmark, the two coding passes for the 3,806 titles in the @Wiegand2021 dataset represent about 128 person-hours, before rater training or resolving disagreements. At \$29.81 per hour, the May 2025 U.S. median wage for social science research assistants [@BLS2026], this is roughly \$3,800 in direct coding labor.

`MatchMyJob`'s pipeline is released under an MIT license, so occupational assignments and model behavior can be inspected and reproduced, rather than relying on a hosted classifier whose model and training are unavailable. Its value is not limited to reproducing a human-coded reference, because human coding is itself imperfect. Research also shows that even when automated and manual methods produce different detailed SOC codes, the linked values for respective O\*NET variables often remain strongly concordant [@BucknerPetty2019]. `MatchMyJob` returns ranked candidates and confidence tiers so researchers can accept straightforward cases and direct ambiguous titles to human review, shifting effort from exhaustive duplicate lookup toward cases that need judgment. `MatchMyJob` turns a recurring, study-specific, human-intensive manual procedure into a reusable and auditable automated workflow.

# AI usage disclosure

Claude Code and Claude (Anthropic) were used for code completion and refactoring during development and to help draft, edit, and fact-check this manuscript. ChatGPT (OpenAI) was also used on portions of the writing. The authors reviewed and revised all AI-assisted content, made the core design decisions, and take responsibility for the submission. All reported results were produced by running the code in the repository.

# References
