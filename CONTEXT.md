# SparkMIM

SparkMIM selects the features that carry the most information about a target, using mutual information estimated with Spark. This glossary is the canonical language for that domain.

## Language

### Columns

**Feature**:
A column of the input dataframe that is neither the target nor a primary key.
_Avoid_: variable, predictor, attribute

**Target**:
The column being predicted.
_Avoid_: label, outcome

**Primary key**:
A column that identifies a record in the dataframe; never a candidate.
_Avoid_: index, id

### Selection

**Candidate**:
Every feature is a candidate from the start; no stage of the pipeline removes candidacy.
_Avoid_: variable, predictor

**Ranking**:
All candidates in descending order of univariate MI — the full candidate population, not just the advancing set.
_Avoid_: ordering, leaderboard

**Cutoff**:
The stage-1 boundary of the ranking: candidates above it — the significant top-K by MI — advance to the rest of the selection pipeline; the advancing set may be trimmed further to fit the joint tables.
_Avoid_: threshold, filter

**Advancing candidate**:
A candidate above the cutoff — significant and in the top-K by MI — that advances to joint-table evaluation.
_Avoid_: survivor, shortlist

**Criterion**:
The rule that scores a candidate's added information about the target, given the features already selected; one of JMIM, CMIM, mRMR, mIM, JMI.
_Avoid_: metric, method

**Criterion score**:
The value the criterion assigns to a candidate in a round; in round 1 it equals the candidate's univariate MI.
_Avoid_: score, round score

**Stopping threshold**:
The minimum criterion score required for a candidate to be accepted; the greedy loop halts when the best candidate falls below it.
_Avoid_: min_score, cutoff

**Selected feature**:
A candidate the greedy loop accepted, in acceptance order.
_Avoid_: chosen, kept, winner

### Estimation

**Histogram**:
MI estimation for continuous features by binning into quantile intervals and computing MI from the resulting contingency table.
_Avoid_: binning mode

**KSG**:
MI estimation for continuous features by the Kraskov–Stögbauer–Grassberger nearest-neighbour algorithm, without binning.
_Avoid_: kNN mode

**Subsample**:
A subset of the dataframe's rows used to estimate a stage's quantities; each estimation stage has its own (joint tables, permutation test, KSG).
_Avoid_: sample, subset

### Codes

**Code**:
The integer representation of a value of a feature or the target after preprocessing: a bin index for continuous values, a category code for categorical values, and a dedicated code for missing values.
_Avoid_: label, encoding

**Bin**:
A quantile interval of a continuous feature or target; the code of a continuous value is its bin index.
_Avoid_: bucket

**Missing mode**:
How missing values are handled in preprocessing: as a dedicated code, or by dropping the affected rows.
_Avoid_: imputation, missing strategy

### Tables

**Contingency table**:
A table of counts over the combinations of codes of two or more variables; the object from which MI is computed.
_Avoid_: crosstab, matrix

**Screening table**:
The contingency table of a candidate's codes against the target's codes, built in stage 1 for all candidates.
_Avoid_: univariate table

**Joint table**:
The contingency table of two candidates' codes against the target's codes, built in stage 2 for the advancing candidates.
_Avoid_: triple table

**Pair table**:
The contingency table of two candidates' codes, derived as the marginal of their joint table over the target.

### Screening

**Significance test**:
The per-candidate test of stage 1 that decides whether a candidate's MI is real; chi-squared or permutation.
_Avoid_: hypothesis test, p-value filter

**FDR control**:
The Benjamini–Hochberg adjustment of the significance p-values that bounds the expected false-discovery rate.
_Avoid_: BH, multiple-testing correction

**Significant candidate**:
A candidate whose p-value survives FDR control at the configured level; only these can pass the cutoff.
_Avoid_: true positive
