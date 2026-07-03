# Datasets

Networks used in Zhu et al. 2021 (Table 1), sourced from KONECT and SNAP. Start with the four small networks — they're the only ones where the paper (and this codebase, until FASTGREEDY is implemented) can compute the *exact* optimum via brute force, which is what makes SPGREEDY/FASTGREEDY's reproduction actually checkable.

| Network | n | m | Source |
|---|---|---|---|
| Karate | 34 | 78 | KONECT `ucidata-zachary` |
| Dolphins | 62 | 159 | KONECT `dolphins` |
| Netscience | 379 | 914 | KONECT `netscience` |
| Diseasome | 516 | 1188 | KONECT `moreno_disease` |
| GrQc | 4,158 | 13,422 | SNAP `ca-GrQc` |
| USgrid | 4,941 | 6,594 | KONECT `opsahl-powergrid` |
| ... | ... | ... | see Zhu et al. 2021 Table 1 for the remaining 15 networks (up to YoutubeSnap, 1.13M nodes) |

## Fetching

Not automated yet — `download.sh` is a stub. KONECT (http://konect.cc) and SNAP (https://snap.stanford.edu/data/) both provide direct edge-list downloads per network; once a dataset format is picked (edge list vs. `.mtx`), wire up `download.sh` to fetch + extract into `datasets/raw/<network>/` (gitignored — see `.gitignore`).

Do not commit raw dataset files to git — they belong in `datasets/raw/` which is gitignored; only this README and any small derived fixtures belong in version control.
