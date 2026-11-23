# ApacheJIT Data Dictionary

All entries below describe the actual 18-column training dataset.

| Column | Data type | Meaning | Training use |
|---|---|---|---|
| commit_id | string | Unique commit identifier | Excluded: identifier |
| project | string | Apache project name | Excluded: metadata |
| buggy | bool | Original defect target (`True`/`False`) | Excluded as a feature; converted to target 0/1 |
| fix | bool | Fix-status field | Excluded: target-adjacent metadata |
| year | int64 | Year in which the commit occurred | Excluded: chronological metadata |
| author_date | int64 | Commit author timestamp | Excluded: chronology metadata |
| la | int64 | Lines added | Included |
| ld | int64 | Lines deleted | Included |
| nf | int64 | Number of modified files | Included |
| nd | int64 | Number of modified directories | Included |
| ns | int64 | Number of modified subsystems | Included |
| ent | float64 | Entropy of modified code | Included |
| ndev | float64 | Number of developers | Included |
| age | float64 | Average age of modified files | Included |
| nuc | float64 | Number of unique changes | Included |
| aexp | int64 | Author experience | Included |
| arexp | float64 | Author recent experience | Included |
| asexp | float64 | Author subsystem experience | Included |
