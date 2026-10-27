# Findings

**Table 1. CNN and SVM Character Classification Performance**

| Evaluation metric | CNN | SVM |
| --- | --- | --- |
| Classification accuracy | 99.50% | 88.02% |
| Macro-averaged F1-score | 0.9945 | 0.8726 |
| Number of evaluation samples | 7176 | 7176 |

CNN training used 23026 glyphs; SVM training used 2500 glyphs. Both used the same source-group-disjoint test split.

**Table 2. Baybayin Word Identification Performance**

| Evaluation metric | Result |
| --- | --- |
| Number of images tested | 1001 |
| Failed images | 0 |
| Top-1 exact-match accuracy | 66.63% |
| Top-5 word identification accuracy | 71.13% |
| Mean CER against filename references | 0.1717 |

**Table 3. Sample Baybayin Word Recognition and English Translation**

| Processing stage | System output |
| --- | --- |
| Input image | dataset/Baybayin Word Images/1 Gwapo.PNG |
| Baybayin reading (approx.) | gwapo |
| Recognized Tagalog word | gwapo |
| English translation | handsome |
| Translation method | dictionary |
| Unverified words | None |
| Processing time | 0.001 s |

**Table 4. Tagalog Word Candidate Generation Results (oracle test)**

| Evaluation metric | Result |
| --- | --- |
| Words tested | 2000 |
| Candidate recall | 95.00% |
| Top-1 accuracy | 61.50% |
| Top-5 accuracy | 87.35% |
| Mean number of candidates | 18.34 |
| Median number of candidates | 10.0 |
| Processing time | 9.1 s |
| Tagalog words in the word list | 41350 |
| Number of prefixes | 125710 |
| Language-model tokens | 6193195 |
| Bigrams | 180571 |

**Table 5. Sample Errors in Tagalog Word Candidate Generation**

| Reference word | First predicted candidate | Second predicted candidate | Third predicted candidate |
| --- | --- | --- | --- |
| maibhan | madeha | mamihag | mabihag |
| pagligaw | palinga | paligaw | pagligaw |
| humalaw | humalay | humalaw | humadlang |
| maymgamali | magmaliw | magmuli | magmalas |
| haklit | hali | hale | halik |
| klub | lol | luh | la |
| sigit | sige | sigi | sigid |
| sandal | sara | sarap | saray |

**Sentence-image evaluation:** no checked Tagalog guide transcription is available, so sentence accuracy is not scored.
