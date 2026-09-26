# Blocking evaluation: full, cand_val_50k, its val_ids queries


Queries: 50,000 S1 entities; pool: the FULL train S2+S3 of the same country. k = {'p1': 50, 'p2': 20, 'p4': 10}; P4 max per S1 = 100; P3 max block = 20; df cap = 0.005.


## Recall

EDA baseline (raw text, P1 alone, k=50): 0.9451 overall, US 0.9757, India 0.8998.

| candidates | group | true_pairs | pair_recall | entities | entity_all_found |
|---|---|---|---|---|---|
| union (final k) | ALL | 173019 | 0.9795 | 47253 | 0.9392 |
| union (final k) | US | 103490 | 0.9921 | 28335 | 0.9720 |
| union (final k) | India | 69529 | 0.9609 | 18918 | 0.8901 |
| union (final k) | Indic-script S2/S3 names | 11934 | 0.8695 | 6063 | 0.8054 |
| P1 alone, k=50 (EDA baseline set-up) | ALL | 173019 | 0.9474 | 47253 | 0.8556 |
| P1 alone, k=50 (EDA baseline set-up) | India | 69529 | 0.9000 | 18918 | 0.7422 |
| P1 alone, k=50 (EDA baseline set-up) | US | 103490 | 0.9792 | 28335 | 0.9313 |
| P1 alone, k=50 (EDA baseline set-up) | Indic-script S2/S3 names | 11934 | 0.7078 | 6063 | 0.6000 |

## Oracle ceiling (perfect matcher on these candidates)

Macro-F0.5 with precision 1 and recall = blocking recall per S1; singletons score 1 (nothing predicted); an S1 with no true match among its candidates scores 0.

| candidates | group | entities | singleton_share | oracle_macro_f05 |
|---|---|---|---|---|
| union (final k) | ALL | 50000 | 0.0549 | 0.9928 |
| union (final k) | India | 20002 | 0.0542 | 0.9855 |
| union (final k) | US | 29998 | 0.0554 | 0.9977 |
| P1 alone, k=50 | ALL | 50000 | 0.0549 | 0.9801 |

## Candidates per S1 entity (final set)

| queries | pairs | mean | p50 | p95 | max |
|---|---|---|---|---|---|
| 50000 | 4433888 | 88.6778 | 82.0000 | 137.0000 | 205 |

## Recall vs k (each pass alone)

| pass | k | pair_recall | recall_US | recall_India | mean_cands |
|---|---|---|---|---|---|
| p1 | 5 | 0.8247 | 0.8636 | 0.7668 | 4.9999 |
| p1 | 10 | 0.9122 | 0.9539 | 0.8501 | 9.9998 |
| p1 | 20 | 0.9318 | 0.9689 | 0.8767 | 19.9996 |
| p1 | 50 | 0.9474 | 0.9792 | 0.9000 | 49.9990 |
| p1 | 100 | 0.9474 | 0.9792 | 0.9000 | 49.9990 |
| p1 | 200 | 0.9474 | 0.9792 | 0.9000 | 49.9990 |
| p2 | 5 | 0.4613 | 0.4922 | 0.4154 | 4.9989 |
| p2 | 10 | 0.5535 | 0.5954 | 0.4910 | 9.9978 |
| p2 | 20 | 0.6164 | 0.6651 | 0.5440 | 19.9956 |
| p2 | 50 | 0.6164 | 0.6651 | 0.5440 | 19.9956 |
| p2 | 100 | 0.6164 | 0.6651 | 0.5440 | 19.9956 |
| p4 | 5 | 0.9570 | 0.9786 | 0.9248 | 17.9917 |
| p4 | 10 | 0.9626 | 0.9821 | 0.9337 | 32.5592 |
| p4 | 20 | 0.9626 | 0.9821 | 0.9337 | 32.5592 |

## Budget grid: recall-vs-mean-candidates frontier

| k1 | k2 | k4 | p3 | pair_recall | entity_all_found | mean_cands |
|---|---|---|---|---|---|---|
| 20 | 0 | 0 | False | 0.9318 | 0.8175 | 19.9996 |
| 20 | 5 | 0 | False | 0.9415 | 0.8414 | 22.7428 |
| 20 | 0 | 3 | False | 0.9636 | 0.8947 | 24.7151 |
| 20 | 5 | 3 | False | 0.9681 | 0.9071 | 27.3553 |
| 20 | 10 | 3 | False | 0.9694 | 0.9110 | 31.4899 |
| 20 | 5 | 5 | False | 0.9704 | 0.9137 | 32.2497 |
| 20 | 5 | 3 | True | 0.9724 | 0.9181 | 34.2862 |
| 20 | 10 | 3 | True | 0.9737 | 0.9220 | 38.4205 |
| 20 | 5 | 5 | True | 0.9741 | 0.9229 | 39.0808 |
| 20 | 10 | 5 | True | 0.9752 | 0.9262 | 43.1295 |
| 20 | 5 | 10 | True | 0.9760 | 0.9288 | 51.6690 |
| 20 | 20 | 5 | True | 0.9764 | 0.9297 | 52.1375 |
| 20 | 10 | 10 | True | 0.9770 | 0.9318 | 55.5776 |
| 30 | 20 | 5 | True | 0.9771 | 0.9318 | 60.6278 |
| 30 | 10 | 10 | True | 0.9776 | 0.9336 | 63.3216 |
| 20 | 20 | 10 | True | 0.9781 | 0.9348 | 64.3443 |
| 30 | 20 | 10 | True | 0.9787 | 0.9365 | 71.9979 |
| 50 | 20 | 10 | True | 0.9795 | 0.9392 | 88.6778 |
| 100 | 20 | 10 | True | 0.9795 | 0.9392 | 88.6778 |

## Marginal contribution (true pairs no other pass found)

| pass | true_pairs_found | found_only_by_this_pass | marginal_recall | marginal_recall_indic | candidates_from_pass |
|---|---|---|---|---|---|
| p1 | 163912 | 797 | 0.0046 | 0.0040 | 2499950 |
| p2 | 106657 | 785 | 0.0045 | 0.0002 | 999780 |
| p3 | 109131 | 453 | 0.0026 | 0.0296 | 490090 |
| p4 | 166550 | 1921 | 0.0111 | 0.0660 | 1627961 |

## 20 true matches still missed (3,541 missed pairs in total)

| s1_id | cand_id | country | s1_name_core | s1_addr_core | s1_addr_house_number | c_name_core | c_addr_core | c_addr_house_number | c_name_script | p1_cos | p1_rank | p2_rank | p1_rank_rev |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1-846775261 | S2-24197726 | India | balaji industries | 132 2nd stage industrial suburb yashwanthpur bangalore | 132 | ಬಾಲಾಜಿ ಇಂಡಸ್ಟ್ರೀಸ್ | b3/132 bangalore | b3/132 | Kannada |  |  |  |  |
| S1-279444989 | S2-951239534 | India | jai trading | mumbai agarwal estate mumbai city unit 4 chimat pada marol andheri east | 4 | privatejaitrading | agarwal estate mumbai |  | Latin |  |  |  |  |
| S1-815210935 | S2-207473615 | India | best alpha ventures | 77 1st floor 1st main 1st stage 5th phase mahaganapathinagar west of c hord road bangalore | 77 | ಬೆಸ್ಟ್ ಆಲ್ಫಾ ವೆಂಚರ್ಸ್ | 92 bangalore | 92 | Kannada |  |  |  |  |
| S1-312716450 | S2-292397840 | India | brave academy | a-1 indira shetty chl lbs marg ghatkopar west mumbai mumbai city | a-1 | vantageorbi | mumbai mumbai city a-1 | a-1 | Latin |  |  |  |  |
| S1-210007107 | S2-398114311 | India | sree jay business | flat b-1316 ashiana le residency 13 floor pocket p1 golf links nh24 ghaziabad | b-1316 | श्री जय बिजनेस | ghaziabad flat b b3/1316 | b3/1316 | Devanagari |  |  |  |  |
| S1-460573618 | S2-21709992 | US | beacon freight technologies | 289 anchor avenue warrenton | 289 | beacon technologies center |  |  | Latin |  |  |  |  |
| S1-433774950 | S2-27358581 | India | north enterprises | house 6 floor 1st block d 13 delhi north west delhi | 13 | नॉर्थ एंटरप्राइजेज | house 6 floor 1st block d 13 delhi north west delhi | 13 | Devanagari |  |  |  |  |
| S1-286613697 | S3-167613789 | US | ear nose and throat health | 2017 246 general booth boulevard virginia beach city | 2017 | ear nose and throat health |  |  | Latin |  |  |  |  |
| S1-627453996 | S3-817736848 | India | future impex | 201 2nd podium sanskruti splendour shiv vallabh road dahisar east mumbai | 201 | future center |  |  | Latin |  |  |  |  |
| S1-477006150 | S3-941984589 | India | city consulting | 1 shri shreeji park dsouza colony college road nashik | 1 | सिटी कंसल्टिंग | 1 nashik | 1 | Devanagari |  |  |  |  |
| S1-497213421 | S3-317256577 | US | polaris | 5061 lewiston drive unit condo indianapolis | 5061 | polaris services |  |  | Latin |  |  |  |  |
| S1-851725987 | S3-550793931 | US | safe viking | 1936 wade stedman road stedman | 1936 | safe services enterprises |  |  | Latin |  |  |  |  |
| S1-227560429 | S3-650352720 | India | swastik logistics | shiv ashish 2nd floor andheri kurla road sakinaka mumbai |  | swastik center |  |  | Latin |  |  |  |  |
| S1-115489451 | S3-598388142 | India | sai bharat infra | 617 regus magarpatta centre level 6 pentagon p-2 magarpatta city hadapsar pune | 617 | साईं भारत इंफ्रा | pune chakati mh block g-604 617 | g-604 | Devanagari |  |  |  |  |
| S1-115489451 | S3-457096109 | India | sai bharat infra | 617 regus magarpatta centre level 6 pentagon p-2 magarpatta city hadapsar pune | 617 | sai भारत इंफ्रा | block g-604 617 pune chakati mh | 617 | Devanagari |  |  |  |  |
| S1-187973623 | S3-599190401 | India | dream business | 1104-b5 majestique signature towers haveli pune | 1104-b5 | ड्रीम बिजनेस | b5 haveli pune | b5 | Devanagari |  |  |  |  |
| S1-982919391 | S3-549473155 | US | back alley bakery | 762 poker flats dickenson county | 762 | back alley balhmry |  |  | Latin |  |  |  |  |
| S1-255549687 | S3-636190635 | India | ggg india | shop 4 1st floor essem city center bhubaneswar khordha | 4 | quoariawex | shop 4 bhubaneswar khordha | 4 | Latin |  |  |  |  |
| S1-193625163 | S3-456645527 | India | jd dawakhana | 18/45 1st floor om sakthi nagar mangadu mangadu main road chennai | 18/45 | sri jd dfawkahana |  |  | Latin |  |  |  |  |
| S1-176786160 | S3-846704047 | India | eastern investment | 25/2 mgr nagar podanur coimbatore | 25/2 | ஈஸ்டர்ன் இன்வெஸ்ட்மெண்ட் | 23/2 coimbatore | 23/2 | Tamil |  |  |  |  |
