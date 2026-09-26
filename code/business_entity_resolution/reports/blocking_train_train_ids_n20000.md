# Blocking evaluation: dev, train_ids sample


Queries: 20,000 S1 entities; pool: the FULL train S2+S3 of the same country. k = {'p1': 50, 'p2': 20, 'p4': 10}; P4 max per S1 = 100; P3 max block = 20; df cap = 0.005.


## Recall

EDA baseline (raw text, P1 alone, k=50): 0.9451 overall, US 0.9757, India 0.8998.

| candidates | group | true_pairs | pair_recall | entities | entity_all_found |
|---|---|---|---|---|---|
| union (final k) | ALL | 69291 | 0.9801 | 18859 | 0.9420 |
| union (final k) | US | 41460 | 0.9927 | 11272 | 0.9746 |
| union (final k) | India | 27831 | 0.9612 | 7587 | 0.8936 |
| union (final k) | Indic-script S2/S3 names | 4921 | 0.8722 | 2519 | 0.8178 |
| P1 alone, k=50 (EDA baseline set-up) | ALL | 69291 | 0.9483 | 18859 | 0.8584 |
| P1 alone, k=50 (EDA baseline set-up) | India | 27831 | 0.9014 | 7587 | 0.7463 |
| P1 alone, k=50 (EDA baseline set-up) | US | 41460 | 0.9798 | 11272 | 0.9338 |
| P1 alone, k=50 (EDA baseline set-up) | Indic-script S2/S3 names | 4921 | 0.7125 | 2519 | 0.6145 |

## Candidates per S1 entity (final set)

| queries | pairs | mean | p50 | p95 | max |
|---|---|---|---|---|---|
| 20000 | 1772152 | 88.6076 | 82.0000 | 137.0000 | 202 |

## Recall vs k (each pass alone)

| pass | k | pair_recall | recall_India | recall_US | mean_cands |
|---|---|---|---|---|---|
| p1 | 5 | 0.8261 | 0.7674 | 0.8655 | 4.9997 |
| p1 | 10 | 0.9139 | 0.8496 | 0.9570 | 9.9995 |
| p1 | 20 | 0.9325 | 0.8765 | 0.9701 | 19.9990 |
| p1 | 50 | 0.9483 | 0.9014 | 0.9798 | 49.9975 |
| p1 | 100 | 0.9563 | 0.9144 | 0.9843 | 99.9950 |
| p1 | 200 | 0.9631 | 0.9271 | 0.9873 | 199.9900 |
| p2 | 5 | 0.4635 | 0.4110 | 0.4988 | 4.9990 |
| p2 | 10 | 0.5572 | 0.4845 | 0.6060 | 9.9980 |
| p2 | 20 | 0.6191 | 0.5345 | 0.6759 | 19.9960 |
| p2 | 50 | 0.6844 | 0.6024 | 0.7395 | 49.9900 |
| p2 | 100 | 0.7341 | 0.6648 | 0.7806 | 99.9800 |
| p4 | 5 | 0.9585 | 0.9270 | 0.9797 | 22.8406 |
| p4 | 10 | 0.9641 | 0.9353 | 0.9834 | 45.8042 |
| p4 | 20 | 0.9684 | 0.9423 | 0.9860 | 92.5846 |

## Budget grid: recall-vs-mean-candidates frontier

| k1 | k2 | k4 | p3 | pair_recall | entity_all_found | mean_cands |
|---|---|---|---|---|---|---|
| 20 | 0 | 0 | False | 0.9325 | 0.8205 | 19.9990 |
| 20 | 5 | 0 | False | 0.9414 | 0.8433 | 22.7352 |
| 20 | 0 | 3 | False | 0.9650 | 0.8989 | 25.5950 |
| 20 | 5 | 3 | False | 0.9692 | 0.9111 | 28.2316 |
| 20 | 10 | 3 | False | 0.9705 | 0.9148 | 32.3606 |
| 20 | 5 | 5 | False | 0.9712 | 0.9166 | 33.4100 |
| 20 | 5 | 3 | True | 0.9734 | 0.9221 | 35.1566 |
| 20 | 10 | 3 | True | 0.9746 | 0.9258 | 39.2854 |
| 20 | 5 | 5 | True | 0.9749 | 0.9267 | 40.2441 |
| 20 | 10 | 5 | True | 0.9759 | 0.9297 | 44.2936 |
| 20 | 20 | 3 | True | 0.9760 | 0.9299 | 48.4395 |
| 20 | 5 | 10 | True | 0.9765 | 0.9313 | 51.5659 |
| 30 | 10 | 5 | True | 0.9769 | 0.9325 | 52.8963 |
| 20 | 20 | 5 | True | 0.9771 | 0.9333 | 53.3218 |
| 20 | 10 | 10 | True | 0.9775 | 0.9341 | 55.4739 |
| 30 | 20 | 5 | True | 0.9780 | 0.9361 | 61.7959 |
| 30 | 10 | 10 | True | 0.9783 | 0.9366 | 63.2069 |
| 20 | 20 | 10 | True | 0.9786 | 0.9374 | 64.2618 |
| 30 | 20 | 10 | True | 0.9793 | 0.9398 | 71.9006 |
| 50 | 20 | 10 | True | 0.9801 | 0.9420 | 88.6076 |
| 100 | 20 | 10 | True | 0.9808 | 0.9442 | 133.4959 |

## Marginal contribution (true pairs no other pass found)

| pass | true_pairs_found | found_only_by_this_pass | marginal_recall | marginal_recall_indic | candidates_from_pass |
|---|---|---|---|---|---|
| p1 | 66615 | 161 | 0.0023 | 0.0030 | 1199499 |
| p2 | 50664 | 195 | 0.0028 | 0.0000 | 490766 |
| p3 | 44004 | 112 | 0.0016 | 0.0189 | 196933 |
| p4 | 66942 | 467 | 0.0067 | 0.0482 | 750212 |

## 20 true matches still missed (1,380 missed pairs in total)

| s1_id | cand_id | country | s1_name_core | s1_addr_core | s1_addr_house_number | c_name_core | c_addr_core | c_addr_house_number | c_name_script | p1_cos | p1_rank | p2_rank | p1_rank_rev |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1-84032040 | S2-428247644 | India | al management | h 27 yamuna vihar phase 2 karmyogi kamla nagar agra | 2 | अल मैनेजमेंट | h 27 agra dayalbagh | 27 | Devanagari |  |  |  |  |
| S1-368505763 | S2-979358602 | India | high consulting | house 6 kola magri sai complex udaipur | 6 | हाई कंसल्टिंग | house 6 udaipur | 6 | Devanagari |  |  |  |  |
| S1-196791293 | S2-482817643 | India | arihant management | chennai j 41 bharathidasan street thirunagar jafferkanpet chennai | 41 | அரிஹந்த் மேனேஜ்மெண்ட் | chennai j b3/41 ashok nagar | b3/41 | Tamil |  |  |  |  |
| S1-259295595 | S2-480028175 | India | bombay trading | plot e south west delhi new delhi silver oak enclave 142 mandi road jaunapur | 142 | बॉम्बे ट्रेडिंग | hn 34 plot e new delhi south west delhi | 34 | Devanagari |  |  |  |  |
| S1-837757240 | S2-430974563 | India | sree healthcare | mumbai 99/5 rajgor chambersmaz floor surat street masjid e mumbai | 99/5 | sree healthcare |  |  | Latin | 0.3774 | 54 | 37 |  |
| S1-366953189 | S2-239667823 | US | pediatric dentistry clinic | 3114 81st avenue hyattsville | 3114 | pediatric clinic service |  |  | Latin |  |  |  |  |
| S1-181172933 | S2-214464661 | India | laxmi builders | 25 1st floor 6th cross leela palace road vruddhi complex kodihalli bangalore | 25 | ಲಕ್ಷ್ಮಿ ಬಿಲ್ಡರ್ಸ್ | 25 bangalore bangalore | 25 | Kannada |  |  |  |  |
| S1-968614538 | S2-149316626 | India | ram impex | house 17 pocket c-9 sector 8 rohini | c-9 | राम इम्पेक्स | house 17 delhi rohini | 17 | Devanagari |  |  |  |  |
| S1-887040268 | S2-249355952 | India | jai energy | 27 aliganj kotla mubarakpur new delhi south delhi | 27 | जय एनर्जी | c-777 27 south delhi new delhi | 27 | Devanagari |  |  |  |  |
| S1-531293556 | S2-120427759 | India | kolkata solutions | 3rd floor room 308 south city business park 770 anandapur kolkata kolkata howrah | 770 | kolkata solutions | 3rd floor howrah kolkata |  | Latin |  |  |  |  |
| S1-999343053 | S2-633952008 | US | primary care physicians | 136 nansemond pointe drive suffolk city | 136 | primary care phhsyicans |  |  | Latin |  |  |  |  |
| S1-757037882 | S2-576744820 | US | ridge committee | 1854 miners creek drive lincolnton | 1854 | ridge committee ridgecomm |  |  | Latin | 0.4228 | 70 |  |  |
| S1-399097157 | S3-898694344 | India | first shivam projects | centre shop 1 cts 6977 krishna business jalna | 1 | wexfaye | centre shop 1 jalna | 1 | Latin |  |  |  |  |
| S1-205183650 | S3-702227317 | India | pioneer jain foundation | 404 2nd floor anchor mall jaipur jaipur | 404 | पायोनियर जैन फाउंडेशन | 404 2nd floor jaipur | 404 | Devanagari |  |  |  |  |
| S1-212516524 | S3-743799339 | India | unique infrastructure | aligarh |  | unique ifnratnrucsue | aligarh opeosite a von dalsev |  | Latin | 0.2755 |  |  | 1 |
| S1-418719805 | S3-639876564 | India | indian it | kizhalathottam mylampatti karayampalayam coimbatore 2/150-a coimbatore | 2/150-a | இந்தியன் ஐடி | 150-a coimbatore | 150-a | Tamil |  |  |  |  |
| S1-631861985 | S3-724413578 | India | anand tech | plot 4 shivalik luxuria kalawad road rajkot | 4 | આનંદ ટેક પ્રા લિ | plot 4/9 rajkot | 4/9 | Gujarati |  |  |  |  |
| S1-12179263 | S3-52926798 | India | krish academy | ghaziabad b-39 surya enclave gt road ghaziabad | b-39 | krish aadey |  |  | Latin |  |  |  |  |
| S1-578106852 | S3-914991045 | India | black tech | 394/2 1st floor ss chambers ring road marathahalli bangalore | 394/2 | black tech |  |  | Latin | 0.3550 |  | 49 |  |
| S1-345254458 | S3-561202866 | India | dynamic properties | flat 12 dipak apartments 720/18 navi peth pune | 12 | डायनामिक प्रॉपर्टीज | door 12 mumbai pune | 12 | Devanagari |  |  |  |  |

## Capped retrieval vs exact search (P1, 2,000 queries)

| country | method | seconds | recall@10 | recall@50 | recall@100 |
|---|---|---|---|---|---|
| India | capped retrieval + exact re-rank | 47 | 0.8453 | 0.8950 | 0.9057 |
| India | exact search | 67 | 0.8450 | 0.8966 | 0.9084 |
| US | capped retrieval + exact re-rank | 49 | 0.9585 | 0.9814 | 0.9865 |
| US | exact search | 86 | 0.9595 | 0.9824 | 0.9880 |

## Pool size vs recall (P1 alone)

Train pool subsampled (x0.5, x0.81 = train/test ratio) or enlarged to x1.23 with test S2/S3 records of the same country (the test pool has ~23% more S2/S3 per S1). True pairs whose record left the pool are excluded.

| country | pool_factor | pool_size | recall@10 | recall@20 | recall@50 | recall@100 |
|---|---|---|---|---|---|---|
| India | 0.5000 | 2066945 | 0.8782 | 0.8991 | 0.9173 | 0.9301 |
| India | 0.8100 | 3347528 | 0.8590 | 0.8834 | 0.9059 | 0.9196 |
| India | 1.0000 | 4133346 | 0.8496 | 0.8765 | 0.9014 | 0.9144 |
| India | 1.2300 | 5082722 | 0.8436 | 0.8714 | 0.8967 | 0.9102 |
| US | 0.5000 | 3092423 | 0.9697 | 0.9779 | 0.9850 | 0.9881 |
| US | 0.8100 | 5010306 | 0.9612 | 0.9732 | 0.9816 | 0.9857 |
| US | 1.0000 | 6186873 | 0.9570 | 0.9702 | 0.9798 | 0.9843 |
| US | 1.2300 | 7607162 | 0.9543 | 0.9680 | 0.9785 | 0.9834 |
