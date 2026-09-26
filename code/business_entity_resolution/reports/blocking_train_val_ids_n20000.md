# Blocking evaluation: dev, val_ids sample


Queries: 20,000 S1 entities; pool: the FULL train S2+S3 of the same country. k = {'p1': 50, 'p2': 20, 'p4': 10}; P4 max per S1 = 100; P3 max block = 20; df cap = 0.005.


## Recall

EDA baseline (raw text, P1 alone, k=50): 0.9451 overall, US 0.9757, India 0.8998.

| candidates | group | true_pairs | pair_recall | entities | entity_all_found |
|---|---|---|---|---|---|
| union (final k) | ALL | 69387 | 0.9800 | 18891 | 0.9416 |
| union (final k) | US | 41705 | 0.9926 | 11341 | 0.9736 |
| union (final k) | India | 27682 | 0.9611 | 7550 | 0.8934 |
| union (final k) | Indic-script S2/S3 names | 4814 | 0.8681 | 2444 | 0.8097 |
| P1 alone, k=50 (EDA baseline set-up) | ALL | 69387 | 0.9488 | 18891 | 0.8596 |
| P1 alone, k=50 (EDA baseline set-up) | India | 27682 | 0.9021 | 7550 | 0.7499 |
| P1 alone, k=50 (EDA baseline set-up) | US | 41705 | 0.9798 | 11341 | 0.9326 |
| P1 alone, k=50 (EDA baseline set-up) | Indic-script S2/S3 names | 4814 | 0.7073 | 2444 | 0.6060 |

## Candidates per S1 entity (final set)

| queries | pairs | mean | p50 | p95 | max |
|---|---|---|---|---|---|
| 20000 | 1775509 | 88.7755 | 83.0000 | 137.0000 | 193 |

## Recall vs k (each pass alone)

| pass | k | pair_recall | recall_US | recall_India | mean_cands |
|---|---|---|---|---|---|
| p1 | 5 | 0.8281 | 0.8650 | 0.7724 | 5.0000 |
| p1 | 10 | 0.9147 | 0.9558 | 0.8526 | 10.0000 |
| p1 | 20 | 0.9331 | 0.9693 | 0.8786 | 20.0000 |
| p1 | 50 | 0.9488 | 0.9798 | 0.9021 | 50.0000 |
| p1 | 100 | 0.9572 | 0.9847 | 0.9156 | 100.0000 |
| p1 | 200 | 0.9634 | 0.9876 | 0.9269 | 200.0000 |
| p2 | 5 | 0.4636 | 0.4918 | 0.4210 | 4.9992 |
| p2 | 10 | 0.5539 | 0.5952 | 0.4917 | 9.9985 |
| p2 | 20 | 0.6158 | 0.6652 | 0.5413 | 19.9970 |
| p2 | 50 | 0.6825 | 0.7302 | 0.6106 | 49.9925 |
| p2 | 100 | 0.7330 | 0.7753 | 0.6693 | 99.9850 |
| p4 | 5 | 0.9579 | 0.9787 | 0.9266 | 23.5126 |
| p4 | 10 | 0.9640 | 0.9824 | 0.9362 | 46.9528 |
| p4 | 20 | 0.9678 | 0.9853 | 0.9416 | 93.4635 |

## Budget grid: recall-vs-mean-candidates frontier

| k1 | k2 | k4 | p3 | pair_recall | entity_all_found | mean_cands |
|---|---|---|---|---|---|---|
| 20 | 0 | 0 | False | 0.9331 | 0.8196 | 20.0000 |
| 20 | 5 | 0 | False | 0.9428 | 0.8451 | 22.7410 |
| 20 | 0 | 3 | False | 0.9646 | 0.8984 | 25.7018 |
| 20 | 5 | 3 | False | 0.9688 | 0.9103 | 28.3356 |
| 20 | 10 | 3 | False | 0.9700 | 0.9136 | 32.4796 |
| 20 | 5 | 5 | False | 0.9708 | 0.9158 | 33.4184 |
| 20 | 5 | 3 | True | 0.9733 | 0.9218 | 35.3171 |
| 20 | 10 | 3 | True | 0.9745 | 0.9251 | 39.4611 |
| 20 | 5 | 5 | True | 0.9747 | 0.9256 | 40.3083 |
| 20 | 10 | 5 | True | 0.9758 | 0.9286 | 44.3689 |
| 20 | 20 | 3 | True | 0.9758 | 0.9291 | 48.6132 |
| 30 | 5 | 5 | True | 0.9758 | 0.9291 | 48.9936 |
| 20 | 5 | 10 | True | 0.9761 | 0.9298 | 51.6944 |
| 30 | 10 | 5 | True | 0.9768 | 0.9318 | 52.9891 |
| 20 | 20 | 5 | True | 0.9770 | 0.9322 | 53.3993 |
| 20 | 10 | 10 | True | 0.9770 | 0.9323 | 55.6140 |
| 30 | 5 | 10 | True | 0.9771 | 0.9329 | 59.4995 |
| 30 | 20 | 5 | True | 0.9779 | 0.9351 | 61.8978 |
| 30 | 10 | 10 | True | 0.9779 | 0.9352 | 63.3725 |
| 20 | 20 | 10 | True | 0.9782 | 0.9359 | 64.4036 |
| 30 | 20 | 10 | True | 0.9790 | 0.9384 | 72.0718 |
| 50 | 20 | 5 | True | 0.9790 | 0.9387 | 79.9370 |
| 50 | 10 | 10 | True | 0.9790 | 0.9386 | 80.2355 |
| 50 | 20 | 10 | True | 0.9800 | 0.9416 | 88.7755 |
| 100 | 20 | 5 | True | 0.9801 | 0.9417 | 127.0014 |
| 100 | 20 | 10 | True | 0.9809 | 0.9442 | 133.6058 |

## Marginal contribution (true pairs no other pass found)

| pass | true_pairs_found | found_only_by_this_pass | marginal_recall | marginal_recall_indic | candidates_from_pass |
|---|---|---|---|---|---|
| p1 | 66721 | 169 | 0.0024 | 0.0029 | 1200082 |
| p2 | 50665 | 204 | 0.0029 | 0.0002 | 490732 |
| p3 | 44040 | 137 | 0.0020 | 0.0237 | 197648 |
| p4 | 67001 | 344 | 0.0050 | 0.0345 | 750098 |

## 20 true matches still missed (1,385 missed pairs in total)

| s1_id | cand_id | country | s1_name_core | s1_addr_core | s1_addr_house_number | c_name_core | c_addr_core | c_addr_house_number | c_name_script | p1_cos | p1_rank | p2_rank | p1_rank_rev |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1-173934251 | S2-138701797 | India | san industries | 16 nandalal mitra lane ground floor salkia howrah | 16 | 5an industries | 16 howrah | 16 | Latin |  |  |  |  |
| S1-539391374 | S2-729405440 | US | dental medicine | 75 hidden lane horner | 75 | dental center |  |  | Latin |  |  |  |  |
| S1-105889926 | S2-579859900 | India | sunrise media | plot b mumbai city mumbai flat 1201 floor 12 wing b sainath park chsl neelam nagar ap road gavanpada mulund e | 1201 | सनराइज मीडिया | plt b mumbai mumbai city |  | Devanagari |  |  |  |  |
| S1-941977108 | S2-257622166 | India | supreme media | gurgaon dlf qe unit 1202 12th floor millenium plaza sector 27 | 27 | सुप्रीम मीडिया | unit 1202 12th floir gurgaon coimbatore | 1202 | Devanagari |  |  |  |  |
| S1-417046167 | S2-107509215 | India | prime infotech | 4/176 eswaran koil street okkiyam thoraipakkam chennai | 4/176 | prime infotech |  |  | Latin | 0.3851 |  | 34 |  |
| S1-368545529 | S2-830893607 | India | alpha builders | hn 12 ward 31 nagpur road shanti nath mandir chhindwara | 12 | अल्फा बिल्डर्स | plot 554/8 hn 12 mokhed | 12 | Devanagari |  |  |  |  |
| S1-440401924 | S2-738274381 | India | high energy | hig 1 5th phase kphb colony tirumalagiri hyderabad | 1 | హై ఎనర్జీ | hig 1 tirumalagiri hyderabad | 1 | Telugu |  |  |  |  |
| S1-484248881 | S2-165754098 | India | east and sons | 70 kumanan street karungalpalayam erode | 70 | east and sbrn |  |  | Latin |  |  |  |  |
| S1-6132204 | S3-336945089 | India | apex impex | 20 plot 18 manikandan street iyappa nagar madipakkam kanchipuram saidapet kanchipuram | 18 | அபெக்ஸ் இம்பெக்ஸ் | kanchipuram 20 | 20 | Tamil |  |  |  |  |
| S1-311059637 | S3-473686726 | India | classic projects | 63 ground floor east anjaneya temple street basavanagudi bangalore | 63 | ಕ್ಲಾಸಿಕ್ ಪ್ರಾಜೆಕ್ಟ್ಸ್ | 63 bangalore | 63 | Kannada |  |  |  |  |
| S1-64912846 | S3-473632570 | US | education project | 157 greene avenue lindenhurst | 157 | education pgcrject |  |  | Latin |  |  |  |  |
| S1-882028671 | S3-754887039 | US | keystone | 85 mcadenville road belmont | 85 | keystone lp trading |  |  | Latin |  |  |  |  |
| S1-769821075 | S3-203296933 | India | pioneer infotech | flat 6 5396 to 5405 and 5407 to 5412 sai square maruti mandi chowk talegaon pune | 6 | पायोनियर इंफोटेक | flat g-6 pune region pune | g-6 | Devanagari |  |  |  |  |
| S1-304167648 | S3-458598852 | India | shyam business | 401 tulshi palace raghav society village saktasanala morvi rajkot | 401 | shyam બિઝનેસ | 401 ahmadabad | 401 | Gujarati |  |  |  |  |
| S1-7934738 | S3-533328702 | India | bharat trading | 301 ahmedabad maruti arcade shivranjani cross road satellite | 301 | ભારત ટ્રેડિંગ | 301 ahmedabad nehrunagar | 301 | Gujarati |  |  |  |  |
| S1-212112279 | S3-927833654 | India | nature producer | house 4 1st floor and 2nd floor block b nagar landmark nr metro piller 371 new delhi west delhi | 371 | nature plrduier | dor 759 house 4 west delhi new delhi | 759 | Latin |  |  |  |  |
| S1-225749302 | S3-221045735 | India | techno and brothers | unit 1 and 2 ground floor good earth business bay 1 sector 58 gurgaon bhondsi gurgaon | 2 | techno and service |  |  | Latin |  |  |  |  |
| S1-880116120 | S3-836198081 | US | lopez therapy | 516 2nd street unit 2 washington | 516 | synecto | 516 2nd street po box 9349 washignton | 516 | Latin | 0.2528 |  |  | 1 |
| S1-505771104 | S3-671160453 | India | india ashirwad mill | d-228 phase 7 focal point ludhiana | d-228 | india ashrorwad mill |  |  | Latin |  |  |  |  |
| S1-684367924 | S3-70125331 | India | om investment | 702 kritika tower chs ltd sion trombay road chembur e mumbai mumbai | 702 | ओम इन्वेस्टमेंट | 702/7 mumbai | 702/7 | Devanagari |  |  |  |  |
