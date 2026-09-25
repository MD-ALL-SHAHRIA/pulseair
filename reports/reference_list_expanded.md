# Expanded reference list

Compiled in Phase 1 of the thesis write-up. Two parts, kept distinguishable on purpose:

* **Existing entries** are reproduced **exactly** as supplied, unchanged in wording,
  spelling and punctuation. They are not re-verified here.
* **New entries** are marked **[NEW]**. Every one was confirmed against a bibliographic
  registry — Crossref for DOIs, the arXiv API for preprints, or the publisher's own
  page — and not from recollection. Titles, author order, venue, volume and year below
  are what the registry returned. Candidates that could not be fetched and confirmed
  were dropped; they are listed at the end with the reason.

> **Count discrepancy, flagged rather than resolved.** The brief describes the supplied
> list as *37 entries*; it contains **39**. All 39 are carried through unchanged.
> Totals below use the actual count.

**Total: 61 entries — 39 existing, 22 new.**

---

1. **[NEW]** Adiputra, I Nyoman Mahayasa; Wanchai, Paweena (2024). "CTGAN-ENN: a tabular GAN-based hybrid sampling method for imbalanced and overlapped data in customer churn prediction." *Journal of Big Data* 11(1), art. 121. DOI: 10.1186/s40537-024-00982-x
   *Relevance:* CTGAN for imbalance; the closest published comparator to this project's Phase 4 ablation.

2. **[NEW]** Ali, Md. Arfan; Bilal, Muhammad; Wang, Yu; Nichol, Janet E.; Mhawish, Alaa; Qiu, Zhongfeng; de Leeuw, Gerrit; Zhang, Yuanzhi; Zhan, Yating; Liao, Kuo; Almazroui, Mansour; Dambul, Ramzah; Shahid, Shamsuddin; Islam, M. Nazrul (2022). "Accuracy assessment of CAMS and MERRA-2 reanalysis PM2.5 and PM10 concentrations over China." *Atmospheric Environment* 288, 119297. DOI: 10.1016/j.atmosenv.2022.119297
   *Relevance:* Independent evidence that reanalysis products misestimate PM2.5 — background for Phase 11.

3. Angelopoulos & Bates 2023

4. **[NEW]** Azani Hassan Abadi, Maede; Wang, Shouyi (2026). "Interpretable Machine Learning for Air Pollution and Respiratory Health Prediction: A Socioeconomic Subgroup Analysis." arXiv:2607.17024
   *Relevance:* Recent interpretable-ML air-pollution work; subgroup framing parallels this project's protected-class rule.

5. Barber, Candès, Ramdas, Tibshirani 2023

6. Bergmeir & Benítez 2012

7. Breiman 2001

8. "Can Apps Make Air Pollution Visible?" 2019, Journal of Business Ethics

9. **[NEW]** Cerda-Mardini, Diego; Chandar, Sarath; Madathil, Sreenath (2026). "Consistent but Miscalibrated: Evaluating LLM Limitations for Risk Communication in Natural Language." arXiv:2607.03882
   *Relevance:* Directly motivates the external validator on the advisory layer: LLM hedging does not track probability.

10. Chawla, Bowyer, Hall, Kegelmeyer 2002

11. Chen & Guestrin 2016

12. **[NEW]** Cheung, Taizhen (2026). "When Directional Accuracy Lies: A Base-Rate-Honest Benchmark for LoRA-Adapted TimesFM on Equity Forecasting." arXiv:2607.12248
   *Relevance:* The persistence-floor argument in another domain — a base-rate-honest benchmark.

13. **[NEW]** Corani, Giorgio; Benavoli, Alessio; Demšar, Janez; Mangili, Francesca; Zaffalon, Marco (2017). "Statistical comparison of classifiers through Bayesian hierarchical modelling." *Machine Learning* 106(11), 1817–1837. DOI: 10.1007/s10994-017-5641-9
   *Relevance:* Alternative to null-hypothesis testing over folds; context for the Wilcoxon/Bonferroni choices.

14. CREA 2025

15. **[NEW]** Demšar, Janez (2006). "Statistical Comparisons of Classifiers over Multiple Data Sets." *Journal of Machine Learning Research* 7(1), 1–30.
   *Relevance:* The canonical reference for the Wilcoxon signed-rank test over folds, which this project uses.

16. **[NEW]** Ding, Tiffany; Fermanian, Jean-Baptiste; Salmon, Joseph (2025). "Conformal Prediction for Long-Tailed Classification." arXiv:2507.06867
   *Relevance:* Class-conditional vs marginal conformal coverage under long tails — the Phase 6 problem.

17. **[NEW]** Djupskås, Aslak; Stasik, Alexander Johannes; Riemer-Sørensen, Signe (2025). "Unreliable Uncertainty Estimates with Monte Carlo Dropout." arXiv:2512.14851
   *Relevance:* A caveat on MC dropout that belongs in the limitations chapter.

18. Gal & Ghahramani 2016

19. Gibbs & Candès 2021

20. **[NEW]** Gondal, Moazzam Umer; Qudous, Hamad ul; Farhan, Asma Ahmad (2025). "Beyond the Hype: Comparing Lightweight and Deep Learning Models for Air Quality Forecasting." arXiv:2512.09076
   *Relevance:* Independent corroboration of this project's central negative result on model complexity.

21. **[NEW]** Gondal, Moazzam Umer; Qudous, Hamad ul; Farhan, Asma Ahmad; Alamri, Sultan (2026). "Interpretable PM2.5 Forecasting for Urban Air Quality: A Comparative Study of Operational Time-Series Models." arXiv:2603.25495
   *Relevance:* Lightweight models remain competitive for urban PM2.5 — same finding, different method family.

22. **[NEW]** Gowri, L.; Sriya, Nitta Sai; Sowmya, Muppa; Lakshmi, Anna Reddy Harshitha; Amirtharajan, Rengarajan (2025). "Explainable AI for urban air quality: SHAP interpretation of stacked ensemble AQI forecast." *Theoretical and Applied Climatology* 156(10), art. 533. DOI: 10.1007/s00704-025-05741-3
   *Relevance:* Recent SHAP-on-AQI work; comparator for the Phase 6 attribution analysis.

23. **[NEW]** Gündüz, Ali Fatih; Şahin, Canan Batur (2026). "Synthetic Data Augmentation for Imbalanced Tabular Protein Subcellular Localization: A Comparative Study of SMOTE, CTGAN, TVAE, and TabDDPM Methods." *Applied Sciences* 16(8), 3694. DOI: 10.3390/app16083694
   *Relevance:* Head-to-head SMOTE vs CTGAN, the same control this project ran.

24. **[NEW]** Hasan, Kamrul; Rahman, Mustafizur; Akhter, Momotaj; Mohinuzzaman, Mohammad; Kayes, Imrul; Rahman, Shahanaj (2024). "A new dynamic approach using data-driven and machine learning models for forecasting particulate matter in Dhaka megacity." *Environmental Pollution and Management* 1, 235–247. DOI: 10.1016/j.epm.2024.11.005
   *Relevance:* Dhaka PM forecasting that includes a NAIVE comparator — rare, and directly relevant.

25. Hochreiter & Schmidhuber 1997

26. **[NEW]** Huam Ming Ken; Behjati, Mehran (2025). "Advancing Air Quality Monitoring: TinyML-Based Real-Time Ozone Prediction with Cost-Effective Edge Devices." arXiv:2504.03776. Also in *Selected Proceedings from the 2nd ICIMR 2024*, Lecture Notes in Networks and Systems vol. 1316, Springer, Singapore.
   *Relevance:* TinyML air-quality inference on edge hardware — the comparator for Phase 7.

27. **[NEW]** Islam, Abu Reza Md. Towfiqul; Al Awadh, Mohammed; Mallick, Javed; Pal, Subodh Chandra; Chakraborty, Rabin; Fattah, Md. Abdul; Ghose, Bonosri; Kakoli, Most. Kulsuma Akther; Islam, Md. Aminul; Naqvi, Hasan Raja; Bilal, Muhammad; Elbeltagi, Ahmed (2023). "Estimating ground-level PM2.5 using subset regression model and machine learning algorithms in Asian megacity, Dhaka, Bangladesh." *Air Quality, Atmosphere & Health* 16(6), 1117–1139. DOI: 10.1007/s11869-023-01329-w
   *Relevance:* Dhaka-specific ML baseline for the external-validation chapter.

28. Lee, Bayazid, Rosenthal, Khan, Arku, Barratt, Quayyum, Baumgartner 2025, Scientific Reports

29. Lemaître, Nogueira, Aridas 2017

30. Liang, Maimury, Chen, Juarez 2020

31. Liang, Xia, Ke, Wang, Wen, Zhang, Zheng, Zimmermann 2023 (AirFormer)

32. **[NEW]** Liu, Siwei; Zhou, Di Jody (2024). "Using cross-validation methods to select time series models: Promises and pitfalls." *British Journal of Mathematical and Statistical Psychology* 77(2), 337–355. DOI: 10.1111/bmsp.12330 (published online 7 December 2023)
   *Relevance:* Direct support for the rolling-origin protocol and against naive CV on ordered data.

33. Lundberg & Lee 2017

34. Lundberg, Erion, Lee 2018

35. Mao, Wang, Jiao, Zhao, Liu 2021

36. **[NEW]** Meyer, Marcel; Kaltenpoth, Sascha; Zalipski, Kevin; Müller, Oliver (2025). "Rethinking Evaluation in the Era of Time Series Foundation Models: (Un)known Information Leakage Challenges." arXiv:2510.13654
   *Relevance:* Evaluation-integrity argument for time-series benchmarks; supports Chapter 1.2.

37. "Mobile phones as monitors of personal exposure to air pollution" 2018, PLOS ONE

38. Özüpak, Alpsalaz, Aslan 2025

39. Pak, Ma, Ryu, Ryom, Juhyok, Pak, Pak 2020

40. Paszke et al. 2019

41. Patki, Wedge, Veeramachaneni 2016

42. Pedregosa et al. 2011

43. Pineda-Tobón, Espinosa-Bedoya, Branch-Bedoya 2024

44. Rahman & Meng 2024

45. Rahman, Begum, Hopke, Nahar, Newman, Thurston 2021

46. Ray 2022

47. RSC Environmental Science: Atmospheres 2026

48. Selva 2025

49. **[NEW]** Sharma, Sonali; Alaa, Ahmed M.; Daneshjou, Roxana (2025). "A Systematic Analysis of Declining Medical Safety Messaging in Generative AI Models." arXiv:2507.08030
   *Relevance:* Why safety wording cannot be left to the model — the rationale for the rule-based fallback.

50. **[NEW]** Singh, Manpreet; Srikantha, Akshatha; Lakhanpal, Shyamal (2026). "Cost-Sensitive Conformal Prediction and Human-in-the-Loop Abstention for Imbalanced High-Stakes Decision Support: A Multi-Domain Benchmark." arXiv:2607.27143
   *Relevance:* Benchmarks marginal vs Mondrian CP on imbalanced data — the closest comparator to Phase 6.

51. **[NEW]** Singh, Vivek; Singh, Sumit; Sharma, Nabin; Singh, Amarendra; Srivastava, Aman; Srivastava, Atul Kumar; Bisht, Deewan Singh; Patel, Kalpana; Singh, Neeti; Almazroui, Mansour; Choudhary, Arti (2026). "Estimation of surface PM2.5 over the Indo-Gangetic Basin using MERRA-2 reanalysis and machine learning." *Scientific Reports* 16(1), art. 13755. DOI: 10.1038/s41598-026-37934-9
   *Relevance:* Reanalysis underestimates South Asian PM2.5 — independent support for the Phase 11 result.

52. **[NEW]** Somvanshi, Shriyank; Islam, Md Monzurul; Chhetri, Gaurab; Chakraborty, Rohit; Mimi, Mahmuda Sultana; Shuvo, Sawgat Ahmed; Islam, Kazi Sifatul; Javed, Syed Aaqib; Rafat, Sharif Ahmed; Dutta, Anandi; Das, Subasish (2025). "From Tiny Machine Learning to Tiny Deep Learning: A Survey." arXiv:2506.18927; *ACM Computing Surveys* (2025).
   *Relevance:* Survey framing for the edge-deployment chapter and the ESP32 feasibility discussion.

53. Tashman 2000

54. U.S. EPA 2024

55. UCI Machine Learning Repository 2017/2019

56. Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin 2017

57. Wang 2025

58. Xu, Skoularidou, Cuesta-Infante, Veeramachaneni 2019

59. Zhao, Kunar, Birke, Chen 2021

60. Zhao, Kunar, Birke, Chen 2022

61. Zhou, Wang, Zhu, Qiao, Kang 2024

---

## Candidates dropped for failing verification

Per the brief, a candidate that cannot be independently confirmed is not included.

| Candidate | Why it was dropped |
| --- | --- |
| "Evaluating deep learning time series models for PM2.5 forecasting across diverse horizons" (PMC12907896) | The PubMed Central page is behind a reCAPTCHA and returned no article metadata. No DOI was recoverable from the search result, so title, authors, journal and year could not be confirmed from any registry. Topically relevant; re-check manually if wanted. |
| "Rethinking Aleatoric and Epistemic Uncertainty" (reported as ICML 2025) | Named in a search summary with no link or identifier. No arXiv ID or DOI could be resolved, so nothing could be verified. |

## Verification method

Each new entry was resolved through one of:

* **Crossref REST API** (`api.crossref.org/works/<doi>`) — publisher-deposited metadata.
* **arXiv API** (`export.arxiv.org/api/query?id_list=<id>`) — includes journal-ref where one exists.
* **The publisher's own article page**, where the work predates or sits outside both
  (used for Demšar 2006, confirmed on jmlr.org).

Two corrections came out of this and are worth recording, because both would have been
wrong had the search summaries been trusted:

* arXiv:2512.14851 is titled *"Unreliable Uncertainty Estimates with Monte Carlo Dropout"*,
  not "Unreliable Monte Carlo Dropout Uncertainty Estimation" as a search result rendered it.
* Liu & Zhou appeared as 2024 in search results and as 2023 in Crossref's `issued` field.
  Both are right: online-first 7 December 2023, print issue May 2024. The issue year is
  used above, with the online date noted.
