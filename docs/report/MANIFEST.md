# МАНИФЕСТ АРХИВА ДЛЯ АГЕНТА №2

**От:** Агент №1. **Дата:** 2026-10-24.

Архив содержит весь проект в поддерживаемых Агентом №2 MIME-типах:
`image/png`, `text/plain`, `text/markdown`, `text/csv`, `application/json`,
`application/pdf`. Файлы проекта в неподдерживаемых типах переименованы без
изменения содержимого: `.py` → `<имя>.py.txt`, `.ipynb` → `<имя>.ipynb.json`
(плюс читаемый `<имя>.ipynb.md`), служебные `.gitignore`/`.gitkeep` → `.txt`.

**Точка входа:** `00_INDEX.md` — состав, MIME, размеры, SHA-256, дерево пакета.
**Пояснение и запрос одобрения:** `docs/report/PACKET_2_TRANSFER_NOTE.md`.

Файлов в архиве (без этого манифеста): **75**.

Внутри архива нет ни одного файла в неподдерживаемом типе: расширения только `.png`, `.pdf`, `.csv`, `.json`, `.md`, `.txt`.

## Содержимое

| № | Файл | MIME | Байт | SHA-256 |
|---:|---|---|---:|---|
| 1 | `.gitignore.txt` | `text/plain` | 807 | `c2de6b6fef0de00f14151646068dd60c73b1a9712d4025bc62125ff0437e8f18` |
| 2 | `00_INDEX.md` | `text/markdown` | 15,054 | `bfba505ca50059b2ec2d92748b7784d6f02d7d4d69b7970ba36cf4d2b0cc5234` |
| 3 | `README.md` | `text/markdown` | 15,067 | `bc83c3682baa39ae9c85774d8915af385d76bfc3d28708367bde8185aa20d2f3` |
| 4 | `docs/A2_RESPONSE.md` | `text/markdown` | 20,595 | `d04f96acf2dacd2a4e361347ac05949fcc3a8196f84b5abad19c9e0edec4ed31` |
| 5 | `docs/ARCHITECTURE.md` | `text/markdown` | 19,003 | `42791b941890ef22a3091d24837b27e7c03207f78de3162392de5a9b8a156e2a` |
| 6 | `docs/QA_CHECKLIST.md` | `text/markdown` | 8,590 | `246c2fc8d5eeaeafff9e4039de66c249b2323c75e4d0ed47d14d7b3bb6732039` |
| 7 | `docs/SPEC_REQUEST_AGENT2.md` | `text/markdown` | 20,206 | `ecb872d05e04790ef0df69b34cec87fa4d51563afc9b118a13187e803dd676ff` |
| 8 | `docs/SPEC_REQUEST_AGENT2_NEXT.md` | `text/markdown` | 16,137 | `48c50946615ec0ca4fc52e2ee30ba067ad5b6e18fff21aeed5b8e4c005069d71` |
| 9 | `docs/TEST_STATE.md` | `text/markdown` | 5,452 | `14985803a0dcae607c550a3cc79a8a052be6542754bf4a23788317b386637b0b` |
| 10 | `docs/presentation/.gitkeep.txt` | `text/plain` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| 11 | `docs/report/.gitkeep.txt` | `text/plain` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| 12 | `docs/report/HISTOGRAM_CHECKS.md` | `text/markdown` | 926 | `d699177ec1b278c6deb62f716891a863ac7be68d82caf81d4e06b1bfb44a0d25` |
| 13 | `docs/report/PACKET_2_RESULTS.md` | `text/markdown` | 47,447 | `63f61927a4111cc438fa0cd6fea502a630af1733cc0f78d15c99011b706ebe9c` |
| 14 | `docs/report/PACKET_2_TRANSFER_NOTE.md` | `text/markdown` | 10,483 | `4bc3eb955c86e018a6314f324fec7587c9691e122fe290dadc12d526416f2809` |
| 15 | `docs/report/PACKET_HAMMING_RESULTS.md` | `text/markdown` | 12,062 | `5e57e7b8d53727c6344d0a5cde9e7eec057d9bbdc79243d9ce2c30605a83494a` |
| 16 | `docs/report/REFINEMENT_RESULTS.md` | `text/markdown` | 6,625 | `9fef4a03d791d6e638a3e85c65acfe924d7f50c0da0b6ef7637a4a788b667341` |
| 17 | `docs/report/VITERBI_VERIFICATION.md` | `text/markdown` | 3,362 | `207e66727f9c45fd9931df7789f8d5c75875320e84b282c2669f6f24f010f712` |
| 18 | `notebooks/01_research.ipynb.json` | `application/json` | 1,337,962 | `7fb66e4ad5a8860ce1f1d9fb2e795ab26655f1d33134ccf11701c6ba98cb3041` |
| 19 | `notebooks/01_research.ipynb.md` | `text/markdown` | 60,872 | `a6dc68a9db889eb61748354bead5dbfc66344aa65e8eee29fc5d0ea1791801db` |
| 20 | `requirements.txt` | `text/plain` | 707 | `cbd6e4bcaeb1ad085bedafe61ba06dc510cc676d140d5d5e4b7c8b8dffd845fb` |
| 21 | `results/data/.gitkeep.txt` | `text/plain` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| 22 | `results/data/ber_refinement.csv` | `text/csv` | 8,094 | `71620be42661e0904c7e99829efec1ffa8c98a6ebf1a1da56d1ec5f788a58e1d` |
| 23 | `results/data/ber_results.csv` | `text/csv` | 14,297 | `3665477b7bda3fae6bf3d92b822112b5981ce440cb478cb1db629aec882d0aa6` |
| 24 | `results/data/ber_results_extended.csv` | `text/csv` | 20,649 | `2e928a80093d418ad2cb0c8ae14c14d5535ef3568321650e649b9c88800a2416` |
| 25 | `results/data/cluster_intervals.csv` | `text/csv` | 10,923 | `35e57a5b081e457a08537fae4c76c7265d49142b70db4061485545a4fc4b62bf` |
| 26 | `results/data/energy_gain.csv` | `text/csv` | 4,743 | `3df8c3e15c172e42ac4d0839c814cae982342b09e0a878e7c487411610324088` |
| 27 | `results/data/histogram_checks.json` | `application/json` | 34,066 | `375158fe97aa668ba9b440afe3599b7e9a61aac0b005b8fd2e001694cf30dc1b` |
| 28 | `results/data/performance_probe.json` | `application/json` | 429 | `6c6ec1552dc4d99fa8654e0a97cce16da56b78388b9cbf5eba3f25489962531a` |
| 29 | `results/data/qa_checklist.json` | `application/json` | 2,961 | `667964b44840073279afb83a031f38cd68b0c0e9792f96b27ad71e6422dd1541` |
| 30 | `results/data/refinement_cluster.csv` | `text/csv` | 3,876 | `3928c3951fc5e940ba83d0816610a6661098613b91ebcd9074c03a4de1c3e0c7` |
| 31 | `results/data/refinement_gains.csv` | `text/csv` | 899 | `9108c9b36ea890caf409960323a2876ad43f0512447879c5fb9d3250e8540283` |
| 32 | `results/data/refinement_plan.json` | `application/json` | 816 | `12232fccf0d55b196e2c0c5c137dc079eadf7efd17501e3f1630d399dd84dd45` |
| 33 | `results/data/theory_curves.csv` | `text/csv` | 10,639 | `e651060fa6de4a338f75caedf82d0540e66940a3dc0eea5a62b7ccbd5c4733a7` |
| 34 | `results/data/viterbi_verification.json` | `application/json` | 2,496 | `59bcaf3b4c05a93ce9ec34d2340b0ea39c6a6269f79113dccaa4d6a048e84942` |
| 35 | `results/figures/.gitkeep.txt` | `text/plain` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| 36 | `results/figures/ber_comparison.pdf` | `application/pdf` | 46,949 | `0e2081a7341904c8ac787a5934f73c8473780081d90132d78f7140f2826a4825` |
| 37 | `results/figures/ber_comparison.png` | `image/png` | 498,143 | `5b7d98cbb41f9db0afc93c02a02a3f31de8ceb5896d0055eb0caddfce540e1f9` |
| 38 | `results/figures/ber_zoom.pdf` | `application/pdf` | 33,756 | `a302527d0addf70b1c068743b1b37f253c8eb476902394290b85b743b1af28c1` |
| 39 | `results/figures/ber_zoom.png` | `image/png` | 203,912 | `8e979c26e56d5d26de281d326c6582c5d974b0db6730cfd39f4faf3f04aba854` |
| 40 | `results/figures/channel_scheme.pdf` | `application/pdf` | 42,389 | `eed096381da5503b35ccdff34c97afae6218742d36ec1e7a1e83da045d312bd7` |
| 41 | `results/figures/channel_scheme.png` | `image/png` | 120,996 | `d87395844eb4d146e241c05af7e4f9303a01069e327763d84ae896b6de339bfe` |
| 42 | `results/figures/hamming_correction.pdf` | `application/pdf` | 18,638 | `96819a9767a26fac326329fa142fc09691cf266f3207829417a5d30657b32c92` |
| 43 | `results/figures/hamming_correction.png` | `image/png` | 78,804 | `51325e518be903730065043c02742e9fe7310bb6862b54d3c7e146824916469f` |
| 44 | `results/figures/state_diagram.pdf` | `application/pdf` | 24,454 | `39c766b12a0cf126247821edd42761e1d6822bc415669257d457b3f2ec27fab7` |
| 45 | `results/figures/state_diagram.png` | `image/png` | 159,742 | `42eeca5f69bb9638b93bcb33cfe178402aa87adebe06cb7a503d82772413f7f9` |
| 46 | `results/figures/trellis_example.pdf` | `application/pdf` | 15,891 | `715d245c792b794cfc8f5aee482e0cf509c8019131fa564afcb2b917807aff59` |
| 47 | `results/figures/trellis_example.png` | `image/png` | 85,885 | `7698135ee588d3fcfabf940e764beb36237b0b1b74905527942019d6a8df9a7f` |
| 48 | `scripts/build_notebook.py.txt` | `text/plain` | 44,030 | `52e5f2310e68268b78d956632a2005e00f2c4b3f7d7df2a2eae520ccc66e3e7c` |
| 49 | `scripts/check_histograms.py.txt` | `text/plain` | 5,122 | `374ae50a8586de8eba05019da11b2ac6965aa1d71ef673122ba9e7775392d326` |
| 50 | `scripts/check_transfer_packet.py.txt` | `text/plain` | 6,845 | `404172f2bb8225145ea23eab52f889245700165c69f23a357088b3017da9205f` |
| 51 | `scripts/make_figures.py.txt` | `text/plain` | 1,284 | `95c1b962d421bc898dafd777f71e30edd0b4ed6bb79baaae34d2c4539ee0eaea` |
| 52 | `scripts/make_report_tables.py.txt` | `text/plain` | 44,525 | `d681ab1775d0f9ecfc007dc4eba132a8d8c8c892dddb49230f7f28864d830713` |
| 53 | `scripts/make_transfer_archive.py.txt` | `text/plain` | 9,632 | `6ca92bcc64805b99e823839af9079330c84b386ea0fcc32c8cd61ef49e31ee51` |
| 54 | `scripts/make_transfer_packet.py.txt` | `text/plain` | 12,359 | `812555cd283012ba91bce012b96c2e6a908ca4892116c2222d285dd9c1b0371a` |
| 55 | `scripts/qa_check.py.txt` | `text/plain` | 5,584 | `f005471d639b1d66883f67bf8d1d18129bb77b3c7de0cb6d634addddb82e7468` |
| 56 | `scripts/refinement_analysis.py.txt` | `text/plain` | 12,844 | `dc1f7e670f3bacb951dcd8b2a7d30c4823f94d3451597125906ad2ae3bdba29d` |
| 57 | `scripts/run_experiment.py.txt` | `text/plain` | 3,815 | `3d63388bf1c974d2d037f4b139f6e63d04f3c49b28e0e0c5d7a92bbc71c81fb3` |
| 58 | `scripts/run_refinement.py.txt` | `text/plain` | 3,713 | `4421563eb3838d519437a26965e6c22f55e18d4f9e634bb8834a20e872beab1e` |
| 59 | `scripts/verify_hamming_spec.py.txt` | `text/plain` | 19,358 | `b632c3fb7b81e32f13347f01801bdad7dfa41e364e84fcd1adbae658f8bc37e6` |
| 60 | `scripts/verify_viterbi.py.txt` | `text/plain` | 12,242 | `b4dead0d6bef17892d0101c4e7965ff4734f270d0d843bca8c8b336c56e01532` |
| 61 | `src/__init__.py.txt` | `text/plain` | 693 | `9280e2a07d65164cb1705c0db9c74fcc4a69c42bf982d849b8bd1629d2ee2411` |
| 62 | `src/channel.py.txt` | `text/plain` | 6,768 | `74fa2118be40aabd5b0666fa6e126ebd6cebdae2fb205f52fd67d87c688ed2a5` |
| 63 | `src/convolutional.py.txt` | `text/plain` | 8,810 | `0c1b5b3dec1780c373a6c8bdec230dd858c0ff3c7dd5c03f93d5d6f303a8e78d` |
| 64 | `src/hamming.py.txt` | `text/plain` | 12,661 | `ce0638514b8935efd1d2e1d7269d6997e18650cf1e767aa9b886364b58560f76` |
| 65 | `src/metrics.py.txt` | `text/plain` | 30,554 | `0d6828dad742d973b094b7415860333c939c9037639fcf5a520b197fc3704a65` |
| 66 | `src/plotting.py.txt` | `text/plain` | 22,842 | `64c1079d29c32925408d28e65638243a5d7e37b0027be299cc27ceee54dd2306` |
| 67 | `src/simulation.py.txt` | `text/plain` | 19,623 | `43b3a9a00f4d8af2a14c800a7b622ba5326e41965a3c60f13d08d7f2db395fd4` |
| 68 | `src/viterbi.py.txt` | `text/plain` | 9,309 | `969fc75d8c339ddf39712eaa6bd9faab91e63906ebcd70e1697303397b5a7e53` |
| 69 | `tests/__init__.py.txt` | `text/plain` | 383 | `894c484c555fb852ee1f820d27474f326b67a568c88443def6cbd8286da434c1` |
| 70 | `tests/test_channel.py.txt` | `text/plain` | 11,267 | `cdb064c2ca350bd850a140fa873c4e6bf93178ce1b4084a229d3302859166c8c` |
| 71 | `tests/test_convolutional.py.txt` | `text/plain` | 6,468 | `45bba21bf8ec3c0c935e88f07c4749c942fc101f7b4849721e1f6f34257fcd0f` |
| 72 | `tests/test_hamming.py.txt` | `text/plain` | 12,068 | `ff0265b488b1528598259f4adb3517285ce6f5b6c5fe0105931200a6375c8534` |
| 73 | `tests/test_simulation.py.txt` | `text/plain` | 12,200 | `5159556b06bc6fc0e35b98b16e34fab48c714bff70c62b2d9ac1d6ffd4ed99e3` |
| 74 | `tests/test_viterbi.py.txt` | `text/plain` | 11,808 | `e8f09810a0415952ab058c7e476f0990fdb80b6593e94cc4df801c9be17387d1` |
| 75 | `results/data/transfer_check.json` | `application/json` | 512 | `15c630a1bfa2d1bb6a1b9885107245a1753ad263d46f5e007c13d8fc063ebe5a` |

## Проверка целостности

SHA-256 каждого файла указан в таблице выше и проверяется пересчётом:

```bash
python scripts/check_transfer_packet.py   # пакет: MIME, побайтовость, полнота
python scripts/make_transfer_archive.py   # архив: самопроверка после записи
```

SHA-256 самого архива печатается сборкой и сохраняется в
`results/data/transfer_archive.json`. Внутрь архива это значение не включается:
манифест — часть архива, поэтому его изменение меняло бы SHA-256 архива
(самоссылочность). Сборка детерминированная: повторный запуск без изменения
файлов даёт тот же SHA-256.
