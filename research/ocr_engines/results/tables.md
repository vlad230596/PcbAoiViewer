| engine | src | exact, turn given | exact, turn unknown | same+close (given) | CER (given) | R (80) | IC (17) | D/Q (5) | C/L/X (14) | blank FP |
|---|---|---|---|---|---|---|---|---|---|---|
| doctr_-db_resnet50+parseq | raw | **1** | 1 | 1 | 0.97 | 0 | 0 | 0 | 1 | 0/40 |
| doctr_-nodet+parseq | body | **18** | 10 | 52 | 0.57 | 18 | 0 | 0 | 0 |  |
| easyocr_-wl | body | **60** | 36 | 73 | 0.29 | 58 | 1 | 1 | 0 | 1/40 |
| easyocr_-wl | fmed | **54** | 33 | 68 | 0.37 | 53 | 0 | 0 | 1 | 1/40 |
| easyocr_-wl | frame | **60** | 32 | 74 | 0.31 | 58 | 1 | 0 | 1 | 1/40 |
| easyocr_-wl | raw | **59** | 33 | 72 | 0.33 | 58 | 1 | 0 | 0 | 1/40 |
| easyocr_-wl+clahe | frame | **44** | — | 65 | 0.40 | 43 | 1 | 0 | 0 |  |
| easyocr_-wl+clahe | raw | **43** | — | 63 | 0.40 | 43 | 0 | 0 | 0 |  |
| easyocr_-wl+close | frame | **59** | — | 75 | 0.32 | 57 | 1 | 0 | 1 |  |
| easyocr_-wl+close | raw | **62** | — | 75 | 0.31 | 61 | 0 | 0 | 1 |  |
| easyocr_-wl+otsu | frame | **39** | — | 52 | 0.52 | 38 | 1 | 0 | 0 |  |
| easyocr_-wl+otsu | raw | **33** | — | 54 | 0.55 | 32 | 1 | 0 | 0 |  |
| easyocr_-wl+up2 | frame | **50** | — | 68 | 0.39 | 48 | 1 | 0 | 1 |  |
| easyocr_-wl+up2 | raw | **50** | — | 68 | 0.40 | 49 | 0 | 0 | 1 |  |
| easyocr_-wl-mag2 | body | **41** | — | 77 | 0.33 | 41 | 0 | 0 | 0 |  |
| easyocr_-wl-mag2 | raw | **57** | — | 78 | 0.32 | 55 | 1 | 0 | 1 |  |
| fuse-mix-raw | fused | **84** | 78 | 93 | 0.14 | 76 | 4 | 1 | 3 |  |
| fuse-qwen2b-5img | fused | **92** | 81 | 101 | 0.08 | 79 | 6 | 1 | 6 | 0/40 |
| fuse-qwen2b-raw+frame | fused | **93** | 82 | 102 | 0.08 | 79 | 6 | 1 | 7 |  |
| fuse-qwen2b4b-raw+frame | fused | **93** | 85 | 102 | 0.08 | 79 | 5 | 1 | 8 |  |
| fuse-qwen4b-raw+frame | fused | **92** | 79 | 100 | 0.10 | 78 | 5 | 1 | 8 |  |
| fuse-vlm3-raw | fused | **88** | 74 | 98 | 0.11 | 79 | 3 | 1 | 5 |  |
| paddle_ | body | **46** | 40 | 58 | 0.46 | 42 | 1 | 1 | 2 | 1/40 |
| paddle_ | raw | **60** | 50 | 77 | 0.32 | 56 | 1 | 0 | 3 | 1/40 |
| rapid3-v5-mobile-en | body | **4** | 4 | 5 | 0.91 | 0 | 0 | 1 | 3 | 3/40 |
| rapid3-v5-mobile-en | mimg | **6** | 5 | 9 | 0.85 | 1 | 0 | 1 | 4 | 3/40 |
| rapid3-v5-mobile-en | raw | **34** | 27 | 46 | 0.58 | 32 | 0 | 0 | 2 | 3/40 |
| rapid3-v5-mobile-en-nodet | body | **51** | 42 | 75 | 0.38 | 51 | 0 | 0 | 0 |  |
| rapid3-v5-server | body | **52** | 39 | 69 | 0.35 | 40 | 4 | 3 | 5 | 6/40 |
| rapid3-v5-server | mimg | **50** | 41 | 71 | 0.35 | 44 | 1 | 2 | 3 | 6/40 |
| rapid3-v5-server | raw | **63** | 50 | 79 | 0.28 | 60 | 1 | 0 | 2 | 6/40 |
| rapid3-v6-medium | body | **47** | 41 | 73 | 0.35 | 39 | 2 | 1 | 5 | 6/40 |
| rapid3-v6-medium | mimg | **49** | 40 | 66 | 0.39 | 44 | 0 | 1 | 4 | 6/40 |
| rapid3-v6-medium | raw | **54** | 47 | 72 | 0.37 | 51 | 0 | 0 | 3 | 6/40 |
| rapid3-v6-small | body | **43** | 36 | 78 | 0.35 | 39 | 1 | 1 | 2 | 19/40 |
| rapid3-v6-small | mimg | **14** | 12 | 34 | 0.66 | 9 | 1 | 1 | 3 | 19/40 |
| rapid3-v6-small | raw | **27** | 20 | 48 | 0.56 | 25 | 0 | 0 | 2 | 19/40 |
| rapid_v1 | body | **11** | 9 | 18 | 0.80 | 7 | 1 | 1 | 2 | 0/40 |
| rapid_v1 | mimg | **30** | 26 | 41 | 0.61 | 25 | 1 | 1 | 3 | 0/40 |
| rapid_v1 | raw | **37** | 38 | 51 | 0.53 | 31 | 1 | 1 | 4 | 0/40 |
| tesseract-best-psm11-wl | body | **4** | 3 | 16 | 0.82 | 4 | 0 | 0 | 0 | 29/40 |
| tesseract-best-psm11-wl | mimg | **1** | 0 | 4 | 0.90 | 0 | 1 | 0 | 0 | 29/40 |
| tesseract-best-psm6 | body | **0** | — | 3 | 0.92 | 0 | 0 | 0 | 0 |  |
| tesseract-best-psm6 | mimg | **0** | — | 0 | 0.97 | 0 | 0 | 0 | 0 |  |
| vlm-florence2-large | frame | **63** | 50 | 84 | 0.25 | 58 | 2 | 1 | 2 | 1/40 |
| vlm-florence2-large | raw | **59** | 48 | 80 | 0.28 | 55 | 0 | 1 | 3 | 1/40 |
| vlm-got | frame | **72** | 44 | 93 | 0.20 | 66 | 2 | 1 | 3 | 19/40 |
| vlm-got | raw | **73** | 45 | 86 | 0.22 | 65 | 2 | 1 | 5 | 19/40 |
| vlm-paddleocrvl | frame | **75** | 66 | 89 | 0.20 | 68 | 2 | 1 | 4 | 27/40 |
| vlm-paddleocrvl | raw | **74** | 64 | 86 | 0.21 | 69 | 2 | 1 | 2 | 27/40 |
| vlm-qwen3vl-2b | body | **85** | 69 | 99 | 0.11 | 72 | 5 | 1 | 7 | 0/40 |
| vlm-qwen3vl-2b | fmed | **89** | 73 | 100 | 0.10 | 78 | 3 | 2 | 6 | 0/40 |
| vlm-qwen3vl-2b | frame | **92** | 82 | 101 | 0.09 | 78 | 6 | 1 | 7 | 0/40 |
| vlm-qwen3vl-2b | mimg | **82** | 74 | 92 | 0.14 | 71 | 4 | 1 | 6 | 0/40 |
| vlm-qwen3vl-2b | raw | **90** | 79 | 100 | 0.08 | 79 | 4 | 1 | 6 | 0/40 |
| vlm-qwen3vl-4b | frame | **90** | 77 | 101 | 0.10 | 78 | 4 | 1 | 7 | 0/40 |
| vlm-qwen3vl-4b | raw | **88** | 73 | 96 | 0.11 | 77 | 4 | 1 | 6 | 0/40 |
| vlm-trocr-base | body | **13** | 3 | 34 | 0.67 | 13 | 0 | 0 | 0 |  |

Timing (24 parts, raw, turn given, 4 CPU threads; GPU = RTX 5080 bf16):

| run | s / read |
|---|---|
| gpu-florence2-large | 0.10 |
| tesseract | 0.15 |
| rapid3-v6-small | 0.21 |
| gpu-qwen3vl-2b | 0.26 |
| easyocr | 0.35 |
| gpu-paddleocrvl | 0.40 |
| gpu-got | 0.60 |
| rapid_v1 | 0.67 |
| paddle | 1.15 |
| rapid3-v5-server | 1.70 |
| rapid3-v6-medium | 2.31 |
| cpu-florence2-large | 4.51 |
| cpu-qwen3vl-2b | 7.00 |
