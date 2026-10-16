# PerforatedAI dendrites on BERT MLM: experiment log

Generated 2026-10-15 12:54 from the `*.log` files in `/home/rbrenner/PAI/PerforatedAI/transformers-perforated`.

Every run starts from a converged stage-1 baseline (LR restarts until a cycle gains < 0.1%). A cycle is one
3-epoch LR schedule (2e-5, linear decay), and a dendrite run adds one dendrite per cycle. Each cell is the best
eval loss of that cycle (fixed eval masks), and Δ is the difference against the restart-only control at the same
cycle, which got the same restarts without dendrites. ×N is `--pai_dendrite_lr_multiplier`. Placements: *wide* is
every encoder Linear plus the MLM head dense, *attention* is q/k/v/attention-output plus head, *FFN* is
intermediate/output dense plus head.

## mini / FineWeb-Edu

| Run | Params after last cycle | Cycle 1 | Cycle 2 | Cycle 3 | Cycle 4 | Cycle 5 | Cycle 6 | Cycle 7 | Cycle 8 | Cycle 9 | Cycle 10 | Cycle 11 | Cycle 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Control (restarts only)** | 11.2M | 2.7870 | 2.7856 | 2.7840 | 2.7820 | 2.7808 | 2.7790 | 2.7783 | 2.7769 | 2.7753 | 2.7737 | 2.7727 | 2.7717 |
| Head only, ×1 (init_by_main; default init gives identical numbers) | 11.4M | 2.7870 (-0.0000) | 2.7849 (-0.0006) | 2.7827 (-0.0013) |  |  |  |  |  |  |  |  |  |
| Head only, ×100 | 11.4M | 2.7863 (-0.0006) | 2.7836 (-0.0020) | 2.7803 (-0.0037) |  |  |  |  |  |  |  |  |  |
| Head + layer3, ×1 | 11.9M | 2.7862 (-0.0008) | 2.7845 (-0.0011) |  |  |  |  |  |  |  |  |  |  |
| Wide, ×1 | 21.0M | 2.7867 (-0.0002) | 2.7845 (-0.0011) | 2.7829 (-0.0011) |  |  |  |  |  |  |  |  |  |
| Wide, ×1, init_by_main | 21.0M | 2.7867 (-0.0002) | 2.7845 (-0.0011) | 2.7829 (-0.0011) |  |  |  |  |  |  |  |  |  |
| Head + layer3, ×10 | 12.2M | 2.7862 (-0.0008) | 2.7844 (-0.0012) | 2.7818 (-0.0022) |  |  |  |  |  |  |  |  |  |
| Head + layer3, ×100 | 12.2M | 2.7858 (-0.0012) | 2.7829 (-0.0027) | 2.7786 (-0.0055) |  |  |  |  |  |  |  |  |  |
| Wide, ×10 | 21.0M | 2.7867 (-0.0003) | 2.7843 (-0.0013) | 2.7814 (-0.0026) |  |  |  |  |  |  |  |  |  |
| Wide, ×30, 6 dendrites | 30.9M | 2.7858 (-0.0012) | 2.7770 (-0.0086) | 2.7630 (-0.0210) | 2.7455 (-0.0366) | 2.7265 (-0.0543) | 2.7062 (-0.0728) |  |  |  |  |  |  |
| No dendrites: 9606 extra FFN neurons per layer (as many parameters as 6 wide dendrites), ×30 (stopped when cycle 1 got worse) | 30.9M | 2.8039 (+0.0169) |  |  |  |  |  |  |  |  |  |  |  |
| No dendrites: 9606 extra FFN neurons per layer (as many parameters as 6 wide dendrites), ×1 | 30.9M | 2.7862 (-0.0008) | 2.7835 (-0.0021) | 2.7813 (-0.0027) | 2.7786 (-0.0035) | 2.7765 (-0.0044) | 2.7732 (-0.0058) |  |  |  |  |  |  |
| No dendrites: 9606 extra FFN neurons per layer (as many parameters as 6 wide dendrites), ×3 | 30.9M | 2.7890 (+0.0020) | 2.7867 (+0.0011) | 2.7850 (+0.0010) | 2.7830 (+0.0010) | 2.7816 (+0.0007) | 2.7794 (+0.0003) |  |  |  |  |  |  |
| Wide, ×100, 6 dendrites | 30.9M | 2.7829 (-0.0041) | 2.7717 (-0.0139) | 2.7592 (-0.0248) | 2.7464 (-0.0356) | 2.7329 (-0.0479) | 2.7211 (-0.0579) |  |  |  |  |  |  |
| Wide, ×100, init_by_main | 21.0M | 2.7830 (-0.0040) | 2.7718 (-0.0138) | 2.7587 (-0.0253) |  |  |  |  |  |  |  |  |  |
| Wide, ×300 | 21.0M | 2.7855 (-0.0015) | 2.7817 (-0.0039) | 2.7789 (-0.0051) |  |  |  |  |  |  |  |  |  |
| Wide, ×1000 | 21.0M | 2.7873 (+0.0004) | 2.7867 (+0.0011) | 2.7864 (+0.0024) |  |  |  |  |  |  |  |  |  |
| Wide, ×100, linear (identity) dendrites | 14.4M | 2.7883 (+0.0013) | 2.7890 (+0.0034) |  |  |  |  |  |  |  |  |  |  |
| Wide, ×100 on dendrite output weights only | 21.0M | 2.7870 (+0.0000) | 2.7851 (-0.0005) | 2.7835 (-0.0005) |  |  |  |  |  |  |  |  |  |
| Wide, ×100 on dendrite input weights only | 21.0M | 2.7866 (-0.0004) | 2.7833 (-0.0023) | 2.7790 (-0.0050) |  |  |  |  |  |  |  |  |  |
| Wide, dendrites at 2e-3 with main network frozen (LR 2e-7) | 21.0M | 2.7853 (-0.0017) | 2.7751 (-0.0105) | 2.7642 (-0.0198) |  |  |  |  |  |  |  |  |  |
| FFN only, ×100 | 17.8M | 2.7850 (-0.0019) | 2.7770 (-0.0086) | 2.7673 (-0.0167) |  |  |  |  |  |  |  |  |  |
| Attention only, ×100, 6 dendrites | 18.1M | 2.7853 (-0.0017) | 2.7772 (-0.0084) | 2.7684 (-0.0156) | 2.7606 (-0.0214) | 2.7519 (-0.0290) | 2.7446 (-0.0344) |  |  |  |  |  |  |
| Attention only in top 2 layers (2-3), ×100 | 13.0M | 2.7849 (-0.0020) | 2.7787 (-0.0069) | 2.7721 (-0.0119) |  |  |  |  |  |  |  |  |  |
| Attention only in bottom 2 layers (0-1), ×100 | 13.0M | 2.7859 (-0.0011) | 2.7811 (-0.0045) | 2.7773 (-0.0067) |  |  |  |  |  |  |  |  |  |
| Attention only in top layer (3), ×100 | 12.2M | 2.7857 (-0.0013) | 2.7798 (-0.0058) | 2.7759 (-0.0081) |  |  |  |  |  |  |  |  |  |
| Wide in top layer (3), ×100 | 13.8M | 2.7839 (-0.0031) | 2.7774 (-0.0082) | 2.7681 (-0.0159) |  |  |  |  |  |  |  |  |  |
| Wide in top 2 layers (2-3), ×100 | 16.2M | 2.7842 (-0.0028) | 2.7760 (-0.0096) | 2.7619 (-0.0221) |  |  |  |  |  |  |  |  |  |
| Wide in top layer (3), ×30, 6 dendrites | 16.4M | 2.7859 (-0.0010) | 2.7800 (-0.0056) | 2.7692 (-0.0148) | 2.7578 (-0.0242) | 2.7458 (-0.0351) | 2.7347 (-0.0443) |  |  |  |  |  |  |
| Wide in top layer (3), ×30, 12 dendrites | 21.9M | 2.7859 (-0.0010) | 2.7800 (-0.0056) | 2.7692 (-0.0148) | 2.7578 (-0.0242) | 2.7458 (-0.0351) | 2.7347 (-0.0443) | 2.7236 (-0.0547) | 2.7124 (-0.0645) | 2.7044 (-0.0709) | 2.6941 (-0.0796) | 2.6871 (-0.0857) | 2.6778 (-0.0940) |
| No dendrites: 10199 extra FFN neurons in layer 3 (as many parameters as 6 top-layer dendrites), ×30 (stopped after cycle 3) | 16.4M | 2.7903 (+0.0034) | 2.7892 (+0.0036) | 2.7879 (+0.0039) |  |  |  |  |  |  |  |  |  |
| No dendrites: 10199 extra FFN neurons in layer 3 (as many parameters as 6 top-layer dendrites), ×0.3 | 16.4M | 2.7864 (-0.0006) | 2.7846 (-0.0010) | 2.7827 (-0.0013) | 2.7803 (-0.0018) | 2.7786 (-0.0022) | 2.7763 (-0.0027) |  |  |  |  |  |  |
| No dendrites: 10199 extra FFN neurons in layer 3 (as many parameters as 6 top-layer dendrites), ×1 | 16.4M | 2.7862 (-0.0007) | 2.7845 (-0.0011) | 2.7827 (-0.0013) | 2.7803 (-0.0017) | 2.7785 (-0.0023) | 2.7761 (-0.0029) |  |  |  |  |  |  |
| No dendrites: 10199 extra FFN neurons in layer 3 (as many parameters as 6 top-layer dendrites), ×3 | 16.4M | 2.7872 (+0.0002) | 2.7857 (+0.0001) | 2.7841 (+0.0001) | 2.7822 (+0.0002) | 2.7809 (+0.0001) | 2.7790 (-0.0000) |  |  |  |  |  |  |
| Wide in top 2 layers (2-3), ×30, 6 dendrites | 21.3M | 2.7865 (-0.0005) | 2.7804 (-0.0052) | 2.7661 (-0.0179) | 2.7522 (-0.0299) | 2.7361 (-0.0447) | 2.7206 (-0.0584) |  |  |  |  |  |  |
| Attention only in top 2 layers (2-3), ×30, 6 dendrites | 14.8M | 2.7865 (-0.0005) | 2.7810 (-0.0046) | 2.7753 (-0.0087) | 2.7673 (-0.0148) | 2.7587 (-0.0221) | 2.7509 (-0.0282) |  |  |  |  |  |  |
| Attention only in top layer (3), ×30, 6 dendrites | 13.2M | 2.7869 (-0.0000) | 2.7816 (-0.0039) | 2.7769 (-0.0071) | 2.7716 (-0.0104) | 2.7648 (-0.0160) | 2.7584 (-0.0206) |  |  |  |  |  |  |

## mini / WikiText-103

| Run | Params after last cycle | Cycle 1 | Cycle 2 | Cycle 3 |
|---|---|---|---|---|
| **Control (restarts only)** | 11.2M | 2.3965 | 2.3954 | 2.3917 |
| Wide, ×100 | 21.0M | 2.3905 (-0.0060) | 2.3737 (-0.0217) | 2.3585 (-0.0332) |
| Wide, ×30 | 21.0M | 2.3933 (-0.0032) | 2.3723 (-0.0231) | 2.3488 (-0.0429) |
| Attention only, ×100 | 14.6M | 2.3916 (-0.0049) | 2.3843 (-0.0111) | 2.3737 (-0.0180) |

## small / FineWeb-Edu

| Run | Params after last cycle | Cycle 1 | Cycle 2 | Cycle 3 | Cycle 4 | Cycle 5 | Cycle 6 |
|---|---|---|---|---|---|---|---|
| **Control (restarts only)** | 28.8M | 2.3233 | 2.3207 | 2.3207 | 2.3207 | 2.3207 | 2.3207 |
| Attention only, ×30 | 42.3M | 2.3241 (+0.0008) | 2.3205 (-0.0002) | 2.3160 (-0.0046) |  |  |  |
| Attention only, ×100, 6 dendrites | 55.9M | 2.3240 (+0.0007) | 2.3202 (-0.0005) | 2.3161 (-0.0045) | 2.3129 (-0.0077) | 2.3104 (-0.0103) | 2.3078 (-0.0129) |
| Attention only in top 2 layers (2-3), ×30 | 35.9M | 2.3221 (-0.0011) | 2.3201 (-0.0006) | 2.3180 (-0.0026) |  |  |  |
| Wide in top layer (3), ×30, 6 dendrites | 49.5M | 2.3224 (-0.0009) | 2.3190 (-0.0017) | 2.3156 (-0.0050) | 2.3124 (-0.0083) | 2.3060 (-0.0146) | 2.3007 (-0.0200) |
| Wide in top layer (3), ×100, 6 dendrites | 49.5M | 2.3223 (-0.0010) | 2.3190 (-0.0017) | 2.3165 (-0.0042) | 2.3155 (-0.0051) | 2.3113 (-0.0093) | 2.3086 (-0.0120) |
| Attention only in top layer (3), ×30, 6 dendrites | 36.8M | 2.3223 (-0.0010) | 2.3199 (-0.0007) | 2.3166 (-0.0041) | 2.3151 (-0.0055) | 2.3122 (-0.0084) | 2.3090 (-0.0117) |

## mini / FineWeb-Edu, 2M training samples per epoch (4× longer cycles, same baseline and eval set)

| Run | Params after last cycle | Cycle 1 | Cycle 2 | Cycle 3 | Cycle 4 | Cycle 5 | Cycle 6 |
|---|---|---|---|---|---|---|---|
| **Control (restarts only)** | 11.2M | 2.7505 | 2.7410 | 2.7342 | 2.7287 | 2.7239 | 2.7197 |
| Wide in top layer (3), ×30, 6 dendrites | 16.4M | 2.7358 (-0.0146) | 2.7094 (-0.0315) | 2.6858 (-0.0484) | 2.6634 (-0.0653) | 2.6436 (-0.0803) | 2.6271 (-0.0926) |
| No dendrites: 10199 extra FFN neurons in layer 3 (as many parameters as 6 top-layer dendrites), ×1 | 16.4M | 2.7475 (-0.0030) | 2.7347 (-0.0063) | 2.7211 (-0.0132) | 2.7067 (-0.0220) | 2.6934 (-0.0305) | 2.6814 (-0.0382) |
| No dendrites: 10199 extra FFN neurons in layer 3 (as many parameters as 6 top-layer dendrites), ×3 | 16.4M | 2.7504 (-0.0001) | 2.7407 (-0.0003) | 2.7334 (-0.0008) | 2.7271 (-0.0017) | 2.7211 (-0.0028) | 2.7156 (-0.0041) |

## small / FineWeb-Edu, 2M training samples per epoch (4× longer cycles, same baseline and eval set)

| Run | Params after last cycle | Cycle 1 | Cycle 2 | Cycle 3 | Cycle 4 | Cycle 5 | Cycle 6 |
|---|---|---|---|---|---|---|---|
| **Control (restarts only)** | 28.8M | 2.2785 | 2.2681 | 2.2604 | 2.2562 | 2.2510 | 2.2454 |
| Wide in top layer (3), ×30, 6 dendrites | 49.5M | 2.2785 (-0.0000) | 2.2700 (+0.0019) | 2.2626 (+0.0022) | 2.2571 (+0.0008) | 2.2515 (+0.0006) | 2.2476 (+0.0023) |

## mini / CNN/DailyMail

| Run | Params after last cycle | Cycle 1 | Cycle 2 | Cycle 3 | Cycle 4 | Cycle 5 | Cycle 6 |
|---|---|---|---|---|---|---|---|
| **Control (restarts only)** | 11.2M | 2.4139 | 2.4119 | 2.4102 | 2.4078 | 2.4055 | 2.4033 |
| Wide in top layer (3), ×30, 6 dendrites | 16.4M | 2.4022 (-0.0117) | 2.3821 (-0.0299) | 2.3619 (-0.0483) | 2.3434 (-0.0645) | 2.3256 (-0.0799) | 2.3091 (-0.0942) |
| No dendrites: 10199 extra FFN neurons in layer 3 (as many parameters as 6 top-layer dendrites), ×1 | 16.4M | 2.4125 (-0.0014) | 2.4100 (-0.0020) | 2.4070 (-0.0033) | 2.4019 (-0.0059) | 2.3952 (-0.0103) | 2.3873 (-0.0160) |

## mini / FineWeb-Edu, seed 7 (different data order, dendrite init and eval masks)

| Run | Params after last cycle | Cycle 1 | Cycle 2 | Cycle 3 | Cycle 4 | Cycle 5 | Cycle 6 |
|---|---|---|---|---|---|---|---|
| **Control (restarts only)** | 11.2M | 2.7792 | 2.7769 | 2.7746 | 2.7736 | 2.7715 | 2.7694 |
| Wide in top layer (3), ×30, 6 dendrites | 16.4M | 2.7795 (+0.0004) | 2.7731 (-0.0038) | 2.7628 (-0.0117) | 2.7519 (-0.0216) | 2.7401 (-0.0314) | 2.7292 (-0.0401) |

## mini / FineWeb-Edu, from the pretrained checkpoint (no stage 1; the control is stage 1 itself)

| Run | Params after last cycle | Cycle 1 | Cycle 2 | Cycle 3 | Cycle 4 | Cycle 5 | Cycle 6 |
|---|---|---|---|---|---|---|---|
| **Control (restarts only)** | 11.2M | 2.8645 | 2.8453 | 2.8324 | 2.8226 | 2.8156 | 2.8092 |
| Wide, ×100, 6 dendrites | 30.9M | 2.8555 (-0.0089) | 2.8206 (-0.0247) | 2.7947 (-0.0377) | 2.7725 (-0.0502) | 2.7535 (-0.0620) | 2.7382 (-0.0710) |
| Attention only in top 2 layers (2-3), ×30, 6 dendrites | 14.8M | 2.8641 (-0.0003) | 2.8398 (-0.0055) | 2.8208 (-0.0116) | 2.8027 (-0.0199) | 2.7870 (-0.0285) | 2.7736 (-0.0356) |

