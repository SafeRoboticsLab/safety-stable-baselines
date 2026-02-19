# Circle env
1. Rollout filter
```bash
python3 examples/replay_trained_sac_model_car_circle.py --model experiments/20251008_0029_SAC_CarCircle2_WithRolloutFilter_h100_vt0.0_circle-batch/best/best_model.zip --filter rollout --horizon 100 --velocity-threshold 0.0 --safety-model ./experiments/20251001_2228_SafetySAC_CarCircle2_2M/final/car_circle2.zip --record-video --initial-states initial_states/SafetyCarCircle2-v0_n10_20251104_110032.pkl --run-index 1
```

2. Value filter
```bash
python3 examples/replay_trained_sac_model_car_circle.py --model experiments/20251007_2245_SAC_CarCircle2_WithFilter_epsp0p100_circle-batch/final/car_circle2_sac_withfilter_epsp0p100.zip --filter value --epsilon 0.1 --safety-model experiments/20250926_1953_SafetySAC_CarCircle2_lr1em5/final/car_circle2.zip --record-video --initial-states initial_states/SafetyCarCircle2-v0_n10_20251104_110032.pkl --run-index 1
```

3. No filter
```bash
python3 examples/replay_trained_sac_model_car_circle.py --model experiments/20251007_2157_SAC_CarCircle2_circle-batch/final/car_circle2_sac.zip --record-video --initial-states initial_states/SafetyCarCircle2-v0_n10_20251104_110032.pkl --run-index 1
```
