# Checkpoints

The trained weights are not stored in this repository (most are about 9.6 MB); they are
kept in the project archive and are available on request. This table maps each result in
the report to the checkpoint behind it, with the file's MD5 so that a copy can be checked.

| Role in the report | Run | Checkpoint file | Bytes | MD5 |
|---|---|---|---|---|
| HEIGHT, released checkpoint | `exp_heightpub` | `HEIGHT/checkpoints/237800.pt` (the HEIGHT authors' release) | 1,931,922 | `552b2d840a7924c29bdb9d9977b0ae18` |
| Occlusion teacher (Label_VAE), frozen | pretraining | `data/occlusion_vae/label_vae_ckpt/label_vae_weight_300.pth` | 4,603,158 | `efc4e99e9c0fb9ec2424fa365f063298` |
| Our system before the per-seed fine-tunes | `stage2_HEIGHT_intention_occlusion` | `checkpoints/357141.pt` | 9,577,398 | `ba11c7d6518b25be722dc07ad0a654ce` |
| Our system, seed 1 (Table I; also every frozen-policy result: perception, reciprocity, collisions, intrusions) | `exp_nospeak_s1` | `checkpoints/59522.pt` | 9,573,194 | `5b21509f6cc025850db5553d9004c7c7` |
| Our system, seed 2 | `exp_nospeak_s2` | `checkpoints/59522.pt` | 9,573,194 | `6792f9fd50692592cdebf05723e9b850` |
| Our system, seed 3 | `exp_nospeak_s3` | `checkpoints/59522.pt` | 9,573,194 | `be47b58821bc3242b44acd7abd20e3d6` |
| HEIGHT fine-tuned to 350 M steps | `exp_heightft` | `checkpoints/178570.pt` | 1,929,806 | `9456e9f9282c23f0ff79e39924761d36` |
| Retrained without forecasts, seed 1 | `exp_nopredict_s1` | `checkpoints/416665.pt` | 9,572,278 | `8612ee4b029bcb2d25095ecf1981b62c` |
| Retrained without forecasts, seed 2 | `exp_nopredict_s2` | `checkpoints/416665.pt` | 9,572,278 | `d375513c39b2d37c73605ac6018bac78` |
| Retrained without forecasts, seed 3 | `exp_nopredict_s3` | `checkpoints/416665.pt` | 9,572,278 | `b8db23bdd6ec39bf0369c0d9a8785ec9` |

Paths are relative to each run's directory (`data/<run>/`). Each `exp_nospeak_s*` fine-tune
loads `checkpoints/000000.pt` in its own run directory, the Stage-2 checkpoint copied there
before the fine-tune (see `training.load_path` in its configuration). The update counts
follow from 28 environments x 30 steps = 840 environment steps per update: 59,522 updates
is the 50 M-step fine-tune, 416,665 the 350 M-step retrain.
