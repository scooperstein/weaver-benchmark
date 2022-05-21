# weaver-benchmark

[`Weaver`](https://github.com/hqucms/weaver-core) configurations for ML benchmark tasks

The materials are prepared for the CMS ML Documentation: https://cms-ml.github.io/documentation/inference/particlenet.html

## Pre-processing for weights

Weaver can be run locally on CPUs only to produced a new `data-config` file containing weights for the chosen binning and selection. Everytime you change the input datasets or the selection or the binning defintion or the classes, these files need to be reporduced i.e.:
* Delete them from both git and local directory, since the hash-key of the file will be generated differently from execution to execution.
* Rerun the training aborting it after the production of the data-config file after the pre-processing of the inputs.
* Example of commands:
  ```sh
  python3 ../weaver-core/weaver/train.py  --data-train '/eos/cms/store/group/phys_exotica/monojet/rgerosa/ParticleNetUL/NtupleTrainingAK4Skimmed/tree_*root'  --network-config networks/particle_net_ak4_pf_sv_ext.py --data-config data/ak4_points_pf_sv_mass_decorr.yaml --model-prefix /tmp/rgerosa/ --gpus '' --batch-size 100 --log /tmp/rgerosa/weight_ak4_notau.log --num-workers 1
  python3 ../weaver-core/weaver/train.py  --data-train '/eos/cms/store/group/phys_exotica/monojet/rgerosa/ParticleNetUL/NtupleTrainingAK4Skimmed/tree_*root'  --network-config networks/particle_net_ak4_pf_sv_ext.py --data-config data/ak4_points_pf_sv_mass_decorr_tau.yaml --model-prefix /tmp/rgerosa/ --gpus '' --batch-size 100 --log /tmp/rgerosa/weight_ak4_tau.log --num-workers 1
  python3 ../weaver-core/weaver/train.py  --data-train '/eos/cms/store/group/phys_exotica/monojet/rgerosa/ParticleNetUL/NtupleTrainingAK4Skimmed/tree_*root'  --network-config networks/particle_net_ak4_pf_sv_ext.py --data-config data/ak4_points_pf_sv_mass_decorr_taumuel.yaml --model-prefix /tmp/rgerosa/ --gpus '' --batch-size 100 --log /tmp/rgerosa/weight_ak4_taumuel.log --num-workers 1
  python3 ../weaver-core/weaver/train.py  --data-train '/eos/cms/store/group/phys_exotica/monojet/rgerosa/ParticleNetUL/NtupleTrainingAK4Skimmed/tree_*root'  --network-config networks/particle_net_ak4_pf_sv_class_reg.py --data-config data/ak4_points_pf_sv_mass_decorr_tau_class_reg.yaml --model-prefix /tmp/rgerosa/ --gpus '' --batch-size 100 --log /tmp/rgerosa/weight_ak4_tau_classreg.log --num-workers 1
  python3 ../weaver-core/weaver/train.py  --data-train '/eos/cms/store/group/phys_exotica/monojet/rgerosa/ParticleNetUL/NtupleTrainingAK4Skimmed/tree_*root'  --network-config networks/particle_net_ak4_pf_sv_class_reg.py --data-config data/ak4_points_pf_sv_mass_decorr_taumuel_class_reg.yaml --model-prefix /tmp/rgerosa/ --gpus '' --batch-size 100 --log /tmp/rgerosa/weight_ak4_taumuel_classreg.log --num-workers 1
  python3 ../weaver-core/weaver/train.py  --data-train '/eos/cms/store/group/phys_exotica/monojet/rgerosa/ParticleNetUL/NtupleTrainingAK8Skimmed/tree_*root'  --network-config networks/particle_ne_ak8_pf_sv_ext.py --data-config data/ak8_points_pf_sv_mass_decorr.yaml --model-prefix /tmp/rgerosa/ --gpus '' --batch-size 100 --log /tmp/rgerosa/weight_ak8_notau.log --num-workers 1
  python3 ../weaver-core/weaver/train.py  --data-train '/eos/cms/store/group/phys_exotica/monojet/rgerosa/ParticleNetUL/NtupleTrainingAK8Skimmed/tree_*root'  --network-config networks/particle_net_ak8_pf_sv_ext.py --data-config data/ak8_points_pf_sv_mass_decorr_tau.yaml --model-prefix /tmp/rgerosa/ --gpus '' --batch-size 100 --log /tmp/rgerosa/weight_ak8_tau.log --num-workers 1
  python3 ../weaver-core/weaver/train.py  --data-train '/eos/cms/store/group/phys_exotica/monojet/rgerosa/ParticleNetUL/NtupleTrainingAK8Skimmed/tree_*root'  --network-config networks/particle_net_ak8_pf_sv_class_reg.py --data-config data/ak8_points_pf_sv_mass_decorr_tau_class_reg.yaml --model-prefix /tmp/rgerosa/ --gpus '' --batch-size 100 --log /tmp/rgerosa/weight_ak8_tau_classreg.log --num-workers 1
  python3 ../weaver-core/weaver/train.py  --data-train '/eos/cms/store/group/phys_exotica/monojet/rgerosa/ParticleNetUL/NtupleTrainingAK8Skimmed/tree_*root'  --network-config networks/particle_net_ak8_pf_sv_ext.py --data-config data/ak8_points_pf_sv_mass_decorr_taumuel.yaml --model-prefix /tmp/rgerosa/ --gpus '' --batch-size 100 --log /tmp/rgerosa/weight_ak8_taumuel.log --num-workers 1
  python3 ../weaver-core/weaver/train.py  --data-train '/eos/cms/store/group/phys_exotica/monojet/rgerosa/ParticleNetUL/NtupleTrainingAK8Skimmed/tree_*root'  --network-config networks/particle_net_ak8_pf_sv_class_reg.py --data-config data/ak8_points_pf_sv_mass_decorr_taumuel_class_reg.yaml --model-prefix /tmp/rgerosa/ --gpus '' --batch-size 100 --log /tmp/rgerosa/weight_ak8_taumuel_classreg.log --num-workers 1
```

