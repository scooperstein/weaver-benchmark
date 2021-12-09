import os
import numpy as np
import awkward0 as awkward
import uproot3_methods as uproot_methods
import argparse

import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s')


'''
Datasets introduction:
https://energyflow.network/docs/datasets/#quark-and-gluon-jets

Download:
- Pythia8 Quark and Gluon Jets for Energy Flow:
  - https://zenodo.org/record/3164691#.YbIG9_HML0o

- Herwig7.1 Quark and Gluon Jets:
  - https://zenodo.org/record/3066475#.YbIHUPHML0o
'''


def _transform(X, y, start=0, stop=-1):
    # source_array: (num_data, max_num_particles, 4)
    # (pt,y,phi,pid)

    from collections import OrderedDict
    v = OrderedDict()

    X = X[start:stop].astype(np.float32)
    y = y[start:stop]

    origPT = X[:, :, 0]
    indices = np.argsort(-origPT, axis=1)

    _pt = np.take_along_axis(X[:, :, 0], indices, axis=1)
    _eta = np.take_along_axis(X[:, :, 1], indices, axis=1)
    _phi = np.take_along_axis(X[:, :, 2], indices, axis=1)
    _pid = np.take_along_axis(X[:, :, 3], indices, axis=1)

    mask = _pt > 0
    n_particles = np.sum(mask, axis=1)

    pt = awkward.JaggedArray.fromcounts(n_particles, _pt[mask])
    eta = awkward.JaggedArray.fromcounts(n_particles, _eta[mask])
    phi = awkward.JaggedArray.fromcounts(n_particles, _phi[mask])
    mass = awkward.JaggedArray.zeros_like(pt)
    PID = awkward.JaggedArray.fromcounts(n_particles, _pid[mask])

    p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(pt, eta, phi, mass)
    px = p4.x
    py = p4.y
    pz = p4.z
    energy = p4.energy

    jet_p4 = p4.sum()

    # outputs
    v['label'] = y

    v['jet_px'] = jet_p4.x
    v['jet_py'] = jet_p4.y
    v['jet_pz'] = jet_p4.z
    v['jet_E'] = jet_p4.energy

    v['jet_pt'] = jet_p4.pt
    v['jet_eta'] = jet_p4.eta
    v['jet_phi'] = jet_p4.phi
    v['jet_mass'] = jet_p4.mass
    v['n_parts'] = n_particles

    v['part_px'] = px
    v['part_py'] = py
    v['part_pz'] = pz
    v['part_energy'] = energy

    v['part_pt_log'] = np.log(pt)
    v['part_ptrel'] = pt / v['jet_pt']
    v['part_logptrel'] = np.log(v['part_ptrel'])

    v['part_e_log'] = np.log(energy)
    v['part_erel'] = energy / jet_p4.energy
    v['part_logerel'] = np.log(v['part_erel'])

    v['part_raw_etarel'] = (p4.eta - v['jet_eta'])
    _jet_etasign = np.sign(v['jet_eta'])
    _jet_etasign[_jet_etasign == 0] = 1
    v['part_etarel'] = v['part_raw_etarel'] * _jet_etasign

    v['part_phirel'] = p4.delta_phi(jet_p4)
    v['part_deltaR'] = np.hypot(v['part_etarel'], v['part_phirel'])

    v['part_pid'] = PID
    v['part_isCHPlus'] = ((PID == 211) + (PID == 321) + (PID == 2212)).astype(np.float32)
    v['part_isCHMinus'] = ((PID == -211) + (PID == -321) + (PID == -2212)).astype(np.float32)
    v['part_isNeutralHadron'] = ((PID == 130) + (PID == 2112) + (PID == -2112)).astype(np.float32)
    v['part_isPhoton'] = (PID == 22).astype(np.float32)
    v['part_isEPlus'] = (PID == -11).astype(np.float32)
    v['part_isEMinus'] = (PID == 11).astype(np.float32)
    v['part_isMuPlus'] = (PID == -13).astype(np.float32)
    v['part_isMuMinus'] = (PID == 13).astype(np.float32)

    v['part_isChargedHadron'] = v['part_isCHPlus'] + v['part_isCHMinus']
    v['part_isElectron'] = v['part_isEPlus'] + v['part_isEMinus']
    v['part_isMuon'] = v['part_isMuPlus'] + v['part_isMuMinus']

    v['part_charge'] = (v['part_isCHPlus'] + v['part_isEPlus'] + v['part_isMuPlus']
                        ) - (v['part_isCHMinus'] + v['part_isEMinus'] + v['part_isMuMinus'])

    return v


def convert(sources, destdir, basename):
    if not os.path.exists(destdir):
        os.makedirs(destdir)

    for idx, sourcefile in enumerate(sources):
        npfile = np.load(sourcefile)
        output = os.path.join(destdir, '%s_%d.awkd' % (basename, idx))
        logging.info(sourcefile)
        logging.info(str(npfile['X'].shape))
        logging.info(output)
        if os.path.exists(output):
            os.remove(output)
        v = _transform(npfile['X'], npfile['y'])
        awkward.save(output, v, mode='x')


def natural_sort(l):
    import re
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Convert qg benchmark datasets to awkd')
    parser.add_argument('-i', '--inputdir', required=True, help='Directory of input numpy files.')
    parser.add_argument('-o', '--outputdir', required=True, help='Output directory.')
    parser.add_argument('--train-test-split', default=0.9, help='Training / testing split fraction.')
    args = parser.parse_args()

    import glob
    sources = natural_sort(glob.glob(os.path.join(args.inputdir, 'QG_jets*.npz')))
    n_train = int(args.train_test_split * len(sources))
    train_sources = sources[:n_train]
    test_sources = sources[n_train:]

    convert(train_sources, destdir=args.outputdir, basename='train_file')
    convert(test_sources, destdir=args.outputdir, basename='test_file')
