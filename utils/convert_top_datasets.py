import os
import pandas as pd
import numpy as np
import awkward0 as awkward
import uproot3_methods as uproot_methods
import argparse


import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s')


def _transform(dataframe, start=0, stop=-1):
    from collections import OrderedDict
    v = OrderedDict()

    df = dataframe.iloc[start:stop]

    def _col_list(prefix, max_particles=200):
        return ['%s_%d' % (prefix, i) for i in range(max_particles)]

    _px = df[_col_list('PX')].values
    _py = df[_col_list('PY')].values
    _pz = df[_col_list('PZ')].values
    _e = df[_col_list('E')].values

    mask = _e > 0
    n_particles = np.sum(mask, axis=1)

    px = awkward.JaggedArray.fromcounts(n_particles, _px[mask])
    py = awkward.JaggedArray.fromcounts(n_particles, _py[mask])
    pz = awkward.JaggedArray.fromcounts(n_particles, _pz[mask])
    energy = awkward.JaggedArray.fromcounts(n_particles, _e[mask])

    p4 = uproot_methods.TLorentzVectorArray.from_cartesian(px, py, pz, energy)
    pt = p4.pt

    jet_p4 = p4.sum()

    # outputs
    _label = df['is_signal_new'].values
    v['label'] = _label
    v['train_val_test'] = df['ttv'].values

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

    return v


def convert(source, destdir, basename, step=None, limit=None):
    df = pd.read_hdf(source, key='table')
    logging.info('Total events: %s' % str(df.shape[0]))
    if limit is not None:
        df = df.iloc[0:limit]
        logging.info('Restricting to the first %s events:' % str(df.shape[0]))
    if step is None:
        step = df.shape[0]
    idx = -1
    while True:
        idx += 1
        start = idx * step
        if start >= df.shape[0]:
            break
        if not os.path.exists(destdir):
            os.makedirs(destdir)
        output = os.path.join(destdir, '%s_%d.awkd' % (basename, idx))
        logging.info(output)
        if os.path.exists(output):
            os.remove(output)
        v = _transform(df, start=start, stop=start + step)
        awkward.save(output, v, mode='x')


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Convert top benchmark h5 datasets to awkd')
    parser.add_argument('-i', '--inputdir', required=True, help='Directory of input h5 files.')
    parser.add_argument('-o', '--outputdir', required=True, help='Output directory.')
    args = parser.parse_args()

    # conver training file
    convert(os.path.join(args.inputdir, 'train.h5'), destdir=args.outputdir, basename='train_file')

    # conver validation file
    convert(os.path.join(args.inputdir, 'val.h5'), destdir=args.outputdir, basename='val_file')

    # conver testing file
    convert(os.path.join(args.inputdir, 'test.h5'), destdir=args.outputdir, basename='test_file')
