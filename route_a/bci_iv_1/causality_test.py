"""Causality test by perturbation.

The D12 shot estimates three things from the entire evaluation recording: the
band-pass (`filtfilt` over the whole signal), the feature alignment (a mean over
300 s blocks) and the Euclidean recentering (the mean covariance of the recording).
A live system cannot do any of the three. D13 replaces them with causal versions.
This script checks that the replacement is actually causal rather than assuming it.

The test: a 1e4 uV impulse is added to the evaluation signal at sample t and the
output at every sample before t must be bit-identical to the unperturbed output
(difference exactly 0). Four conditions must hold together:

  1. the fully causal system passes, on the same code path as the shot, at several t;
  2. the offline system fails (otherwise the test cannot tell "causal" from "the
     impulse never arrived");
  3. for each of the three pieces, the system that is causal except for that piece
     fails, so each piece is caught on its own;
  4. the output after t does change, so the impulse did enter the system.

The impulse goes to a subset of channels: added to all 59 and followed by the common
average reference, it would cancel out and the test would pass without testing
anything. The perturbed recording is never scored: `pipeline.correr_eval` refuses to
compute an MSE on a perturbed signal, so no evaluation label is read.

A small model with every piece of the full system is tested at three instants
(1/50 of the cost; the test is about exactness, not accuracy), and the full D12
ensemble at one instant.

Output: outputs/causality_test.json

Usage:
    python causality_test.py              # subject g
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '8')

import argparse
import time

import numpy as np

import components as K
import pipeline as P

# 1e4 uV is about 1000 times the EEG amplitude: anything leaking from the future
# would show.
AMPLITUD = 1e4
# A subset of channels, so that the common average reference cannot cancel it.
CANALES = list(range(0, 30))

# Horizon of the causal alignment and of the causal recentering, in seconds.
ALINEAMIENTO_CAUSAL_S = 300.0     # same horizon as the 300 s blocks of D12
RECENTRADO_CAUSAL_S = 300.0

# The three non-causal pieces, each with its offline (`hoy`) and causal version.
# Every system in this module is composed from this table.
PIEZAS = {
    'filtrado': dict(
        que_es='band-pass: filtfilt over the whole recording',
        hoy=dict(filtro='continuo_zp'),
        causal=dict(filtro='ventana_zp'),
        por_que_no_es_causal=('filtfilt runs forward and backward over the whole '
                              'signal, so every output sample depends on later ones')),
    'alineamiento': dict(
        que_es='mean of the target features, in 300 s blocks of the recording',
        hoy=dict(alineamiento_bloques_s=300.0, alineamiento_causal_s=0.0),
        causal=dict(alineamiento_bloques_s=0.0,
                    alineamiento_causal_s=ALINEAMIENTO_CAUSAL_S),
        por_que_no_es_causal=('within a block the mean uses every row of the block, '
                              'including later ones')),
    'recentrado': dict(
        que_es='Euclidean whitening with the mean covariance of the whole recording',
        hoy=dict(recentrar_causal_s=0.0),
        causal=dict(recentrar_causal_s=RECENTRADO_CAUSAL_S),
        por_que_no_es_causal='the reference covariance averages the whole recording'),
}


def cfgs_del_campeon(s: str) -> list:
    """The two members of the D12 ensemble."""
    from run_shot import cfgs_d12
    return cfgs_d12()


def componer(cfgs: list, causales: set) -> list:
    """Apply the causal version of the listed pieces to every member and the offline
    version of the others."""
    out = []
    for c in cfgs:
        d = dict(c)
        for nombre, pz in PIEZAS.items():
            d.update(pz['causal'] if nombre in causales else pz['hoy'])
        out.append(P.cfg_con(**d))
    return out


def sistemas() -> list[dict]:
    """The systems under test, with the expected outcome declared up front."""
    todas = set(PIEZAS)
    fila = [dict(id='hoy', causales=set(), espera='FALLA',
                 por_que='the three pieces look ahead'),
            dict(id='causal', causales=set(todas), espera='PASA',
                 por_que='the three pieces are causal')]
    for p in PIEZAS:
        fila.append(dict(id=f'causal_menos_{p}', causales=todas - {p}, espera='FALLA',
                         por_que=f'`{p}` is still non-causal and the test must catch it'))
    return fila


def instantes(T_ev: int, n: int = 3) -> list[int]:
    """Early, middle and late, with output on both sides."""
    return [int(round(f * T_ev)) for f in np.linspace(0.2, 0.8, n)]


def _salida(sujeto: str, cfgs: list, perturbacion=None) -> np.ndarray:
    """Output of the complete system, on the same code path as the shot."""
    if len(cfgs) == 1:
        r = P.correr_eval(sujeto, cfgs[0], perturbacion=perturbacion)
    else:
        r = P.correr_eval_ensamblado(sujeto, cfgs, perturbacion=perturbacion)
    assert r.get('mse') is None or perturbacion is None
    return np.asarray(r['salida'], np.float64)


def probar_un_sistema(sujeto: str, cfgs: list, ts: list[int], etiqueta: str) -> dict:
    """Inject the impulse at each instant and compare the output before and after it."""
    base = _salida(sujeto, cfgs)
    filas = []
    for t in ts:
        pert = dict(muestra=int(t), amplitud=AMPLITUD, canales=CANALES,
                    antes_del_car=True)
        o = _salida(sujeto, cfgs, perturbacion=pert)
        n = min(len(base), len(o))
        antes = slice(0, min(int(t), n))
        despues = slice(min(int(t), n), n)
        d_antes = float(np.max(np.abs(base[antes] - o[antes]))) if antes.stop else 0.0
        d_despues = (float(np.max(np.abs(base[despues] - o[despues])))
                     if despues.stop > despues.start else 0.0)
        filas.append(dict(
            sistema=etiqueta, sujeto=sujeto, muestra=int(t),
            segundo=float(t) / float(P.FS),
            n_muestras_antes=int(antes.stop),
            max_dif_antes=d_antes, max_dif_despues=d_despues,
            n_muestras_que_cambian_antes=int(np.sum(base[antes] != o[antes])),
            pasa=bool(d_antes == 0.0),
            el_impulso_llego=bool(d_despues > 0.0)))
    return dict(sistema=etiqueta, sujeto=sujeto, filas=filas,
                pasa=bool(all(f['pasa'] for f in filas)),
                el_impulso_llego_siempre=bool(all(f['el_impulso_llego']
                                                  for f in filas)),
                max_dif_antes=float(max(f['max_dif_antes'] for f in filas)))


def cfg_barata(**kw) -> dict:
    """A small system with every piece of the full one: same code path, 1/50 of the
    cost. The test is about exactness (0 or not 0), so a small model answers it."""
    c = dict(banco='mb', w_s=2.0, m_csp=2, nk=2, regresor='multinomial', C=0.03,
             prior='objetivo', post='lineal', post_taps=40, post_alpha=1.0,
             car=True, recentrar=True, alineamiento='centrado')
    c.update(kw)
    return P.cfg_con(**c)


def fase1(sujeto: str = 'g', n_instantes: int = 3, con_campeon: bool = True) -> dict:
    t0 = time.time()
    ses = P.Sesion(sujeto, cfg_barata(), con_eval=True)
    T_ev = int(ses.T_ev)
    del ses
    ts = instantes(T_ev, n_instantes)

    res = []
    for sis in sistemas():
        # the one-piece-off systems only need one instant: their role is to show the
        # test catches that piece, not to sweep time
        tt = ts if sis['id'] in ('hoy', 'causal') else ts[len(ts) // 2:len(ts) // 2 + 1]
        cfgs = componer([cfg_barata()], sis['causales'])
        r = probar_un_sistema(sujeto, cfgs, tt, sis['id'])
        r.update(espera=sis['espera'], por_que=sis['por_que'],
                 causales=sorted(sis['causales']), modelo='barato',
                 cumple_lo_esperado=bool((sis['espera'] == 'PASA') == r['pasa']))
        res.append(r)
        print(f"   [small] {sis['id']:<26s} expected {sis['espera']:<5s} -> "
              f"{'PASA' if r['pasa'] else 'FALLA'}  max_diff_before="
              f"{r['max_dif_antes']:.3e}  ({time.time() - t0:.0f}s)", flush=True)

    campeon = []
    if con_campeon:
        # and on the full system, a two-member ensemble, so the test also goes
        # through that code path
        base = cfgs_del_campeon(sujeto)
        for sis in [s for s in sistemas() if s['id'] in ('hoy', 'causal')]:
            cfgs = componer(base, sis['causales'])
            r = probar_un_sistema(sujeto, cfgs, ts[len(ts) // 2:len(ts) // 2 + 1],
                                  sis['id'])
            r.update(espera=sis['espera'], causales=sorted(sis['causales']),
                     modelo='campeon', n_miembros=len(cfgs),
                     cumple_lo_esperado=bool((sis['espera'] == 'PASA') == r['pasa']))
            campeon.append(r)
            print(f"   [D12]   {sis['id']:<26s} expected {sis['espera']:<5s} -> "
                  f"{'PASA' if r['pasa'] else 'FALLA'}  max_diff_before="
                  f"{r['max_dif_antes']:.3e}  ({time.time() - t0:.0f}s)", flush=True)

    todos = res + campeon
    return dict(
        sujeto=sujeto, T_ev=T_ev, instantes=[int(t) for t in ts],
        amplitud=AMPLITUD, n_canales_perturbados=len(CANALES),
        resultados=res, resultados_del_campeon=campeon,
        n_sistemas=len(todos),
        n_como_se_esperaba=int(sum(1 for r in todos if r['cumple_lo_esperado'])),
        todo_como_se_esperaba=bool(all(r['cumple_lo_esperado'] for r in todos)),
        el_sistema_causal_pasa=bool(all(r['pasa'] for r in todos
                                        if r['sistema'] == 'causal')),
        el_sistema_de_hoy_falla=bool(all(not r['pasa'] for r in todos
                                         if r['sistema'] == 'hoy')),
        cada_pieza_queda_cazada=bool(all(not r['pasa'] for r in todos
                                         if r['sistema'].startswith('causal_menos_'))),
        el_impulso_siempre_llega=bool(all(r['el_impulso_llego_siempre']
                                          for r in todos)),
        segundos=round(time.time() - t0, 1))


# ------------------------------------------------------------ checks of the test itself
def _t_el_impulso_no_se_cancela_con_el_car():
    """The impulse survives the common average reference, and perturbing all 59
    channels (which the reference would cancel) is refused."""
    x = P._senal_perturbada('g', True, dict(muestra=1000, amplitud=AMPLITUD,
                                            canales=CANALES, antes_del_car=True))
    x0 = P._senal_perturbada('g', True, None)
    d = float(np.max(np.abs(x[:, 1000] - x0[:, 1000])))
    assert d > AMPLITUD * 0.1, d
    try:
        P._senal_perturbada('g', True, dict(muestra=1000, amplitud=AMPLITUD,
                                            canales=None, antes_del_car=True))
    except AssertionError:
        return dict(nombre='impulse_survives_car_and_all_channels_refused',
                    salto_tras_el_car=d, ok=True)
    raise AssertionError('perturbing all 59 channels was accepted')


def _t_sin_perturbacion_la_senal_es_la_de_siempre():
    """The perturbation hook changes nothing when unused, or when the amplitude is 0."""
    a = P._senal_perturbada('g', True, None)
    b = K.senal_evaluacion('g', True)
    d = float(np.max(np.abs(a - b)))
    assert d == 0.0, d
    c = P._senal_perturbada('g', True, dict(muestra=1000, amplitud=0.0,
                                            canales=CANALES, antes_del_car=True))
    d2 = float(np.max(np.abs(a - c)))
    assert d2 == 0.0, d2
    return dict(nombre='unperturbed_signal_is_bit_identical',
                max_dif=d, max_dif_amplitud_cero=d2, ok=True)


def _t_la_composicion_cubre_las_tres_piezas():
    """`componer` touches exactly the keys of the table, and the causal system differs
    from the offline one in all three pieces."""
    base = [cfg_barata()]
    hoy = componer(base, set())[0]
    cau = componer(base, set(PIEZAS))[0]
    difs = {k: (hoy[k], cau[k]) for k in hoy if hoy[k] != cau[k]}
    esperadas = set()
    for pz in PIEZAS.values():
        esperadas |= set(pz['hoy']) | set(pz['causal'])
    cambian = {k for k, (a, b) in difs.items() if a != b}
    assert cambian <= esperadas, cambian - esperadas
    for nombre, pz in PIEZAS.items():
        sin = componer(base, set(PIEZAS) - {nombre})[0]
        assert any(sin[k] != cau[k] for k in pz['causal']), nombre
    return dict(nombre='composition_touches_exactly_the_three_pieces',
                claves_que_cambian=sorted(cambian),
                claves_declaradas=sorted(esperadas), ok=True)


def _t_una_perturbacion_en_el_pasado_SI_cambia_la_salida():
    """An impulse in the past must change the later output of any system, causal
    included; otherwise the test would only show that the impulse does not enter."""
    cfgs = componer([cfg_barata(post='ninguno')], set(PIEZAS))
    ses = P.Sesion('g', cfgs[0], con_eval=True)
    T_ev = int(ses.T_ev)
    del ses
    t = int(0.3 * T_ev)
    base = _salida('g', cfgs)
    o = _salida('g', cfgs, perturbacion=dict(muestra=t, amplitud=AMPLITUD,
                                             canales=CANALES, antes_del_car=True))
    n = min(len(base), len(o))
    d_desp = float(np.max(np.abs(base[t:n] - o[t:n])))
    d_antes = float(np.max(np.abs(base[:t] - o[:t])))
    assert d_desp > 0.0, 'the impulse did not arrive'
    assert d_antes == 0.0, d_antes
    return dict(nombre='impulse_changes_the_future_not_the_past',
                max_dif_despues=d_desp, max_dif_antes=d_antes,
                muestra=int(t), ok=True)


AUTOTESTS = [_t_sin_perturbacion_la_senal_es_la_de_siempre,
             _t_el_impulso_no_se_cancela_con_el_car,
             _t_la_composicion_cubre_las_tres_piezas,
             _t_una_perturbacion_en_el_pasado_SI_cambia_la_salida]


def autotests(verbose: bool = True) -> list[dict]:
    r = []
    for f in AUTOTESTS:
        v = f()
        r.append(v)
        if verbose:
            print(f'  ok  {v["nombre"]}', flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sujeto', default='g')
    ap.add_argument('--sin-campeon', action='store_true',
                    help='skip the full D12 ensemble (small model only)')
    ap.add_argument('--sin-tests', action='store_true',
                    help='skip the checks of the test itself')
    a = ap.parse_args()

    t0 = time.time()
    tests = []
    if not a.sin_tests:
        print('checks of the test itself')
        tests = autotests()
    print('\nperturbation test', flush=True)
    f1 = fase1(a.sujeto, con_campeon=not a.sin_campeon)

    out = dict(piezas=PIEZAS, sistemas_declarados=[
                   dict(id=s['id'], causales=sorted(s['causales']), espera=s['espera'],
                        por_que=s['por_que']) for s in sistemas()],
               amplitud_del_impulso=AMPLITUD, canales_perturbados=CANALES,
               alineamiento_causal_s=ALINEAMIENTO_CAUSAL_S,
               recentrado_causal_s=RECENTRADO_CAUSAL_S,
               fase1_perturbacion=f1, autotests=tests,
               segundos=round(time.time() - t0, 1))
    K.guardar_json(K.OUT / 'causality_test.json', out)

    print(f'\n{f1["n_como_se_esperaba"]}/{f1["n_sistemas"]} systems behave as expected')
    print(f'  causal system passes      : {f1["el_sistema_causal_pasa"]}')
    print(f'  offline system fails      : {f1["el_sistema_de_hoy_falla"]}')
    print(f'  each piece caught alone   : {f1["cada_pieza_queda_cazada"]}')
    print(f'  impulse always arrives    : {f1["el_impulso_siempre_llega"]}')
    print(f'\n-> {K.OUT / "causality_test.json"}   ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
