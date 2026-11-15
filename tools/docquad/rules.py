import pickle, sys, itertools
rows = pickle.load(open(sys.argv[1], 'rb'))
sets = ['standard', 'unseen', 'hard', 'real', 'amz_sheet', 'amz_book', 'pack_in', 'pack_out']
def score(rule):
    out = {}
    for r in rows:
        e = r['e_dq'] if rule(r) else r['e_v3']
        o = out.setdefault(r['set'], [0, 0, 0]); o[0] += e < (0.02 if r['set'] in ('standard', 'unseen', 'hard') else 0.03); o[1] += e < 0.05; o[2] += 1
    return out
def show(name, rule):
    o = score(rule); tot = sum(v[1] for v in o.values())
    print(f"{name:42s} 5%-total {tot:3d} | " + " ".join(f"{k}:{o[k][1]}/{o[k][2]}" for k in sets if k in o))
show("v3 only", lambda r: False)
show("dq only", lambda r: True)
for s_t, n_t, pk_t in itertools.product([0.04, 0.065, 0.09], [0.0, 0.4, 0.5], [0.3, 0.4]):
    show(f"switch s<{s_t} or (net<{n_t} & pk>={pk_t})", lambda r, s_t=s_t, n_t=n_t, pk_t=pk_t: r['s_v3'] < s_t or ((r['net_v3'] or 0) < n_t and r['pk'] >= pk_t))
