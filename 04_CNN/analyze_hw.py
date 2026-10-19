# coding: utf-8
"""
analyze_hw.py

Sweep the PE array shape (ROWS x COLS) and report, per layer and in total:

    cycles            estimates under this model's assumptions
    DRAM traffic      input words, weight words, output words
    PE utilisation    useful MACs / MAC slots offered
    FPGA resources    LUT, DSP, BRAM36 on XC7Z020

ROWS = reduction axis  (input channels, or C*R*S when im2col'd)
COLS = output channels computed in parallel

Historical design-space model: NET below uses Conv3 pad1 / Affine-1152,
whereas convnet_hw.py uses Conv3 pad0 / Affine-512. The simplified compute
model is not the current four-state RTL pixel schedule. Keep these estimates
separate from RTL simulation and on-board timing measurements.

AI-assisted comparison script; candidate results and the 8x32 array selection
were evaluated by the project author.

Calibrated against the Task 1 synthesis report:
    1024 PE  ->  86,368 LUT   (84.3 LUT per PE, DSP mostly not inferred)
    non-PE logic ~5,500 LUT

Usage
    python analyze_hw.py                  # sweep the built-in candidates
    python analyze_hw.py 16 32            # just report ROWS=16, COLS=32
"""
import sys

# ----------------------------------------------------------------- device
DEV = dict(name="XC7Z020", lut=53200, ff=106400, dsp=220, bram36=140)

# ------------------------------------------------------- calibrated costs
LUT_PER_PE   = 84.3     # 86368 / 1024, measured
FF_PER_PE    = 8.0      # 8192 / 1024  (the weight register)
LUT_FIXED    = 5500     # controller, dma, buffers, bn_regfile, requant
DSP_BATCHNORM_PER_COL = 1   # batch_norm used 32 DSP for 32 columns

WORD = 4                # bytes per DRAM word
PUSH_OVERHEAD = 2       # ST_SWAP + margin; the push itself is ROWS+1 cycles
ACC_RMW = 2             # accumulator read-modify-write cycles per pixel
ACC_W    = 26           # accumulator width in bits


def ceil_div(a, b):
    return -(-a // b)


class Layer:
    """One layer of the network.

    mode 'im2col' : reduction axis is C*R*S laid flat on PE rows (first layer)
    mode 'chan'   : reduction axis is the channel group; (r,s) becomes ig tiles
    mode 'fc'     : affine, a single output "pixel"
    """

    def __init__(self, name, cin, cout, k=3, pad=1, stride=1,
                 hin=1, win=1, pool=1, mode='chan'):
        self.name, self.cin, self.cout = name, cin, cout
        self.k, self.pad, self.stride = k, pad, stride
        self.hin, self.win, self.pool = hin, win, pool
        self.mode = mode

        if mode == 'fc':
            self.hout = self.wout = 1
        else:
            self.hout = (hin + 2*pad - k)//stride + 1
            self.wout = (win + 2*pad - k)//stride + 1
        self.P = self.hout * self.wout

    # --------------------------------------------------- shape on an array
    def tiles(self, R, C):
        og = ceil_div(self.cout, C)
        if self.mode == 'im2col':
            red = self.cin * self.k * self.k
            ig = ceil_div(red, R)
        elif self.mode == 'fc':
            red = self.cin
            ig = ceil_div(red, R)
        else:
            red = self.cin * self.k * self.k
            ig = ceil_div(self.cin, R) * self.k * self.k
        return ig, og, red

    def macs(self):
        if self.mode == 'fc':
            return self.cin * self.cout
        return self.P * self.cout * self.cin * self.k * self.k

    def weights(self):
        if self.mode == 'fc':
            return self.cin * self.cout
        return self.cout * self.cin * self.k * self.k

    # ------------------------------------------------------------- traffic
    def words_per_row(self, R):
        """A buffer row is R bytes wide -> R/4 DRAM words."""
        return max(1, R // WORD)

    def in_words(self, R, C):
        _, og, _ = self.tiles(R, C)
        w = self.words_per_row(R)
        if self.mode == 'im2col':
            base = self.P * w
        elif self.mode == 'fc':
            base = ceil_div(self.cin, R) * w
        else:
            base = self.hin * self.win * ceil_div(self.cin, R) * w
        return base * og          # og is the outer loop -> input re-read

    def weight_words(self, R, C):
        ig, og, _ = self.tiles(R, C)
        return ig * og * (R * C // WORD)

    def out_words(self, R, C):
        _, og, _ = self.tiles(R, C)
        po = (self.hout//self.pool) * (self.wout//self.pool)
        return po * og * self.words_per_row(C)

    # -------------------------------------------------------------- cycles
    def compute_cycles(self, R, C):
        ig, og, _ = self.tiles(R, C)
        tile_words = R * C // WORD
        push = R + 1 + PUSH_OVERHEAD
        work = self.P * (ACC_RMW if ig > 1 else 1) + push
        return ig * og * max(tile_words, work)

    def drain_cycles(self, R, C):
        _, og, _ = self.tiles(R, C)
        return og * self.P

    def total_cycles(self, R, C):
        return (self.in_words(R, C) + self.compute_cycles(R, C)
                + self.drain_cycles(R, C) + self.out_words(R, C))

    def util(self, R, C):
        ig, og, _ = self.tiles(R, C)
        return self.macs() / (ig * og * self.P * R * C)

    def acc_depth(self):
        return self.P

    def ibuf_rows(self, R):
        if self.mode == 'im2col':
            return self.P
        if self.mode == 'fc':
            return ceil_div(self.cin, R)
        return self.hin * self.win * ceil_div(self.cin, R)


# --------------------------------------------------------------- resources
def resources(net, R, C):
    pe = R * C
    lut = pe * LUT_PER_PE + LUT_FIXED
    ff = pe * FF_PER_PE + 10000
    dsp = C * DSP_BATCHNORM_PER_COL      # batch_norm only; PE mults go to LUT

    acc_depth = max(l.acc_depth() for l in net)
    ibuf_rows = max(l.ibuf_rows(R) for l in net)
    obuf_rows = max(l.P for l in net)

    def bram36(width_bits, depth):
        """How many BRAM36 a width x depth memory needs."""
        per = max(1, 36864 // max(1, width_bits))
        return ceil_div(depth, per)

    # accumulator : one independent memory per column lane
    acc_bram = C * bram36(ACC_W, acc_depth)
    # ibuf / wbuf / obuf : R (or C) bytes wide, ping-pong = x2
    ibuf_bram = 2 * bram36(R * 8, ibuf_rows)
    wbuf_bram = 2 * bram36(R * 8, R)
    obuf_bram = 2 * bram36(C * 8, obuf_rows)

    return dict(pe=pe, lut=int(lut), ff=int(ff), dsp=dsp,
                bram=acc_bram + ibuf_bram + wbuf_bram + obuf_bram,
                acc_bram=acc_bram, ibuf_bram=ibuf_bram,
                wbuf_bram=wbuf_bram, obuf_bram=obuf_bram,
                acc_depth=acc_depth, ibuf_rows=ibuf_rows)


# --------------------------------------------------------------- the model
NET = [
    Layer('Conv1_1',   1,  32, k=3, pad=1, hin=28, win=28, mode='im2col'),
    Layer('Conv1_2',  32,  32, k=3, pad=1, hin=28, win=28, pool=2),
    Layer('Conv2_1',  32,  64, k=3, pad=1, hin=14, win=14, pool=1),
    Layer('Conv2_2',  64,  64, k=3, pad=1, hin=14, win=14, pool=2),
    Layer('Conv3',    64, 128, k=3, pad=1, hin=7,  win=7,  pool=2),
    Layer('Affine',  1152,  10, mode='fc'),
]


def report(R, C, net=NET, verbose=True):
    res = resources(net, R, C)
    tot = sum(l.total_cycles(R, C) for l in net)
    macs = sum(l.macs() for l in net)
    slots = sum(l.tiles(R, C)[0] * l.tiles(R, C)[1] * l.P * R * C for l in net)

    if verbose:
        print("=" * 92)
        print(f"  PE array {R} x {C} = {R*C} PE      total {tot:,} cycles"
              f"   @100MHz {tot/1e5:.2f} ms")
        print("=" * 92)
        print(f"  {'layer':<10}{'ig':>5}{'og':>4}{'P':>6}{'cycles':>11}"
              f"{'share':>8}{'util':>8}{'in w':>9}{'wgt w':>9}{'out w':>8}")
        print("-" * 92)
        for l in net:
            ig, og, _ = l.tiles(R, C)
            c = l.total_cycles(R, C)
            print(f"  {l.name:<10}{ig:>5}{og:>4}{l.P:>6}{c:>11,}"
                  f"{100*c/tot:>7.1f}%{100*l.util(R, C):>7.1f}%"
                  f"{l.in_words(R, C):>9,}{l.weight_words(R, C):>9,}"
                  f"{l.out_words(R, C):>8,}")
        print("-" * 92)
        print(f"  overall PE utilisation {100*macs/slots:5.1f} %")
        print()
        print(f"  {'resource':<12}{'used':>10}{'avail':>10}{'%':>8}   note")
        rows = [("LUT", res['lut'], DEV['lut']),
                ("FF", res['ff'], DEV['ff']),
                ("DSP", res['dsp'], DEV['dsp']),
                ("BRAM36", res['bram'], DEV['bram36'])]
        for nm, used, av in rows:
            flag = "OVER" if used > av else ""
            print(f"  {nm:<12}{used:>10,}{av:>10,}{100*used/av:>7.0f}%   {flag}")
        print(f"  BRAM split : acc {res['acc_bram']} (depth {res['acc_depth']})"
              f" + ibuf {res['ibuf_bram']} (rows {res['ibuf_rows']})"
              f" + wbuf {res['wbuf_bram']} + obuf {res['obuf_bram']}")
        print("=" * 92)
        print()

    return tot, res


def sweep(candidates, net=NET):
    print("=" * 92)
    print("  array sweep")
    print("=" * 92)
    print(f"  {'R x C':<12}{'PE':>6}{'cycles':>12}{'vs32x32':>9}"
          f"{'LUT':>9}{'LUT%':>7}{'BRAM':>7}{'BRAM%':>7}   fits")
    print("-" * 92)
    base = None
    out = []
    for R, C in candidates:
        tot, res = report(R, C, net, verbose=False)
        if R == 32 and C == 32:
            base = tot
    if base is None:
        base = report(32, 32, net, verbose=False)[0]
    for R, C in candidates:
        tot, res = report(R, C, net, verbose=False)
        fits = (res['lut'] <= DEV['lut'] and res['bram'] <= DEV['bram36']
                and res['dsp'] <= DEV['dsp'])
        out.append((R, C, tot, res, fits))
        print(f"  {f'{R} x {C}':<12}{R*C:>6}{tot:>12,}{tot/base:>8.2f}x"
              f"{res['lut']:>9,}{100*res['lut']/DEV['lut']:>6.0f}%"
              f"{res['bram']:>7}{100*res['bram']/DEV['bram36']:>6.0f}%"
              f"   {'YES' if fits else 'no'}")
    print("=" * 92)
    print()

    ok = [o for o in out if o[4]]
    if ok:
        best = min(ok, key=lambda o: o[2])
        print(f"  fastest configuration that fits : {best[0]} x {best[1]}"
              f"   {best[2]:,} cycles"
              f"   LUT {100*best[3]['lut']/DEV['lut']:.0f}%"
              f"   BRAM {100*best[3]['bram']/DEV['bram36']:.0f}%")
    else:
        print("  nothing fits; shrink the array further")
    print()
    return out


CANDIDATES = [
    (32, 32),           # current, does not fit
    (32, 16), (16, 32), (8, 64),        # 512 PE
    (16, 16), (32, 8), (8, 32),         # 256 PE
    (16, 8),  (8, 16),                  # 128 PE
]


if __name__ == '__main__':
    if len(sys.argv) == 3:
        report(int(sys.argv[1]), int(sys.argv[2]))
    else:
        out = sweep(CANDIDATES)
        # detail for the three 512-PE shapes
        for R, C in [(32, 16), (16, 32), (8, 64)]:
            report(R, C)
