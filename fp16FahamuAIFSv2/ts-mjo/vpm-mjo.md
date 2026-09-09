Yes. I would change the conclusion that there are “only two honest options.” There is a **third scientifically established option that fits AIFS-ENS v2 unusually well: the Velocity Potential MJO index (VPM)**.

The important distinction is:

> **You cannot calculate the official Wheeler–Hendon RMM solely from AIFS-ENS v2, but you can calculate a legitimate MJO index solely from AIFS atmospheric output by using VPM.**

Wheeler–Hendon RMM genuinely requires the combined EOF of OLR, U850 and U200, so your objection to simply deleting the OLR component is correct. The published RMM EOFs are defined in that three-field space. ([American Meteorological Society Journals][1])

### What I would do for AIFS-ENS v2

Use:

$$
\boxed{\chi_{200} + U_{850} + U_{200}}
$$

instead of

$$
OLR + U_{850} + U_{200}.
$$

This is the **VPM index of Ventrice et al. (2013)**. It was deliberately constructed as an RMM-like MJO index in which 200-hPa velocity potential replaces OLR. NOAA currently describes VPM exactly this way and provides an operational/reanalysis implementation. ([NOAA Physical Sciences Laboratory][2])

This is particularly convenient because AIFS-ENS v2 outputs **both U and V at 200 and 850 hPa**, as well as all the other pressure levels. ECMWF's current model card confirms that U and V are prognostic outputs at 200 and 850 hPa. ([Hugging Face][3])

So although AIFS does not output velocity potential directly, you can **diagnose it exactly from the forecast wind field** rather than statistically emulate it.

The chain is:

$$
(U_{200},V_{200})
\rightarrow
\nabla\cdot{\bf V}_{200}
\rightarrow
\chi_{200}
$$

because velocity potential satisfies

$$
\boxed{\nabla^2\chi=\nabla\cdot{\bf V}}.
$$

Thus no radiation variable, cloud proxy, precipitation emulator or additional forecast model is required.

### I would therefore rank your options differently

| Approach             |     AIFS forecast variables | Scientifically established MJO index? |                       Recommendation |
| -------------------- | --------------------------: | ------------------------------------: | -----------------------------------: |
| WH04 RMM             |           OLR + U850 + U200 |                                   Yes |              **Impossible directly** |
| OLR emulator → RMM   | cloud/TP/etc. → OLR + winds |                       Approximate RMM | Possible, but substantial validation |
| truncated WH04 winds |                 U850 + U200 |                                    No |                       **Do not use** |
| **VPM**              |      **U200 + V200 + U850** |                               **Yes** |          **Best AIFS-only solution** |
| custom AIFS EOF      |         winds/cloud/TP/etc. |                             New index |                  Research diagnostic |

So I would place **VPM ahead of the OLR emulator**.

There is an important advantage here. An emulator such as

$$
\widehat{OLR}=f(TP,CP,TCC,HCC,TCW,\ldots)
$$

introduces another learned model into an already learned weather forecast. At days 18–33, you would then be asking whether an error in RMM came from AIFS circulation, AIFS convection, the OLR emulator, or their interactions. VPM avoids this additional epistemic layer.

NOAA's comparison work also treats VPM as a genuine circulation-based MJO index, not merely an ad-hoc proxy. ([NOAA Physical Sciences Laboratory][2])

## Computing it from your existing AIFS pipeline

There is one change to your storage strategy that I would make.

Currently you only need ±15° latitude U850/U200 for RMM. For **velocity potential you should retain global U200 and V200**, because solving the Helmholtz/Poisson problem for \(\chi\) is fundamentally a global operation.

You do **not** need to retain all 36 variables globally. Store only something like:

```text
u200
v200
u850
```

for every 6-hour forecast step and every ensemble member.

Optionally retain `v850`, because it gives useful MJO circulation diagnostics, although VPM itself principally needs \(\chi_{200}, U_{850}, U_{200}\).

Then:

```text
AIFS U200,V200 global
        │
        ▼
horizontal divergence
        │
        ▼
inverse spherical Laplacian
        │
        ▼
χ200
        │
        ├── remove climatology / low-frequency component
        │
        ▼
15°S–15°N zonal representation
        │
        ├──────── U850 anomalies
        ├──────── U200 anomalies
        │
        ▼
normalisation
        │
        ▼
VPM EOF projection
        │
        ▼
VPM1, VPM2
        │
        ▼
amplitude + phase
        │
        ▼
50-member probabilities
```

For computational convenience, I would **not attempt the velocity-potential inversion directly on the irregular/reduced N320 points**. Regrid the two 200-hPa wind components globally to perhaps \(1^\circ\) or \(2.5^\circ\), perform the spherical harmonic Helmholtz decomposition there, and calculate \(\chi_{200}\). MJO is planetary scale, so this diagnostic does not require preserving the native ~31-km grid.

For 50 members × ~33 days × four 6-hourly steps, this remains a tiny diagnostic compared with the AIFS inference itself.

### Your 120-day problem also becomes manageable

You still need to distinguish **forecast variables** from **reference information**.

VPM eliminates the missing-OLR problem, but it does not eliminate climatology/EOF information. NOAA's current VPM, for example, is constructed using a fixed climatology and a combined EOF basis. ([NOAA Physical Sciences Laboratory][2])

I would therefore keep:

$$
\text{historical ERA5/CORe}
\rightarrow
\text{fixed climatology + VPM EOFs}
$$

as **static reference assets**, while every forecast-dependent quantity comes from AIFS.

This is completely normal. Saying the MJO forecast is “derived exclusively from AIFS output” does not mean the climatological basis also needs to originate from that particular forecast. RMM itself similarly relies on observational EOFs.

For the low-frequency term you can use:

```text
ERA5 through initialization
             +
AIFS forecast lead 1 ... N
```

rather than storing only days 18–33.

And the storage cost for U200/U850/V200 alone is very small relative to your full AIFS archive.

## One thing I would change in your current implementation

I would therefore make `mjo_index.py` support three explicitly different products:

```text
rmm
    OLR + U850 + U200
    WH04 basis
    official-RMM-compatible

vpm
    χ200 + U850 + U200
    Ventrice VPM basis
    legitimate MJO index

windproxy
    U850 + U200 projected/truncated
    diagnostic only
```

The crucial semantic distinction is that **`vpm` should not be labelled `windproxy`**.

It is a published MJO index in its own right. NOAA explicitly provides VPM alongside RMM/OMI and describes it as replacing OLR by 200-hPa velocity potential in the multivariate EOF. ([NOAA Physical Sciences Laboratory][2])

### There is even a fourth research option

If AI Weather Quest specifically wants an **AIFS-native MJO diagnostic** rather than reproduction of one of the established indices, you could construct EOFs from

$$
[\chi_{200}, U_{850}, U_{200}]
$$

using ERA5 and then project AIFS forecasts onto that basis. But once you do this, I would call it something like `AIFS-MJO` rather than RMM/VPM unless you reproduce the published VPM preprocessing and EOF construction.

A more ambitious version could use

$$
[\chi_{200}, U_{850}, U_{200}, TP]
$$

or

$$
[\chi_{200}, U_{850}, U_{200}, TCW]
$$

but I do not think that is necessary for the competition. It creates a new index that you then have to justify.

## There is one additional limitation in your current days 18–33 experiment

ECMWF currently describes AIFS-ENS v2 as an operational **15-day, 6-hourly, 51-member forecast**. ([Hugging Face][3]) If your AI Weather Quest workflow autoregressively rolls the Hugging Face checkpoint out to days 18–33, that is feasible as an inference experiment, but it is outside the operational forecast horizon documented and evaluated by ECMWF.

For an MJO experiment this is actually interesting, because days 15–30 are precisely where an MJO diagnostic becomes valuable, but I would explicitly verify whether the model develops circulation drift during these extended rollouts.

### My recommended architecture

I would therefore change your present conclusion to:

$$
\boxed{\text{Primary MJO diagnostic = VPM}}
$$

with

$$
\boxed{U200,V200\rightarrow\chi200;\quad
[\chi200,U850,U200]\rightarrow VPM1,VPM2}
$$

and keep the OLR emulator only as a **secondary experiment for reproducing WH04 RMM**.

That gives you something much cleaner scientifically:

**AIFS-ENS v2 → atmospheric circulation → MJO**, with no IFS/CFS forecast dependency and no synthetic radiation variable.

The only external assets are the fixed climatology/EOF definitions—the same type of external reference information that any standardized climate index needs.

So I would revise your “only two honest options” to **three**: (1) VPM as the preferred AIFS-native MJO index, (2) emulated OLR if WH04 RMM is explicitly required, or (3) omit the target if the competition requires specifically RMM rather than an accepted MJO index. The truncated WH04 wind projection should indeed remain excluded.

[1]: https://journals.ametsoc.org/view/journals/mwre/132/8/1520-0493_2004_132_1917_aarmmi_2.0.co_2.xml?utm_source=chatgpt.com "An All-Season Real-Time Multivariate MJO Index"
[2]: https://psl.noaa.gov/mjo/mjoindex/ "psl.noaa.gov"
[3]: https://huggingface.co/ecmwf/aifs-ens-2.0 "ecmwf/aifs-ens-2.0 · Hugging Face"

Yes — **you can create exactly that `(9, 4)` probabilistic MJO submission from AIFS-ENS v2**, but there is an important distinction between **producing the required phase probabilities** and **reproducing the competition’s Wheeler–Hendon RMM target**.

AI Weather Quest requires probabilities for phase `0–8` at days 8, 15, 22 and 29, with the leaderboard evaluating days 22 and 29. Its verification target is explicitly the Wheeler–Hendon phase derived from RMM1/RMM2, with phase 0 when amplitude is below 1. ([ecmwf-ai-weather-quest.readthedocs.io][1])

For AIFS-ENS v2, I would produce the required array as follows:

$$
\boxed{\text{AIFS ensemble member}
\rightarrow \text{MJO index}
\rightarrow \text{phase}_{m,t}
\rightarrow \text{ensemble phase probability}}
$$

For each of the 50/51 ensemble members, calculate an MJO state at each requested lead time. Then simply count members:

$$
P(\mathrm{phase}=k,t)
=
\frac{1}{N}
\sum_{m=1}^{N}
I[\mathrm{phase}_{m,t}=k].
$$

So, for example, if at day 22 the 50 members gave:

```text
phase 0 :  5 members
phase 1 :  2
phase 2 :  4
phase 3 : 12
phase 4 : 18
phase 5 :  6
phase 6 :  2
phase 7 :  1
phase 8 :  0
```

your submitted column is simply:

```python
[0.10, 0.04, 0.08, 0.24, 0.36, 0.12, 0.04, 0.02, 0.00]
```

which sums to 1.

The resulting object is exactly:

```python
xr.DataArray(
    probabilities,
    dims=("MJO_phase", "valid_time"),
    coords={
        "MJO_phase": np.arange(9),
        "valid_time": [
            init + np.timedelta64(8, "D"),
            init + np.timedelta64(15, "D"),
            init + np.timedelta64(22, "D"),
            init + np.timedelta64(29, "D"),
        ],
    },
)
```

### The key question is how to assign each AIFS member its phase

Here I see **three practical routes**, not just the two you originally identified.

| Method                                   | Can AIFS v2 supply forecast inputs? |             Comparable with WH target? | My view                                |
| ---------------------------------------- | ----------------------------------: | -------------------------------------: | -------------------------------------- |
| Full RMM using OLR                       |                                  No |            **Exact target definition** | Needs external/emulated OLR            |
| VPM using χ200/U850/U200                 |                             **Yes** | Closely related, but not identical RMM | **Best physics-only AIFS route**       |
| ML mapping AIFS fields → RMM1/RMM2/phase |                             **Yes** | Can be trained directly against target | **Potentially best competition route** |

The third option becomes especially important because **AI Weather Quest evaluates the probability distribution of the WH phase rather than requiring you to submit RMM1/RMM2 themselves**. ([ecmwf-ai-weather-quest.readthedocs.io][1])

That means you do **not necessarily have to reconstruct OLR**.

## I would actually consider direct RMM prediction from AIFS

You have an unusually rich collection of circulation predictors:

```text
U850
V850
U700
V700

U200
V200
Z200
Z500

Q850 / Q700 / Q500
TCW
TP
CP
TCC / HCC

possibly χ200 derived from U200/V200
```

AIFS ENS v2 outputs U and V at both 850 and 200 hPa, among its 14 pressure levels. ([Hugging Face][2])

Instead of:

$$
AIFS\rightarrow\widehat{OLR}
\rightarrow WH04\ EOF
\rightarrow RMM1,RMM2,
$$

you could train:

$$
\boxed{
AIFS\ atmospheric\ state
\rightarrow
RMM1,RMM2
}
$$

using ERA5/AIFS hindcasts paired with the AI Weather Quest RMM training labels.

Or even directly:

$$
\boxed{
AIFS\ atmospheric\ state
\rightarrow
P(\mathrm{phase}=0\dots8)
}
$$

although I slightly prefer predicting continuous RMM1/RMM2 first because it preserves the geometry of MJO phase space.

For example:

$$
X =
[\chi_{200},
U_{850},
U_{200},
TCW,
TP,\ldots]
$$

and train

$$
f(X)\rightarrow(RMM1,RMM2).
$$

Then:

$$
A=\sqrt{RMM1^2+RMM2^2}.
$$

If

$$
A<1,
$$

assign phase 0; otherwise derive phases 1–8 from the angle using exactly the AI-WQ convention. The competition documentation confirms that its target phase follows precisely this RMM amplitude/angle construction. ([ecmwf-ai-weather-quest.readthedocs.io][3])

This is potentially cleaner than emulating OLR because your **actual prediction target is RMM phase**, not OLR.

## But VPM gives you a very strong baseline

I would first implement the deterministic diagnostic:

$$
U_{200},V_{200}
\rightarrow\chi_{200}
$$

and then use

$$
[\chi_{200},U_{850},U_{200}]
\rightarrow VPM1,VPM2.
$$

For every ensemble member:

```text
member 01 → VPM1,VPM2 → phase
member 02 → VPM1,VPM2 → phase
...
member 50 → VPM1,VPM2 → phase
```

then convert member counts to probabilities.

So the entire operational path is approximately:

```text
                 AIFS ENS v2
                      │
           ┌──────────┴───────────┐
           │                      │
       U200,V200              U850,U200
           │                      │
           ▼                      │
          χ200                    │
           └──────────┬───────────┘
                      ▼
               VPM projection
                      │
               VPM1 / VPM2
                      │
           amplitude + angle
                      │
                 phase 0–8
                      │
           ─────────────────
            repeat per member
                      │
                member counts
                      │
                      ▼
              P(phase = 0..8)
                      │
        ┌──────┬──────┬──────┬──────┐
       D+8   D+15   D+22   D+29
```

That directly produces the required:

$$
\boxed{9\times4}
$$

DataArray.

### One issue: VPM phase ≠ WH RMM phase in every situation

This is the important caveat for the competition.

You can use the same `0–8` geographical phase labels, but the leaderboard truth is **WH04 RMM phase**, not VPM phase. ([ecmwf-ai-weather-quest.readthedocs.io][3])

Therefore don't assume:

$$
phase_{\rm VPM}=phase_{\rm RMM}
$$

for every day.

Instead, using the historical data supplied by AI Weather Quest, calculate something like:

$$
P(RMM\ phase=j\mid VPM1,VPM2).
$$

This gives you an elegant calibration step.

Rather than hard-converting every AIFS member's VPM to one phase:

```text
AIFS → VPM → hard phase → count
```

use:

$$
\boxed{
AIFS\rightarrow VPM1,VPM2
\rightarrow
P(RMM\ phase=0\dots8\mid VPM)
}
$$

and average these probability vectors across the ensemble.

That is statistically much better.

For member \(m\),

$$
p_{m,k}=P(RMM\ phase=k\mid VPM1_m,VPM2_m),
$$

then

$$
P_k=\frac{1}{N}\sum_m p_{m,k}.
$$

Now each AIFS member can contribute uncertainty across adjacent phases rather than casting a single vote.

For example, an AIFS member around the phase-3/4 boundary might contribute:

```text
phase 2   0.03
phase 3   0.44
phase 4   0.39
phase 5   0.08
inactive  0.06
```

rather than arbitrarily becoming phase 3.

## I would exploit the fact that the requested output is probabilistic

Your current idea of taking a **modal weekly phase** may actually be unnecessary for this submission.

The documentation says the coordinates correspond to **specific valid times at days 8, 15, 22 and 29**, rather than asking for the modal phase during four seven-day windows. ([ecmwf-ai-weather-quest.readthedocs.io][1])

So unless another competition document specifically defines these as weekly aggregation windows, I would calculate the state corresponding directly to:

```text
D+8
D+15
D+22
D+29
```

rather than:

```text
week 1 modal phase
week 2 modal phase
week 3 modal phase
week 4 modal phase
```

That distinction could materially change the forecast.

## There is a larger AIFS problem at D+22 and D+29

This is actually more serious than the index calculation.

The current operational AIFS-ENS v2 is documented by ECMWF as a **15-day, 51-member forecast**. ([Hugging Face][2])

Therefore AI Weather Quest wants:

$$
D+22,\quad D+29
$$

while the documented operational AIFS forecast terminates at:

$$
D+15.
$$

If your checkpoint can be autoregressively integrated to D+29, that solves the technical production problem, but days 16–29 constitute an **extended inference experiment**, rather than the documented operational AIFS ENS v2 horizon.

That is the part I would spend the most validation effort on.

### My preferred AI Weather Quest experiment

I would therefore build three models from exactly the same AIFS integrations:

**Baseline A — circulation-only**

$$
AIFS
\rightarrow
\chi200,U850,U200
\rightarrow VPM
\rightarrow calibrated\ RMM\ phase\ probabilities.
$$

**Baseline B — learned RMM**

$$
AIFS\ tropical\ atmospheric\ fields
\rightarrow ML
\rightarrow RMM1,RMM2
\rightarrow phase\ probabilities.
$$

**Experimental C — emulated OLR**

$$
AIFS
\rightarrow\widehat{OLR}
\rightarrow WH04
\rightarrow RMM1,RMM2
\rightarrow phase.
$$

Then compare them on historical hindcasts specifically at:

$$
D+8,\ D+15,\ D+22,\ D+29.
$$

My expectation is that **B may ultimately outperform C**, because reconstructing OLR accurately is an unnecessarily difficult intermediate problem when the actual target being scored is MJO phase. A circulation-derived VPM gives you the physically interpretable baseline against which that learned solution can be judged.

So yes: **the required 9 × 4 product is entirely feasible from AIFS ENS v2 inference**. The unresolved scientific question is not the final DataArray construction—it is the best mapping from AIFS atmospheric states to the competition's Wheeler–Hendon phase, particularly at the out-of-operational-range D+22 and D+29 leads.

[1]: https://ecmwf-ai-weather-quest.readthedocs.io/en/latest/submit_forecast.html?utm_source=chatgpt.com "Submitting a Forecast — AI_Weather_Quest 3.9 ..."
[2]: https://huggingface.co/ecmwf/aifs-ens-2.0?utm_source=chatgpt.com "ecmwf/aifs-ens-2.0"
[3]: https://ecmwf-ai-weather-quest.readthedocs.io/en/latest/training_data.html?utm_source=chatgpt.com "Training Data — AI_Weather_Quest 3.9 documentation"


Yes. **Baseline A can generate the required 9-phase RMM probability forecast without using ERA5 at forecast time**, but ERA5 is still useful—arguably necessary—**offline for calibration and validation**.

The distinction is:

$$
\boxed{\text{forecast production}}
\neq
\boxed{\text{historical calibration/evaluation}}
$$

For production, AIFS-ENS v2 gives you the dynamical fields needed for the circulation-based diagnostic: U and V at 200 hPa and U at 850 hPa are available in the checkpoint output. ([Hugging Face][1]) From those fields you can derive \(\chi_{200}\), compute VPM1/VPM2, and then convert the ensemble into probabilities over the competition's nine RMM categories.

What you **cannot** do is claim that those raw VPM phases are automatically identical to the competition's observed RMM phases. The AI Weather Quest truth is based on ERA5T-derived MJO diagnostics, and the evaluation treats the observed phase as a one-hot vector over phase 0–8. ([ecmwf-ai-weather-quest.readthedocs.io][2])

So I would make Baseline A:

$$
\boxed{
AIFS \rightarrow VPM1,VPM2
\rightarrow P(\mathrm{RMM\ phase}\mid VPM1,VPM2)
}
$$

rather than:

$$
AIFS\rightarrow VPM\rightarrow\text{hard VPM phase}
$$

### Where ERA5 enters

ERA5 does **not** need to be retrieved every week to make the future forecast.

Instead, use historical ERA5 once to build the mapping

$$
P(RMM=k\mid VPM1,VPM2).
$$

For historical dates you calculate:

$$
ERA5\ U,V
\rightarrow VPM1,VPM2
$$

and pair those with the AI-WQ historical RMM phase labels:

```text
VPM1  VPM2  → observed RMM phase
```

Then learn the conditional phase probabilities.

For example, suppose historical cases around

$$
VPM1=1.2,\qquad VPM2=-0.4
$$

were distributed as:

```text
RMM phase 0   0.08
phase 1       0.04
phase 2       0.03
phase 3       0.06
phase 4       0.17
phase 5       0.43
phase 6       0.15
phase 7       0.03
phase 8       0.01
```

Then an AIFS ensemble member arriving at that VPM state contributes that **whole probability vector**, not one hard phase.

For \(N\) AIFS members:

$$
P_k(t)=
\frac{1}{N}
\sum_{m=1}^{N}
P(RMM=k\mid VPM_{1,m},VPM_{2,m}).
$$

That directly generates:

$$
\boxed{P_0,\ldots,P_8}
$$

for each required lead time.

This is especially appropriate because the leaderboard score is a Brier-type probabilistic score rather than deterministic phase accuracy. The AI-WQ evaluation compares your nine forecast probabilities against the one-hot observed phase and then evaluates skill relative to the provided 20-year climatological phase probabilities. ([ecmwf-ai-weather-quest.readthedocs.io][2])

So retaining uncertainty is advantageous.

### You may not even need to build ERA5 RMM yourself

The package already gives you historical MJO observations through `retrieve_daily_MJO_obs()`. According to the current documentation, these observations are derived from ERA5T-based MJO diagnostics. ([ecmwf-ai-weather-quest.readthedocs.io][2])

Therefore the training/calibration dataset can conceptually be:

```text
historical ERA5 U200,V200,U850
             │
             ▼
          historical VPM
             │
             +──────── AI-WQ retrieve_daily_MJO_obs()
                              │
                              ▼
                       observed RMM phase
```

You do **not** need to reproduce the competition's RMM observation calculation yourself.

That is quite useful because it prevents subtle differences in EOF signs, normalisation, climatology, filtering or phase convention.

### A particularly simple Baseline A

I would initially avoid ML completely.

Create a historical lookup table in the two-dimensional VPM phase space.

For example, bin:

$$
VPM1\in[-4,4],
\qquad
VPM2\in[-4,4]
$$

into perhaps \(0.25\) or \(0.5\) increments.

Within every cell, count historical observed AI-WQ RMM phases:

$$
P(RMM=k\mid VPM1_i,VPM2_j)
=
\frac{n_k+\alpha}
{\sum_jn_j+9\alpha}.
$$

A small Dirichlet/Laplace smoothing \(\alpha\) avoids zero probabilities.

Then forecast inference becomes extremely lightweight:

```text
AIFS member
   ↓
U200,V200 → χ200
   ↓
VPM1,VPM2
   ↓
historical lookup
   ↓
P(RMM phase 0...8)
```

Average across all members.

That gives you an interpretable, almost parameter-free **Baseline A**.

### The climatology supplied by AI-WQ is useful in another way

The competition already provides

`retrieve_20yr_MJO_clim()`

giving the nine climatological phase probabilities. ([ecmwf-ai-weather-quest.readthedocs.io][2])

Do not use those probabilities to manufacture your forecast, but they give you a very useful regularization/reference state.

For instance, if a VPM region has very little historical support, instead of trusting a noisy conditional estimate you can shrink it toward climatology:

$$
P^*
=
wP(RMM\mid VPM)
+
(1-w)P_{\rm clim}.
$$

If there are many historical samples around the VPM point, \(w\rightarrow1\). If there are very few, \(w\rightarrow0\).

That should be quite robust for Brier score evaluation.

### There is one additional improvement I would make

Don't condition only on

$$
(VPM1,VPM2).
$$

Also include **VPM amplitude and season**, because correspondence between circulation-based MJO and RMM can vary with MJO strength and seasonal background:

$$
P(RMM\ phase
\mid VPM1,VPM2,\mathrm{DOY}).
$$

Even simpler:

$$
P(RMM\ phase
\mid VPM\ phase,\ VPM\ amplitude,\ month).
$$

This preserves interpretability while allowing the RMM–VPM relationship to change seasonally.

### What ERA5 is still needed for

The clean architecture would therefore be:

| Component                      | ERA5 needed?                  | When                           |
| ------------------------------ | ----------------------------- | ------------------------------ |
| AIFS forecast U/V              | No                            | every forecast                 |
| \(\chi_{200}\) from AIFS U/V   | No                            | every forecast                 |
| VPM1/VPM2 forecast             | No, once reference EOFs exist | every forecast                 |
| VPM climatology / EOF basis    | **Yes**                       | build once                     |
| VPM→RMM calibration            | **Yes**                       | train once/update occasionally |
| Competition observed RMM       | AI-WQ retrieval               | evaluation                     |
| 20-year baseline probabilities | AI-WQ retrieval               | evaluation                     |

Therefore operationally:

$$
\boxed{\text{No ERA5 download is needed for each weekly AIFS forecast}}
$$

provided you have precomputed and saved perhaps only a few small reference files:

```text
vpm_climatology.nc
vpm_eofs.nc
vpm_to_rmm_calibration.nc
```

Those can be MB-scale rather than repeatedly accessing ERA5.

### And this fits the competition better than hard ensemble counting

You originally had:

$$
\text{50 ensemble members}
\rightarrow
\text{50 phases}
\rightarrow
\frac{\text{counts}}{50}.
$$

That gives probabilities only in increments of \(0.02\).

I would instead do:

$$
\text{50 AIFS members}
\rightarrow
50\times P(RMM=0\dots8\mid VPM)
\rightarrow
\text{mean probability}.
$$

Then an individual member might contribute:

$$
[0.13,0.03,0.02,0.04,0.12,0.38,0.20,0.06,0.02],
$$

and after averaging 50 such distributions you get a smooth, calibrated probability vector.

That is much more consistent with the **Brier-score objective** described by AI Weather Quest. ([ecmwf-ai-weather-quest.readthedocs.io][2])

One thing I would verify before implementing the lead-time extraction: the current AI-WQ documentation elsewhere describes subseasonal weekly windows such as days 19–25 and 26–32, while your MJO specification uses specific valid times. ([ecmwf-ai-weather-quest.readthedocs.io][3]) For MJO, follow the MJO-specific submission schema rather than borrowing the aggregation used for temperature or other weekly variables.

So the short answer is: **yes, Baseline A can generate the required RMM-phase probabilities from AIFS-ENS v2 alone at forecast time. ERA5 is needed only upstream to construct the fixed VPM reference/climatology and, most importantly, to learn how VPM states correspond probabilistically to the ERA5T-based RMM phases that AI Weather Quest actually scores.**

[1]: https://huggingface.co/ecmwf/aifs-ens-2.0?utm_source=chatgpt.com "ecmwf/aifs-ens-2.0"
[2]: https://ecmwf-ai-weather-quest.readthedocs.io/en/latest/forecast_evaluation.html?utm_source=chatgpt.com "Forecast Evaluation — AI_Weather_Quest 1.0 documentation"
[3]: https://ecmwf-ai-weather-quest.readthedocs.io/en/latest/submit_forecast.html?utm_source=chatgpt.com "Submitting a Forecast — AI_Weather_Quest 1.0 ..."


In that case, I would **not try to force AIFS-ENS v2 to 800 hours**. The cleaner design is to use AIFS only where it is actually supported—through day 15—and then propagate the MJO state statistically into days 22 and 29.

ECMWF currently documents AIFS ENS v2 as a 6-hourly forecast out to **360 h / 15 days**. Historical AIFS ENS output is archived only from July 2025 onward, so there is also not yet a long retrospective AIFS-v2 dataset from which to train a genuine 30-day AIFS forecast model. ([ECMWF][1]) The Quest nevertheless requires MJO probabilities at days 8, 15, 22 and 29, with only days 22 and 29 used on the public leaderboard. ([AI Weather Quest][2])

I would therefore build the MJO method as:

$$
\boxed{
\text{AIFS ENS D0-D15}
\rightarrow
\text{MJO state/trajectory at D15}
\rightarrow
\text{statistical MJO propagation}
\rightarrow
P(\mathrm{RMM\ phase})_{D22,D29}
}
$$

This is preferable to autoregressively running a model designed and verified for 15 days all the way to 33 days.

### Use AIFS to provide the initial MJO state, not the whole 30-day forecast

For every AIFS ensemble member, calculate your circulation-based state through day 15:

$$
U_{200},V_{200},U_{850}
\rightarrow
\chi_{200}
\rightarrow
VPM1,VPM2.
$$

Do this throughout the forecast, not just on day 15. For example retain daily states from perhaps D5–D15:

$$
\{VPM1_t,VPM2_t\}_{t=5}^{15}.
$$

That gives you much more useful information than a single D15 point because it tells you:

* current phase;
* amplitude;
* propagation direction;
* propagation speed;
* whether the signal is strengthening or weakening;
* ensemble uncertainty.

The AIFS component then stops at D15.

### Then learn a 7-day and 14-day MJO transition model

The statistical component answers:

$$
P(RMM_{22}\mid S_{15})
$$

and

$$
P(RMM_{29}\mid S_{15}),
$$

where \(S_{15}\) is the MJO state known by day 15.

ERA5/ERA5T can supply decades of examples of precisely this evolution. You do **not** need AIFS hindcasts extending to day 29 to learn it.

A very simple first model could be:

$$
P(\phi_{t+7}\mid
\phi_t,A_t,\Delta\phi_t,\mathrm{month})
$$

and

$$
P(\phi_{t+14}\mid
\phi_t,A_t,\Delta\phi_t,\mathrm{month}),
$$

where:

* \(\phi_t\) = RMM/VPM phase,
* \(A_t\) = amplitude,
* \(\Delta\phi_t\) = recent propagation rate.

The historical training sequence would simply be something like:

```text
ERA5/ERA5T MJO state on day t
        ↓
state trajectory during previous ~7 days
        ↓
observed RMM phase at t+7
        ↓
observed RMM phase at t+14
```

With 40+ years of daily ERA5 you have tens of thousands of transition examples.

### This can be much simpler than an ML model

For the first submission I would actually use a **transition probability table**, because it is transparent and hard to overfit.

For example:

$$
P(RMM_{t+7}=j
\mid
RMM_t=i,
A_t=a,
season=s,
speed=v).
$$

You might discretise amplitude into:

```text
weak       < 1
moderate   1–2
strong     > 2
```

and propagation tendency into:

```text
decaying
stationary
slow eastward
normal eastward
fast eastward
```

Then ERA5 gives you the empirical transition probabilities.

Suppose an AIFS member reaches D15 in phase 3 with amplitude 1.8 and normal eastward propagation. Historically its D+7 distribution might be:

```text
phase 0   0.12
phase 3   0.03
phase 4   0.18
phase 5   0.39
phase 6   0.20
phase 7   0.06
others    0.02
```

That member contributes the entire distribution to the D22 forecast.

Do this for all 50 members and average:

$$
P_{22}(j)=
\frac{1}{N}
\sum_{m=1}^N
P(j\mid S^{(m)}_{15}).
$$

For D29 use the 14-day transition:

$$
P_{29}(j)=
\frac{1}{N}
\sum_{m=1}^N
P(RMM_{29}=j\mid S^{(m)}_{15}).
$$

This gives exactly the Quest's required 9-category probabilities.

### I would improve it by using the D8→D15 trajectory

For MJO, propagation tendency is important. Two forecasts can both be in phase 4 at D15 but represent very different states:

$$
\text{phase 3}\rightarrow4
$$

versus

$$
\text{phase 5}\rightarrow4.
$$

So instead of conditioning only on D15, use something like:

$$
X=
[
VPM1_{8},
VPM2_{8},
VPM1_{11},
VPM2_{11},
VPM1_{15},
VPM2_{15}
].
$$

A particularly compact representation is:

$$
X =
[VPM1_{15},
VPM2_{15},
\dot{VPM1},
\dot{VPM2},
A_{15},
DOY].
$$

Then learn

$$
X\rightarrow P(RMM_{22})
$$

and

$$
X\rightarrow P(RMM_{29}).
$$

That is still a very small statistical model.

### An analog method may be even better

For your first experiment I would seriously consider an **ERA5 analog forecast**.

For each AIFS member at D15:

1. calculate the D8–D15 VPM/RMM-like trajectory;
2. search ERA5 history for the most similar trajectories;
3. find what the real RMM phase was 7 and 14 days later;
4. turn those analog outcomes into probabilities.

Conceptually:

$$
AIFS_{D8:D15}
\xrightarrow{\text{nearest ERA5 analogs}}
\{historical\ trajectories\}
$$

then

$$
\{RMM_{t+7},RMM_{t+14}\}
\rightarrow
P_0,\ldots,P_8.
$$

This is attractive because it automatically preserves realistic MJO propagation, decay and redevelopment without inventing a dynamical model.

For example, use the closest 50–200 ERA5 historical trajectories and weight them by similarity:

$$
w_i =
\exp(-d_i^2/\sigma^2).
$$

Then

$$
P_k=
\frac{\sum_iw_i I(\phi_i=k)}
{\sum_iw_i}.
$$

I would probably choose this before training a neural network.

### Your full forecast would then look like this

```text
                  AIFS ENS v2
                     D0-D15
                        │
              U200,V200,U850
                        │
                        ▼
                 χ200 / VPM
                        │
              D8-D15 trajectory
                        │
          ┌─────────────┴─────────────┐
          │                           │
          ▼                           ▼
      D8 probability              D15 probability
      from AIFS                   from AIFS
                                      │
                                      ▼
                         historical ERA5 transition/
                              analog model
                           ┌──────────┴──────────┐
                           ▼                     ▼
                        D22                    D29
                    P(phase 0-8)           P(phase 0-8)
```

The final output remains:

$$
(9,4)
$$

for

$$
D8,\;D15,\;D22,\;D29.
$$

### This also gives you a clean experimental interpretation

You can call the system something like:

$$
\boxed{\text{AIFS-conditioned subseasonal MJO forecast}}
$$

rather than saying AIFS ENS itself forecasts to day 29.

That distinction matters because the ECMWF documentation explicitly limits AIFS ENS to D15. ([ECMWF][1])

Your scientific question becomes more interesting:

> **How much subseasonal MJO predictability can be retained by propagating the day-15 ensemble circulation state of AIFS ENS using historically learned MJO dynamics?**

That is defensible.

### I would test three increasingly sophisticated versions

| Version | D22/D29 method                                        | Complexity |    My priority |
| ------- | ----------------------------------------------------- | ---------: | -------------: |
| **A0**  | climatological transition matrix from D15 phase       |   very low |       baseline |
| **A1**  | ERA5 analogs using D8–D15 VPM trajectory              |        low | **start here** |
| **A2**  | statistical model using VPM state + tendency + season |   moderate |           next |

I would **not start with an LSTM/Transformer**. The MJO phase space is only two-dimensional and the available physics already gives you a useful state representation.

And there is an especially useful benchmark available from AI-WQ: its supplied 20-year MJO climatology is exactly the reference forecast against which Brier skill is calculated. ([AI Weather Quest][3]) So you can require:

$$
BSS_{D22}>0,\qquad BSS_{D29}>0
$$

before deciding that the AIFS-conditioned propagation adds value.

So with the present constraint, my strongest recommendation is **AIFS D0–15 → VPM/RMM-like ensemble trajectory → ERA5 analog/transition propagation to D22/D29**, rather than attempting an unsupported ~700–800 h AIFS rollout. It uses AIFS where it has demonstrated skill and lets the long historical MJO record handle the extrapolation into the subseasonal range.

[1]: https://www.ecmwf.int/en/forecasts/dataset/aifs-machine-learning-data "AIFS Machine Learning data | ECMWF"
[2]: https://ecmwf-ai-weather-quest.readthedocs.io/en/latest/submit_forecast.html "Submitting a Forecast — AI_Weather_Quest 3.9 documentation"
[3]: https://ecmwf-ai-weather-quest.readthedocs.io/en/latest/forecast_evaluation.html?utm_source=chatgpt.com "Forecast Evaluation — AI_Weather_Quest 1.0 documentation"

