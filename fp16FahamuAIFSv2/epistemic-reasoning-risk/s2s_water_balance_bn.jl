#=
S2S catchment water-balance Bayesian network — AIFS-ENS v2 / Icechunk / RxInfer.jl

Adapted from bn-ibf `flood_ibf/flood_bn_ibf_v1.jl` (ICPAC IBF team) to the grouped PROCESS
ontology of `qd-1.md.txt` and to what the AIFS-ENS v2 Icechunk store actually contains.

SCOPE (2026-08-06, DIRECTION_EXPLANATION_FIRST.md): this network is a DOWNSTREAM CONSUMER, not
the primary product. The pipeline's primary output is now the circulation explanation record
(`s2s_bn_evidence_prep.py --explain-out`) — ensemble counts over circulation states with the
effective sample size attached. This BN consumes the same loci and adds the risk framing on
top, but its lower half (`P(W|P,A)`, `P(RO|W)`) is elicited and unverifiable until the
observation side exists, and its antecedent node is a model proxy. Read its output as a
structured hypothesis about catchment water balance, NOT as a calibrated risk product. The
mechanism below is unchanged; only its status is.

What changed from the flood original, and why (full argument in S2S_BN_ONTOLOGY.md):

  * TARGET.  The original targets flood risk at a 0-7 day lead. This store holds hours
             432-792 (days 18-33) ONLY. At that lead the honest endpoint is qd-1's
             "catchment water-balance pressure" — an ordered surplus/deficit tendency that
             serves BOTH flood and drought reasoning — not a flood probability.
  * DAG.     Flat 5-parent expert CPT  ->  a physical CHAIN of process groups:
                 C -> M -> R -> P ,  (P, A) -> W ,  W -> RO
             so correlated meteorological fields never appear as sibling parents (the
             double-counting failure the original's `avoid this structure` warns about).
  * EVIDENCE. The 50 members give ONE joint draw over the whole chain each, so the chain
             CPTs are COUNTED from the joint member sample (see s2s_bn_evidence_prep.py)
             rather than elicited. Default `--evidence-mode chain` therefore attaches
             ensemble evidence only where it is independent (C, A, RO) and lets the counted
             CPTs carry it downstream — attaching it at every node would re-count the same
             50 members up to six times. `--evidence-mode all` reproduces the naive port and
             warns.
  * SEAM.    P(W | P, A) and P(RO | W) are ELICITED, not counted — no observation in this
             system can yet supply them (Route E). They are ordinal-kernel CPTs whose width
             is a stated, tunable uncertainty, wider for the later lead window.

Dependencies:  Pkg.add(["RxInfer", "CSV", "DataFrames", "JSON3"])

Usage:
    julia --project s2s_water_balance_bn.jl --input-csv evidence_20260730.csv \
        --output-csv wbp_20260730.csv [--cpt-json process_cpt_20260730.json] \
        [--evidence-mode chain|all] [--cost-loss-ratio 0.2] [--rxinfer]
    julia --project s2s_water_balance_bn.jl --member-csv evidence_members_20260730.csv \
        --storyline-csv storylines_20260730.csv
    julia --project s2s_water_balance_bn.jl --test
=#

using LinearAlgebra
using Printf
using CSV
using DataFrames
using JSON3
using RxInfer

# ============================================================================
# STATE VOCABULARY  (must match STATES in s2s_bn_evidence_prep.py + the registry)
# ============================================================================

const C_STATES  = ["unfavourable", "convergent"]                                     # 2
# k=2, NOT k=4: k_regimes_test.py found k>=3 clustering UNSTABLE on this ensemble spread
# (subsample ARI 0.56 at k=4 vs 0.82 at k=2; PC1 holds 72-88% of the variance). Four
# regimes were manufacturing structure from ~1 dominant direction.
const M_STATES  = ["deficient", "normal", "enhanced", "extreme_persistent"]          # 4
const R_STATES  = ["suppressed", "weakly_supportive", "supportive",
                   "strongly_supportive"]                                            # 4
const P_STATES  = ["below_normal", "normal", "heavy_episodic", "heavy_persistent"]   # 4
const A_STATES  = ["very_dry", "dry", "normal", "wet", "saturated"]                  # 5
const RO_STATES = ["nil", "low", "moderate", "high"]                                 # 4
const W_STATES  = ["strong_deficit", "mild_deficit", "balanced",
                   "elevated_surplus", "extreme_surplus"]                            # 5

const CRMA_STATES = ["Monitor", "Evaluate", "Assess", "Actionable_Risk"]
const TRAFFIC_LIGHT = Dict("Monitor" => "Green", "Evaluate" => "Yellow",
                           "Assess" => "Orange", "Actionable_Risk" => "Red")

# Ordinal scores. These ARE the elicited content of the lower network — every number here
# is a judgement, versioned in git, and none of it comes from the store (Route E seam).
const M_SCORE  = [-1.2, -0.2, 0.7, 1.4]      # moisture supply
const R_SCORE  = [-1.0, -0.3, 0.5, 1.1]      # rainfall-generation environment
const P_SCORE  = [-1.0, 0.0, 0.8, 1.3]       # precipitation forcing (persistent > episodic)
const A_SCORE  = [-1.0, -0.5, 0.0, 0.6, 1.0] # antecedent wetness
const W_CENTRE = [-2.0, -1.0, 0.0, 1.0, 2.0] # water-balance pressure state centres

"""
Ordinal kernel: map a continuous score onto an ordered state vector by Gaussian weight
around each state centre. `sigma` IS the stated uncertainty of the elicitation — wide at
extended range by design, and widened further for the later lead window.
"""
function ordinal_probs(score::Float64, centres::Vector{Float64}, sigma::Float64)
    w = exp.(-0.5 .* ((centres .- score) ./ sigma) .^ 2)
    s = sum(w)
    return s > 0 ? w ./ s : fill(1.0 / length(centres), length(centres))
end

# ============================================================================
# CPTs
# ============================================================================

"""
P(M | C) — elicited fallback. Overridden by counted tables when --cpt-json is supplied.
A more convergent circulation raises the moisture-supply state.
"""
function build_M_given_C(; sigma::Float64=1.0)
    T = zeros(4, 2)                                   # (M=4, C=2)
    c_score = [-0.8, 0.8]                             # unfavourable / convergent
    m_centre = [-1.2, -0.2, 0.7, 1.4]
    for c in 1:2
        T[:, c] = ordinal_probs(c_score[c], m_centre, sigma)
    end
    return T
end

"""
P(R | M) — elicited fallback. Moisture supply supports, but does not determine, the
rainfall-generation environment (ascent and stability are partly independent).
"""
function build_R_given_M(; sigma::Float64=1.2)
    T = zeros(4, 4)                                   # (R=4, M=4)
    r_centre = [-1.0, -0.3, 0.5, 1.1]
    for m in 1:4
        T[:, m] = ordinal_probs(0.75 * M_SCORE[m], r_centre, sigma)
    end
    return T
end

"""
P(P | M, R) — canonical (noisy-MAX-like) rather than a full 16-row table: one strength
parameter per parent, from which the table is generated (Route C of the plan).

The top mass is split between `heavy_episodic` and `heavy_persistent` by WHICH parent
delivered it: persistent moisture supply (M = extreme_persistent) tilts persistent; a
supportive environment on ordinary moisture tilts episodic.
"""
function build_P_given_MR(; sigma::Float64=1.1)
    T = zeros(4, 4, 4)                                # (P=4, M=4, R=4)
    sev_centre = [-1.0, 0.0, 1.05]                    # below_normal, normal, heavy
    for m in 1:4, r in 1:4
        score = 0.9 * M_SCORE[m] + 0.7 * R_SCORE[r]
        sev = ordinal_probs(score, sev_centre, sigma)             # 3-vector
        # persistence share of the heavy mass
        w_pers = clamp(0.35 + 0.30 * (m - 1) / 3 + 0.10 * (r - 1) / 3, 0.0, 1.0)
        T[1, m, r] = sev[1]
        T[2, m, r] = sev[2]
        T[3, m, r] = sev[3] * (1.0 - w_pers)          # episodic
        T[4, m, r] = sev[3] * w_pers                  # persistent
        T[:, m, r] ./= sum(T[:, m, r])
    end
    return T
end

"""
P(W | P, A) — THE ELICITED SEAM. Not counted, not learnable from this store: no
observation in the system yet closes it (Route E). Rain forcing and antecedent wetness
combine roughly additively, with two named non-linearities:

  * heavy rain on very dry ground is DAMPED (infiltration, no connected saturation),
  * persistent rain on wet/saturated ground is AMPLIFIED (the catchment is already primed).

`sigma` widens the whole table; pass a larger value for the later lead window.
"""
function build_W_given_PA(; sigma::Float64=1.0)
    T = zeros(5, 4, 5)                                # (W=5, P=4, A=5)
    for p in 1:4, a in 1:5
        score = 1.1 * P_SCORE[p] + 0.9 * A_SCORE[a]
        if p >= 3 && a <= 2                           # heavy rain, dry ground
            score -= 0.45
        end
        if p == 4 && a >= 4                           # persistent rain, primed catchment
            score += 0.45
        end
        T[:, p, a] = ordinal_probs(score, W_CENTRE, sigma)
    end
    return T
end

"""
P(RO | W) — the model's own runoff as an EVIDENCE CHILD of the latent water-balance state
(generative direction: evidence hangs off H, it is never a parent of it). Grade B: the
AIFS land scheme has no routing and no real catchment, so this likelihood is deliberately
flat-ish — it can corroborate a surplus, it cannot establish one.
"""
function build_RO_given_W()
    T = [
        0.55  0.40  0.25  0.12  0.05;   # nil
        0.30  0.35  0.35  0.28  0.18;   # low
        0.10  0.18  0.28  0.35  0.35;   # moderate
        0.05  0.07  0.12  0.25  0.42;   # high
    ]
    return T ./ sum(T, dims=1)
end

"""
Optional override of the CHAIN CPTs (M|C, R|M) with tables counted from the joint member
sample. Expects the JSON written by `s2s_bn_evidence_prep.py --cpt-json`:
    {"blocks": {"<unit>|<window>": {"ess": .., "CM_post": [[..]], "MR_post": [[..]]}}}
Counted tables are stored parent-major (`[parent][child]`); RxInfer wants (child, parent),
so they are transposed here.
"""
function load_counted_cpts(path::String)
    isfile(path) || (@warn "cpt-json not found; using elicited chain CPTs" path; return Dict())
    raw = JSON3.read(read(path, String))
    out = Dict{String,Dict{Symbol,Any}}()
    for (blk, v) in pairs(raw["blocks"])
        d = Dict{Symbol,Any}()
        for (src, dst) in ((:CM_post, :M_given_C), (:MR_post, :R_given_M))
            block = get(v, src, nothing)                             # JSON3 keys are Symbols
            if block !== nothing
                rows = [Float64.(r) for r in block]                  # parent-major rows
                d[dst] = reduce(hcat, rows)                          # -> (child, parent)
            end
        end
        ess = get(v, :ess, nothing)
        d[:ess] = ess === nothing ? NaN : Float64(ess)
        out[String(blk)] = d
    end
    @info "loaded counted chain CPTs" blocks=length(out)
    return out
end

# ============================================================================
# EXACT INFERENCE — tensor contraction over the chain (fast path, bulk runs)
# ============================================================================

"""
Exact posterior over W by summing the chain out. `*_ev` are probability vectors (one-hot =
hard evidence, `nothing` = no evidence at that node, which is the correct setting for the
interior nodes in `chain` mode).

Returns (w_probs, p_probs) — the water-balance posterior and the implied precipitation-
forcing marginal, which is what a forecaster reads as "what the chain thinks the rain does".
"""
function infer_chain(
    c_ev::Vector{Float64}, a_ev::Vector{Float64};
    m_ev::Union{Nothing,Vector{Float64}}=nothing,
    r_ev::Union{Nothing,Vector{Float64}}=nothing,
    p_ev::Union{Nothing,Vector{Float64}}=nothing,
    ro_ev::Union{Nothing,Vector{Float64}}=nothing,
    T_MC::Matrix{Float64}, T_RM::Matrix{Float64},
    T_PMR::Array{Float64,3}, T_WPA::Array{Float64,3}, T_ROW::Matrix{Float64},
)
    # forward: C -> M
    m = T_MC * c_ev
    m_ev !== nothing && (m .*= m_ev)
    s = sum(m); s > 0 && (m ./= s)

    # M -> R  (R also depends on M only)
    r = T_RM * m
    r_ev !== nothing && (r .*= r_ev)
    s = sum(r); s > 0 && (r ./= s)

    # (M, R) -> P
    p = zeros(4)
    @inbounds for mi in 1:4, ri in 1:4
        w = m[mi] * r[ri]
        w == 0 && continue
        for pi in 1:4
            p[pi] += T_PMR[pi, mi, ri] * w
        end
    end
    p_ev !== nothing && (p .*= p_ev)
    s = sum(p); s > 0 && (p ./= s)

    # (P, A) -> W
    wv = zeros(5)
    @inbounds for pi in 1:4, ai in 1:5
        wt = p[pi] * a_ev[ai]
        wt == 0 && continue
        for wi in 1:5
            wv[wi] += T_WPA[wi, pi, ai] * wt
        end
    end

    # W -> RO observed: multiply by the likelihood L(W) = sum_ro P(ro|W) * ro_ev(ro)
    if ro_ev !== nothing
        wv .*= vec(ro_ev' * T_ROW)
    end
    s = sum(wv); s > 0 && (wv ./= s)
    return wv, p
end

# ============================================================================
# RxInfer MODEL — same verified idiom as the flood original: each observed node gets a
# `DiscreteTransition(node, diageye(K))` channel carrying one-hot or soft evidence, and the
# queried node terminates its half-edge with a `missing` observation.
# ============================================================================

@model function s2s_wbp_model(T_MC, T_RM, T_PMR, T_WPA, T_ROW,
                              c_data, a_data, ro_data, w_data)
    c  ~ Categorical(fill(1/2, 2))
    a  ~ Categorical(fill(1/5, 5))
    c_data ~ DiscreteTransition(c, diageye(2))
    a_data ~ DiscreteTransition(a, diageye(5))

    m  ~ DiscreteTransition(c, T_MC)
    r  ~ DiscreteTransition(m, T_RM)
    p  ~ DiscreteTransition(m, T_PMR, r)
    w  ~ DiscreteTransition(p, T_WPA, a)
    ro ~ DiscreteTransition(w, T_ROW)

    ro_data ~ DiscreteTransition(ro, diageye(4))
    w_data  ~ DiscreteTransition(w, diageye(5))
end

_wbp_init = @initialization begin
    q(c)  = Categorical(fill(1/2, 2))
    q(a)  = Categorical(fill(1/5, 5))
    q(m)  = Categorical(fill(1/4, 4))
    q(r)  = Categorical(fill(1/4, 4))
    q(p)  = Categorical(fill(1/4, 4))
    q(w)  = Categorical(fill(1/5, 5))
    q(ro) = Categorical(fill(1/4, 4))
end

"""
Soft-evidence inference through RxInfer's message passing. Kept as the reference path;
`infer_chain` above is exact for this DAG and ~1000x cheaper per call, so bulk per-member
storyline runs use that instead.
"""
function infer_rxinfer(c_ev, a_ev, ro_ev, T_MC, T_RM, T_PMR, T_WPA, T_ROW; iterations::Int=10)
    res = infer(
        model = s2s_wbp_model(T_MC = T_MC, T_RM = T_RM, T_PMR = T_PMR,
                              T_WPA = T_WPA, T_ROW = T_ROW),
        data  = (c_data = c_ev, a_data = a_ev,
                 ro_data = ro_ev === nothing ? fill(0.25, 4) : ro_ev,
                 w_data = missing),
        iterations     = iterations,
        initialization = _wbp_init,
    )
    return Vector{Float64}(last(res.posteriors[:w]).p)
end

onehot(idx::Int, k::Int) = (v = zeros(Float64, k); v[idx] = 1.0; v)

# ============================================================================
# CRMA — two-sided, because this target node serves flood AND drought
# ============================================================================

"""
Cost-loss trigger on the water-balance posterior (Murphy 1977; Richardson 2000; Lopez et al.
2020): act when P(event) >= C/L. Applied INDEPENDENTLY to the surplus and the deficit tail,
because an anticipatory action for flooding and one for drought have different triggers and
must not be collapsed into a single "risk" number.
"""
function crma_state(tail_mass::Float64, near_mass::Float64; cost_loss_ratio::Float64=0.2)
    θ_act      = cost_loss_ratio
    θ_assess   = max(2.0 * cost_loss_ratio, 0.40)
    θ_evaluate = max(3.0 * cost_loss_ratio, 0.30)
    if tail_mass >= θ_act
        return 4, "P(tail)=$(round(tail_mass, digits=2)) ≥ C/L=$(round(θ_act, digits=2))"
    elseif near_mass >= θ_assess
        return 3, "P(tail∪near)=$(round(near_mass, digits=2)) ≥ $(round(θ_assess, digits=2))"
    elseif near_mass >= θ_evaluate
        return 2, "P(tail∪near)=$(round(near_mass, digits=2)) ≥ $(round(θ_evaluate, digits=2))"
    else
        return 1, "both tails below thresholds"
    end
end

function assess(w::Vector{Float64}; cost_loss_ratio::Float64=0.2)
    surplus_i, surplus_expl = crma_state(w[5], w[4] + w[5]; cost_loss_ratio)
    deficit_i, deficit_expl = crma_state(w[1], w[1] + w[2]; cost_loss_ratio)
    return (surplus = CRMA_STATES[surplus_i], surplus_expl = surplus_expl,
            surplus_light = TRAFFIC_LIGHT[CRMA_STATES[surplus_i]],
            deficit = CRMA_STATES[deficit_i], deficit_expl = deficit_expl,
            deficit_light = TRAFFIC_LIGHT[CRMA_STATES[deficit_i]])
end

# ============================================================================
# CSV DRIVER
# ============================================================================

"""Read the `{node}_p{k}` soft-evidence block for one node, or `nothing` if absent."""
function soft(row, colnames, prefix::String, k::Int)
    all("$(prefix)_p$i" in colnames for i in 1:k) || return nothing
    v = Float64[row["$(prefix)_p$i"] for i in 1:k]
    s = sum(v)
    return s > 0 ? v ./ s : fill(1.0 / k, k)
end

"""
Widen the elicited seam for the later lead window. Day 25-33 is a broader water-balance
tendency than day 18-24 (qd-1's own lead-time interpretation ladder), so its CPTs are wider.
"""
sigma_for(window::AbstractString) = occursin("W2", String(window)) ? 1.35 : 1.0

function run_csv(input_csv::String, output_csv::String;
                 cpt_json::Union{Nothing,String}=nothing,
                 evidence_mode::String="chain",
                 cost_loss_ratio::Float64=0.2,
                 use_rxinfer::Bool=false)
    @info "DOWNSTREAM CONSUMER (DIRECTION_EXPLANATION_FIRST.md): the primary product is the " *
          "circulation explanation record. Output below carries an ELICITED, unverified " *
          "P(W|P,A) and a MODEL-PROXY antecedent node — a structured hypothesis, not a " *
          "calibrated risk product."
    df = CSV.read(input_csv, DataFrame)
    colnames = names(df)
    counted = cpt_json === nothing ? Dict() : load_counted_cpts(cpt_json)

    if evidence_mode == "all"
        @warn "evidence-mode=all attaches the SAME 50 members as evidence at every chain " *
              "node; the posterior will be over-sharpened. Use it only for comparison."
    end

    T_PMR = build_P_given_MR()
    T_ROW = build_RO_given_W()
    out_rows = Vector{NamedTuple}(undef, nrow(df))

    for (i, row) in enumerate(eachrow(df))
        win  = "lead_window" in colnames ? String(row.lead_window) : "W1"
        unit = "unit_id" in colnames ? String(row.unit_id) : "unit_$i"
        blk  = "$(unit)|$(win)"

        T_MC = haskey(counted, blk) && haskey(counted[blk], :M_given_C) ?
               counted[blk][:M_given_C] : build_M_given_C()
        T_RM = haskey(counted, blk) && haskey(counted[blk], :R_given_M) ?
               counted[blk][:R_given_M] : build_R_given_M()
        T_WPA = build_W_given_PA(; sigma = sigma_for(win))

        c_ev  = something(soft(row, colnames, "C", 2), fill(0.5, 2))
        a_ev  = something(soft(row, colnames, "A", 5), fill(0.2, 5))
        ro_ev = soft(row, colnames, "RO", 4)
        m_ev  = evidence_mode == "all" ? soft(row, colnames, "M", 4) : nothing
        r_ev  = evidence_mode == "all" ? soft(row, colnames, "R", 4) : nothing
        p_ev  = evidence_mode == "all" ? soft(row, colnames, "P", 4) : nothing

        w, p = infer_chain(c_ev, a_ev; m_ev, r_ev, p_ev, ro_ev,
                           T_MC, T_RM, T_PMR, T_WPA, T_ROW)
        if use_rxinfer
            w = infer_rxinfer(c_ev, a_ev, ro_ev, T_MC, T_RM, T_PMR, T_WPA, T_ROW)
        end
        cr = assess(w; cost_loss_ratio)

        out_rows[i] = (
            unit_id = unit,
            unit_name = "unit_name" in colnames ? String(row.unit_name) : unit,
            lead_window = win,
            lead_hours = "lead_hours" in colnames ? String(row.lead_hours) : "",
            ess = "ess" in colnames ? Float64(row.ess) : NaN,
            wbp_state = W_STATES[argmax(w)],
            w_strong_deficit = w[1], w_mild_deficit = w[2], w_balanced = w[3],
            w_elevated_surplus = w[4], w_extreme_surplus = w[5],
            p_surplus = w[4] + w[5], p_deficit = w[1] + w[2],
            surplus_crma = cr.surplus, surplus_light = cr.surplus_light,
            surplus_explanation = cr.surplus_expl,
            deficit_crma = cr.deficit, deficit_light = cr.deficit_light,
            deficit_explanation = cr.deficit_expl,
            precip_forcing_state = P_STATES[argmax(p)],
            p_heavy = p[3] + p[4],
            evidence_mode = evidence_mode,
            chain_cpt_source = haskey(counted, blk) ? "counted(n_cycles=1)" : "elicited",
            seam_note = "P(W|P,A) and P(RO|W) are ELICITED — no observation closes them yet",
        )
    end

    out = DataFrame(out_rows)
    mkpath(dirname(abspath(output_csv)))
    CSV.write(output_csv, out)
    @info "wrote $output_csv rows=$(nrow(out))"
    @info "water-balance state distribution:" combine(groupby(out, :wbp_state), nrow => :n)
    @info "surplus CRMA:" combine(groupby(out, :surplus_crma), nrow => :n)
    @info "deficit CRMA:" combine(groupby(out, :deficit_crma), nrow => :n)
end

"""
Per-member storyline run. Each member is one internally consistent atmospheric future, so
its states are HARD evidence for that member's world; the spread across members is the
uncertainty. Worst / median / best are picked on the surplus tail.
"""
function run_storylines(member_csv::String, storyline_csv::String; cost_loss_ratio::Float64=0.2)
    df = CSV.read(member_csv, DataFrame)
    T_PMR = build_P_given_MR()
    T_ROW = build_RO_given_W()
    rows = NamedTuple[]

    for g in groupby(df, [:unit_id, :lead_window])
        win = String(g[1, :lead_window])
        T_MC, T_RM = build_M_given_C(), build_R_given_M()
        T_WPA = build_W_given_PA(; sigma = sigma_for(win))
        per = NamedTuple[]
        for r in eachrow(g)
            w, _ = infer_chain(onehot(Int(r.C_idx), 2), onehot(Int(r.A_idx), 5);
                               m_ev = onehot(Int(r.M_idx), 4),
                               r_ev = onehot(Int(r.R_idx), 4),
                               p_ev = onehot(Int(r.P_idx), 4),
                               ro_ev = onehot(Int(r.RO_idx), 4),
                               T_MC, T_RM, T_PMR, T_WPA, T_ROW)
            push!(per, (member = r.member, w = w, p_surplus = w[4] + w[5],
                        p_deficit = w[1] + w[2], tp_mm = r.tp_window_mm))
        end
        sorted = sort(per, by = x -> x.p_surplus, rev = true)
        n = length(sorted)
        for (label, pick) in (("worst_surplus", sorted[1]),
                              ("median",        sorted[div(n, 2) + 1]),
                              ("worst_deficit", sorted[n]))
            n_ge = count(x -> x.p_surplus >= pick.p_surplus, sorted)
            cr = assess(pick.w; cost_loss_ratio)
            push!(rows, (storyline = label, unit_id = String(g[1, :unit_id]),
                         lead_window = win, member = pick.member,
                         wbp_state = W_STATES[argmax(pick.w)],
                         p_surplus = round(pick.p_surplus, digits=3),
                         p_deficit = round(pick.p_deficit, digits=3),
                         tp_window_mm = pick.tp_mm,
                         surplus_crma = cr.surplus, deficit_crma = cr.deficit,
                         exceedance = round(n_ge / n, digits=3), n_members = n))
        end
    end

    out = DataFrame(rows)
    mkpath(dirname(abspath(storyline_csv)))
    CSV.write(storyline_csv, out)
    @info "wrote $storyline_csv rows=$(nrow(out))"
end

# ============================================================================
# SELF-TEST
# ============================================================================

function self_test()
    @info "Running self-test..."
    T_MC, T_RM = build_M_given_C(), build_R_given_M()
    T_PMR, T_ROW = build_P_given_MR(), build_RO_given_W()
    T_WPA = build_W_given_PA()

    for (name, T) in (("M|C", T_MC), ("R|M", T_RM), ("RO|W", T_ROW))
        @assert all(isapprox.(sum(T, dims=1), 1.0; atol=1e-8)) "$name columns must normalise"
    end
    @assert all(isapprox.(sum(T_PMR, dims=1), 1.0; atol=1e-8)) "P|M,R must normalise"
    @assert all(isapprox.(sum(T_WPA, dims=1), 1.0; atol=1e-8)) "W|P,A must normalise"

    # 1. Convergent circulation + saturated ground -> surplus-leaning
    w_wet, _ = infer_chain(onehot(2, 2), onehot(5, 5); T_MC, T_RM, T_PMR, T_WPA, T_ROW)
    # 2. Unfavourable circulation + very dry ground -> deficit-leaning
    w_dry, _ = infer_chain(onehot(1, 2), onehot(1, 5); T_MC, T_RM, T_PMR, T_WPA, T_ROW)
    @info "wet-case" state=W_STATES[argmax(w_wet)] p_surplus=round(w_wet[4]+w_wet[5], digits=3)
    @info "dry-case" state=W_STATES[argmax(w_dry)] p_deficit=round(w_dry[1]+w_dry[2], digits=3)
    @assert (w_wet[4] + w_wet[5]) > (w_dry[4] + w_dry[5]) "wet case must carry more surplus mass"
    @assert (w_dry[1] + w_dry[2]) > (w_wet[1] + w_wet[2]) "dry case must carry more deficit mass"

    # 3. Runoff evidence corroborates but must not dominate (grade B likelihood)
    w_ro_hi, _ = infer_chain(onehot(2, 2), onehot(3, 5); ro_ev = onehot(4, 4),
                             T_MC, T_RM, T_PMR, T_WPA, T_ROW)
    w_ro_no, _ = infer_chain(onehot(2, 2), onehot(3, 5);
                             T_MC, T_RM, T_PMR, T_WPA, T_ROW)
    shift = (w_ro_hi[4] + w_ro_hi[5]) - (w_ro_no[4] + w_ro_no[5])
    @info "runoff corroboration shift" shift=round(shift, digits=3)
    @assert shift > 0 "high runoff must raise surplus mass"
    @assert shift < 0.30 "runoff is grade B — it must not dominate the posterior"

    # 4. The later window must be less certain than the earlier one
    w_w1, _ = infer_chain(onehot(2, 2), onehot(5, 5); T_MC, T_RM, T_PMR,
                          T_WPA = build_W_given_PA(sigma = sigma_for("W1_wk3")), T_ROW)
    w_w2, _ = infer_chain(onehot(2, 2), onehot(5, 5); T_MC, T_RM, T_PMR,
                          T_WPA = build_W_given_PA(sigma = sigma_for("W2_wk45")), T_ROW)
    ent(v) = -sum(p * log(max(p, 1e-12)) for p in v)
    @info "lead-window entropy" w1=round(ent(w_w1), digits=3) w2=round(ent(w_w2), digits=3)
    @assert ent(w_w2) > ent(w_w1) "the later lead window must be wider"

    @info "All self-tests passed."
end

# ============================================================================
# CLI
# ============================================================================

function getarg(flag::String)
    i = findfirst(==(flag), ARGS)
    return i === nothing || i == length(ARGS) ? nothing : ARGS[i + 1]
end

function main()
    if "--test" in ARGS
        self_test()
        return
    end

    member_csv = getarg("--member-csv")
    storyline_csv = getarg("--storyline-csv")
    cl = getarg("--cost-loss-ratio")
    cost_loss_ratio = cl === nothing ? 0.2 : parse(Float64, cl)

    if member_csv !== nothing && storyline_csv !== nothing
        run_storylines(member_csv, storyline_csv; cost_loss_ratio)
        return
    end

    input_csv = getarg("--input-csv")
    output_csv = getarg("--output-csv")
    if input_csv !== nothing && output_csv !== nothing
        run_csv(input_csv, output_csv;
                cpt_json = getarg("--cpt-json"),
                evidence_mode = something(getarg("--evidence-mode"), "chain"),
                cost_loss_ratio = cost_loss_ratio,
                # exact tensor contraction is the default; --rxinfer switches to message
                # passing, which is equivalent on this DAG but far slower per call
                use_rxinfer = "--rxinfer" in ARGS)
        return
    end

    @info "S2S catchment water-balance BN (AIFS-ENS v2 / Icechunk / RxInfer)"
    @info "  julia s2s_water_balance_bn.jl --input-csv IN.csv --output-csv OUT.csv " *
          "[--cpt-json CPT.json] [--evidence-mode chain|all] [--rxinfer]"
    @info "  julia s2s_water_balance_bn.jl --member-csv M.csv --storyline-csv S.csv"
    @info "  julia s2s_water_balance_bn.jl --test"
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
