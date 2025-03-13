import Random
import Statistics
import StatsBase
import FFTW
import LsqFit
import Interpolations
import Plots
import Base.Threads
import Measures

using Random: rand
using Statistics: mean, std
using StatsBase: countmap, sample
using FFTW: fft, fftfreq
using LsqFit: curve_fit
using Interpolations: interpolate, Gridded, Linear, extrapolate, Flat
using Plots: heatmap!, plot!, savefig, plot, scatter!, hline!, xlabel!, ylabel!, annotate!
using Measures, Plots

if !isdir("results")
    mkpath("results")
end

"""
Block-average data to estimate mean and standard error.
"""
function block_average(data::Vector{T}, block_size::Int) where T<:Real
    N = length(data)
    nblocks = div(N, block_size)
    block_means = Vector{Float64}(undef, nblocks)
    for i in 1:nblocks
        start_idx = (i-1)*block_size + 1
        end_idx = i*block_size
        block_means[i] = mean(data[start_idx:end_idx])
    end
    return mean(block_means), std(block_means)/sqrt(nblocks)
end

"""
Compute FWHM of a peak in 1D data (x, y) using linear interpolation.
"""
function calculate_fwhm(x::Vector{Float64}, y::Vector{Float64})
    max_y = maximum(y)
    half_max = max_y / 2
    above_half = findall(y .>= half_max)
    if isempty(above_half)
        return 0.0
    end
    left_idx = minimum(above_half)
    right_idx = maximum(above_half)
    if left_idx > 1 && right_idx < length(x)
        left_x = x[left_idx-1] + (x[left_idx] - x[left_idx-1]) *
                 (half_max - y[left_idx-1]) / (y[left_idx] - y[left_idx-1])
        right_x = x[right_idx] + (x[right_idx+1] - x[right_idx]) *
                  (half_max - y[right_idx]) / (y[right_idx+1] - y[right_idx])
        return abs(right_x - left_x)
    else
        return abs(x[right_idx] - x[left_idx])
    end
end

"""
Initialize an AFM spin lattice with given vacancy fraction.
"""
function initialize_lattice(size::Int, vacancy_fraction::Float64=0.2)
    total_sites = size * size
    num_vacancies = Int(round(vacancy_fraction * total_sites))
    indices = collect(1:total_sites)
    vacancy_indices = sample(indices, num_vacancies, replace=false)
    s = Array{Int,2}(undef, size, size)
    for i in 1:size, j in 1:size
        s[i,j] = (-1)^(i + j)
    end
    s[vacancy_indices] .= 0
    return s
end

"""
Periodic boundary index.
"""
function pbc(idx, size)
    wrapped = mod(idx - 1, size)
    return wrapped + 1
end

"""
Energy difference for flipping spin s[i, j].
"""
function deltaE(s::Array{Int,2}, i::Int, j::Int, size::Int; J::Int=-1)
    if s[i,j] == 0
        return 0
    end
    top_i    = pbc(i - 1, size)
    bottom_i = pbc(i + 1, size)
    left_j   = pbc(j - 1, size)
    right_j  = pbc(j + 1, size)
    top    = s[top_i, j]
    bottom = s[bottom_i, j]
    left   = s[i, left_j]
    right  = s[i, right_j]
    sum_neighbors = 0
    if top != 0
        sum_neighbors += top
    end
    if bottom != 0
        sum_neighbors += bottom
    end
    if left != 0
        sum_neighbors += left
    end
    if right != 0
        sum_neighbors += right
    end
    return 2 * J * s[i, j] * sum_neighbors
end

"""
One Metropolis sweep at temperature T.
"""
function monte_carlo_step!(s::Array{Int,2}, T::Float64, size::Int; J::Int=-1)
    for _ in 1:(size^2)
        i = rand(1:size)
        j = rand(1:size)
        if s[i,j] == 0
            continue
        end
        Ediff = deltaE(s, i, j, size, J=J)
        if Ediff <= 0 || rand() < exp(-Ediff / T)
            s[i,j] *= -1
        end
    end
end

"""
Compute |m_s|, energy/bond, and m_s^2.
"""
function calculate_properties(s::Array{Int,2}, size::Int)
    num_sites = 0
    staggered_magnetization = 0.0
    for i in 1:size, j in 1:size
        s_ij = s[i,j]
        if s_ij != 0
            staggered_magnetization += (-1)^(i + j) * s_ij
            num_sites += 1
        end
    end
    if num_sites > 0
        staggered_magnetization /= num_sites
    else
        staggered_magnetization = 0.0
    end
    energy = 0.0
    num_bonds = 0
    for i in 1:size, j in 1:size
        s_ij = s[i,j]
        if s_ij == 0
            continue
        end
        neighbors = [(pbc(i+1, size), j),
                     (i, pbc(j+1, size))]
        for (ni, nj) in neighbors
            s_n = s[ni, nj]
            if s_n != 0
                energy += s_ij * s_n
                num_bonds += 1
            end
        end
    end
    if num_bonds > 0
        energy /= num_bonds
    else
        energy = 0.0
    end
    return abs(staggered_magnetization), energy, staggered_magnetization^2
end

"""
Return S(Q) = |FFT(s)|^2 / N_occ and corresponding Q_x, Q_y.
"""
function compute_structure_factor(s::Array{Int,2})
    lattice_size = size(s, 1)
    S_k = fft(s)
    N_occ = count(!=(0), s)
    if N_occ == 0
        error("All sites are vacancies! (N_occ=0)")
    end
    S_Q = abs.(S_k).^2 ./ N_occ
    Q_x = [2π*(i-1)/lattice_size for i in 1:lattice_size]
    Q_y = [2π*(j-1)/lattice_size for j in 1:lattice_size]
    return S_Q, Q_x, Q_y
end

"""
Compute C = (⟨E²⟩ - ⟨E⟩²)/T².
"""
function calculate_specific_heat(energies::Vector{Float64}, T::Float64)
    if T == 0
        return NaN
    end
    return (mean(energies.^2) - mean(energies)^2) / T^2
end

"""
Return T where specific heat is maximal.
"""
function estimate_transition_temperature(temperatures::Vector{Float64}, 
                                         specific_heats::Vector{Float64})
    max_idx = argmax(specific_heats)
    return temperatures[max_idx]
end

"""
Fit Lorentzian near S(Q) peak to estimate correlation length.
"""
function calculate_correlation_length(S_Q::AbstractMatrix{<:Real}, 
                                      Q_x::Vector{Float64}, 
                                      Q_y::Vector{Float64})
    max_val, max_idx = findmax(S_Q)
    if max_val <= 0
        return NaN
    end
    peak_idx = CartesianIndices(S_Q)[max_idx]
    peak_idx_x, peak_idx_y = peak_idx[1], peak_idx[2]
    lattice_size = length(Q_x)
    Q_range = min(5, lattice_size ÷ 8)
    left_bound  = max(1, peak_idx_x - Q_range)
    right_bound = min(lattice_size, peak_idx_x + Q_range)
    x_indices = left_bound:right_bound
    S_Q_line = S_Q[x_indices, peak_idx_y]
    Q_line   = Q_x[x_indices] .- Q_x[peak_idx_x]
    if length(S_Q_line) < 5 || maximum(S_Q_line) <= 0
        return NaN
    end
    S_Q_line .= S_Q_line ./ maximum(S_Q_line)
    model(x, p) = 1.0 ./ (1.0 .+ (x ./ p[1]).^2)
    p0 = [0.1]
    try
        fit = curve_fit(model, Q_line, S_Q_line, p0)
        if fit.converged && fit.param[1] > 0
            width = abs(fit.param[1])
            return 1 / width
        else
            return NaN
        end
    catch e
        @warn "Fitting error for correlation length: $e"
        return NaN
    end
end

"""
Fit inverse correlation length (approx A * t^ν).
"""
function fit_nu(reduced_temps::Vector{Float64}, inv_corr_lengths::Vector{Float64})
    model(x, θ) = θ[1] .* x.^θ[2]
    θ0 = [1.0, 1.0]
    idxs = findall(x -> x > 0, reduced_temps)
    if length(idxs) < 2
        return (A=NaN, ν=NaN)
    end
    x_data = reduced_temps[idxs]
    y_data = inv_corr_lengths[idxs]
    valid_data = .!(isnan.(x_data) .| isinf.(x_data) .| isnan.(y_data) .| isinf.(y_data))
    if sum(valid_data) < 2
        return (A=NaN, ν=NaN)
    end
    x_data = x_data[valid_data]
    y_data = y_data[valid_data]
    try
        fit = curve_fit(model, x_data, y_data, θ0)
        A_fit, ν_fit = fit.param
        return (A=A_fit, ν=ν_fit)
    catch e
        println("Fitting error in fit_nu: ", e)
        return (A=NaN, ν=NaN)
    end
end

"""
Run Metropolis for 2D AFM Ising with vacancies, return results.
"""
function run_ising_simulation(; 
    size=64, 
    T_min=1.0, 
    T_max=4.0, 
    T_step=0.1, 
    vacancy_fraction=0.1,
    steps_per_temp=2000,
    equilibration_steps=1000
)
    temperatures = T_min:T_step:T_max
    mag_history     = zeros(length(temperatures))
    energy_history  = zeros(length(temperatures))
    msq_history     = zeros(length(temperatures))
    S_Q_history     = Vector{Matrix{Float64}}()
    Q_x, Q_y        = nothing, nothing
    snapshot_temps = [1.0, 2.0, 3.0, 4.0]
    n_snapshots = length(snapshot_temps)
    p_snapshots = plot(layout=(2, ceil(Int, n_snapshots/2)), size=(1000,800), plot_margin=150mm)
    labels_for_snapshots = ["a)", "b)", "c)", "d)"]
    idx_snapshot_plots = 1
    for (idx, T) in enumerate(temperatures)
        s = initialize_lattice(size, vacancy_fraction)
        for _ in 1:equilibration_steps
            monte_carlo_step!(s, T, size, J=-1)
        end
        mags     = Float64[]
        energies = Float64[]
        msqs     = Float64[]
        S_Q_accum = zeros(Float64, size, size)
        for _ in 1:steps_per_temp
            monte_carlo_step!(s, T, size, J=-1)
            m, e, m_sq = calculate_properties(s, size)
            push!(mags, m)
            push!(energies, e)
            push!(msqs, m_sq)
            this_S_Q, Q_x_tmp, Q_y_tmp = compute_structure_factor(s)
            S_Q_accum .+= this_S_Q
            if Q_x === nothing
                Q_x = Q_x_tmp
                Q_y = Q_y_tmp
            end
        end
        mag_history[idx]    = mean(mags)
        energy_history[idx] = mean(energies)
        msq_history[idx]    = size^2 * mean(msqs)
        push!(S_Q_history, S_Q_accum ./ steps_per_temp)
        if T in snapshot_temps
            heatmap!(p_snapshots[idx_snapshot_plots],
                     s,
                     aspect_ratio=1,
                     c=[:blue, :white, :red],
                     clims=(-1,1),
                     axis=false)
            annotate!(
                p_snapshots[idx_snapshot_plots],
                (-0.18, 0.9),
                text(labels_for_snapshots[idx_snapshot_plots], 10),
                :relative
            )
            idx_snapshot_plots += 1
        end
    end
    savefig(p_snapshots, joinpath("results","ising_snapshots_with_vacancies.png"))
    return temperatures, mag_history, energy_history, msq_history, S_Q_history, Q_x, Q_y
end

"""
Compare results for vacancy_fraction=0.0, 0.15, 0.3.
"""
function compare_vacancies_example()
    temps_no_vac, mags_no_vac, energies_no_vac, msq_no_vac, S_Q_history_no_vac, Q_x_no_vac, Q_y_no_vac =
        run_ising_simulation(size=64,
                             T_min=1.0,
                             T_max=4.0,
                             T_step=0.1,
                             vacancy_fraction=0.0,
                             steps_per_temp=2000,
                             equilibration_steps=800)
    temps_vac, mags_vac, energies_vac, msq_vac, S_Q_history_vac, Q_x, Q_y =
        run_ising_simulation(size=64,
                             T_min=1.0,
                             T_max=4.0,
                             T_step=0.1,
                             vacancy_fraction=0.15,
                             steps_per_temp=2000,
                             equilibration_steps=800)
    temps_1, mags_vac_1, energies_vac_1, msq_vac_1, S_Q_history_vac_1, Q_x_1, Q_y_1 =
        run_ising_simulation(size=64,
                             T_min=1.0,
                             T_max=4.0,
                             T_step=0.1,
                             vacancy_fraction=0.3,
                             steps_per_temp=2000,
                             equilibration_steps=800)
    p_final = plot(layout=(1,3), size=(1200,400), plot_margin=150mm, 
                   left_margin=6mm, bottom_margin=6mm)
    labels_for_subplots = ["a)", "b)", "c)"]
    plot!(p_final[1],
          temps_no_vac, mags_no_vac,
          xlabel="Temperature",
          ylabel="|m_s|",
          marker=:diamond, legend=:top, label="Vac=0.0")
    plot!(p_final[1],
          temps_vac, mags_vac,
          marker=:circle, label="Vac=0.15")
    plot!(p_final[1],
          temps_1, mags_vac_1,
          marker=:square, label="Vac=0.3")
    annotate!(p_final[1],
              (-0.18, 0.9),
              text(labels_for_subplots[1], 10),
              :relative)
    plot!(p_final[2],
          temps_no_vac, energies_no_vac,
          xlabel="Temperature",
          ylabel="Energy/bond",
          marker=:diamond, legend=:top, label="Vac=0.0")
    plot!(p_final[2],
          temps_vac, energies_vac,
          marker=:circle, label="Vac=0.15")
    plot!(p_final[2],
          temps_1, energies_vac_1,
          marker=:square, label="Vac=0.3")
    annotate!(p_final[2],
              (-0.16, 0.9),
              text(labels_for_subplots[2], 10),
              :relative)
    plot!(p_final[3],
          temps_no_vac, msq_no_vac,
          xlabel="Temperature",
          ylabel="m_s^2 (scaled)",
          marker=:diamond, legend=:top, label="Vac=0.0")
    plot!(p_final[3],
          temps_vac, msq_vac,
          marker=:circle, label="Vac=0.15")
    plot!(p_final[3],
          temps_1, msq_vac_1,
          marker=:square, label="Vac=0.3")
    annotate!(p_final[3],
              (-0.18, 0.9),
              text(labels_for_subplots[3], 10),
              :relative)
    savefig(p_final, joinpath("results","ising_model_final_plots_with_and_without_vacancies.png"))
end

"""
Plot S(Q_x, π) for T=1..4 from S_Q_history.
"""
function plot_structure_factor_linecut(temps, S_Q_history, Q_x, Q_y; 
                                       savefile="structure_factor_comparison_all_temps.png")
    temp_values = [1.0, 2.0, 3.0, 4.0]
    p_struct_all = plot(layout=(2,2), size=(1000,800), plot_margin=150mm)
    labels_for_subplots = ["a)", "b)", "c)", "d)"]
    lattice_size = length(Q_x)
    j_pi = floor(lattice_size/2) + 1
    plot_index = 1
    for temp in temp_values
        idx_temp = findfirst(x -> isapprox(x, temp; atol=1e-6), temps)
        if idx_temp === nothing
            continue
        end
        S_Q_avg = S_Q_history[idx_temp]
        S_Q_line = S_Q_avg[:, j_pi]
        plot!(p_struct_all[plot_index],
              Q_x, S_Q_line,
              label="T=$(temp)",
              xlabel="Q_x",
              ylabel="S(Q_x, π)",
              marker=:circle)
        annotate!(p_struct_all[plot_index],
                  (-0.18, 0.9),
                  text(labels_for_subplots[plot_index], 10),
                  :relative)
        plot_index += 1
    end
    savefig(p_struct_all, joinpath("results", savefile))
end

"""
Compare line-cuts of S(Q_x, π) for vac vs. no vac at diff temps
"""
function compare_structure_factor_linecut_vacancies(vac_fraction)
    T_vals = [1.0, 2.0, 3.0, 4.0]
    temps_vac, _, _, _, S_Q_hist_vac, Qx_vac, Qy_vac =
        run_ising_simulation(size=64, T_min=1.0, T_max=4.0, T_step=1.0,
                             vacancy_fraction=vac_fraction,
                             steps_per_temp=1000, equilibration_steps=500)
    temps_novac, _, _, _, S_Q_hist_novac, Qx_novac, Qy_novac =
        run_ising_simulation(size=64, T_min=1.0, T_max=4.0, T_step=1.0,
                             vacancy_fraction=0.0,
                             steps_per_temp=1000, equilibration_steps=500)
    p = plot(layout=(4,1), size=(800,1200), plot_margin=150mm,
             left_margin=6mm, bottom_margin=1mm,
             right_margin=2mm, top_margin=2mm)
    labels_for_subplots = ["a)", "b)", "c)", "d)"]
    lattice_size = length(Qx_vac)
    j_pi_vac   = Int(floor(lattice_size/2) + 1)
    j_pi_novac = j_pi_vac
    for (i, T) in enumerate(T_vals)
        idx_vac   = findfirst(x -> isapprox(x, T; atol=1e-6), temps_vac)
        idx_novac = findfirst(x -> isapprox(x, T; atol=1e-6), temps_novac)
        S_Q_avg_vac    = S_Q_hist_vac[idx_vac]
        S_Q_avg_novac  = S_Q_hist_novac[idx_novac]
        line_vac   = S_Q_avg_vac[:, j_pi_vac]
        line_novac = S_Q_avg_novac[:, j_pi_novac]
        plot!(p[i], 
              Qx_novac, line_novac,
              label="Vac=0.0",
              xlabel="Q_x",
              ylabel="S(Q_x, π)",
              marker=:square)
        plot!(p[i], 
              Qx_vac, line_vac,
              label="Vac=$(vac_fraction)",
              marker=:circle)
        annotate!(p[i],
                  (-0.18, 0.9),
                  text(labels_for_subplots[i], 10),
                  :relative)
    end
    savefig(p, joinpath("results","compare_structure_factor_linecut_vac_fraction_$(vac_fraction).png"))
end

"""
Scan vacancy fractions, measure T_c and peak widths, etc
"""
function analyze_vacancy_dependence(;
    size=32,
    T_min=1.0,
    T_max=4.0,
    T_step=0.04,
    vacancy_fractions=Vector(0.0:0.06:0.30)
)
    temperatures = collect(T_min:T_step:T_max)
    transition_temps         = Float64[]
    specific_heat_widths     = Float64[]
    structure_factor_widths  = Float64[]
    equilibration_steps = 1200
    measurement_steps   = 5000
    for vf in vacancy_fractions
        specific_heats      = Float64[]
        structure_factors   = Float64[]
        correlation_lengths = Float64[]
        Q_x_data = nothing
        Q_y_data = nothing
        for T in temperatures
            s = initialize_lattice(size, vf)
            for _ in 1:equilibration_steps
                monte_carlo_step!(s, T, size)
            end
            energies    = Float64[]
            S_Q_accum   = zeros(Float64, size, size)
            for _ in 1:measurement_steps
                monte_carlo_step!(s, T, size)
                _, e, _ = calculate_properties(s, size)
                push!(energies, e)
                this_S_Q, Qx_tmp, Qy_tmp = compute_structure_factor(s)
                S_Q_accum .+= this_S_Q
                if Q_x_data === nothing
                    Q_x_data = Qx_tmp
                    Q_y_data = Qy_tmp
                end
            end
            S_Q_avg = S_Q_accum ./ measurement_steps
            corr_length = calculate_correlation_length(S_Q_avg, Q_x_data, Q_y_data)
            push!(correlation_lengths, corr_length)
            C_v = calculate_specific_heat(energies, T)
            push!(specific_heats, C_v)
            push!(structure_factors, S_Q_avg[1, 1])
        end
        T_c = estimate_transition_temperature(temperatures, specific_heats)
        push!(transition_temps, T_c)
        push!(specific_heat_widths, calculate_fwhm(temperatures, specific_heats))
        push!(structure_factor_widths, calculate_fwhm(temperatures, structure_factors))
    end
    p_transition = plot(layout=(1,3), size=(1200,400), plot_margin=150mm, 
                        left_margin=5mm, bottom_margin=5mm)
    labels_for_subplots = ["a)", "b)", "c)"]
    plot!(p_transition[1],
          collect(vacancy_fractions), transition_temps,
          xlabel="Vacancy Fraction",
          ylabel="T_c",
          marker=:circle,
          label="")
    hline!(p_transition[1], [2.269], linestyle=:dash, color=:red, label="Clean ~2.269")
    annotate!(p_transition[1],
              (-0.18, 0.9),
              text(labels_for_subplots[1], 10),
              :relative)
    plot!(p_transition[2],
          collect(vacancy_fractions), specific_heat_widths,
          xlabel="Vacancy Fraction",
          ylabel="Specific Heat Peak Width",
          marker=:circle,
          label="")
    annotate!(p_transition[2],
              (-0.17, 0.9),
              text(labels_for_subplots[2], 10),
              :relative)
    plot!(p_transition[3],
          collect(vacancy_fractions), structure_factor_widths,
          xlabel="Vacancy Fraction",
          ylabel="Structure Factor Peak Width",
          marker=:circle,
          label="")
    annotate!(p_transition[3],
              (-0.17, 1.0),
              text(labels_for_subplots[3], 10),
              :relative)
    savefig(p_transition, joinpath("results","vacancy_analysis_summary.png"))
    return (collect(vacancy_fractions), transition_temps, specific_heat_widths,
            structure_factor_widths)
end

"""
Compute M, Binder, Chi, C_v for vac=0.0,0.1,0.2.
"""
function advanced_observables_comparison_vacancies()
    vac_list = [0.0, 0.1, 0.2]
    T_list = 1.0:0.2:4.0
    magnet_dict = Dict{Float64,Vector{Float64}}()
    binder_dict = Dict{Float64,Vector{Float64}}()
    chi_dict    = Dict{Float64,Vector{Float64}}()
    cv_dict     = Dict{Float64,Vector{Float64}}()
    for vac in vac_list
        Ms   = Float64[]
        Bs   = Float64[]
        Chis = Float64[]
        CVs  = Float64[]
        for T in T_list
            size = 64
            eq_steps = 800
            meas_steps = 800
            s = initialize_lattice(size, vac)
            for _ in 1:eq_steps
                monte_carlo_step!(s, T, size)
            end
            mag_samples = Float64[]
            E_samples   = Float64[]
            m2_samples  = Float64[]
            m4_samples  = Float64[]
            for _ in 1:meas_steps
                monte_carlo_step!(s, T, size)
                m_abs, e, m_sq = calculate_properties(s, size)
                push!(mag_samples, m_abs)
                push!(E_samples, e)
                push!(m2_samples, m_sq)
                push!(m4_samples, m_sq^2)
            end
            M_avg   = mean(mag_samples)
            M2_avg  = mean(m2_samples)
            M4_avg  = mean(m4_samples)
            if M2_avg != 0
                U_binder = 1.0 - (M4_avg)/(3*(M2_avg^2))
            else
                U_binder = NaN
            end
            chi_val = (M2_avg - M_avg^2)/T
            E2 = mean(E_samples.^2)
            E1 = mean(E_samples)
            cv_val = (E2 - E1^2)/(T^2)
            push!(Ms, M_avg)
            push!(Bs, U_binder)
            push!(Chis, chi_val)
            push!(CVs, cv_val)
        end
        magnet_dict[vac] = Ms
        binder_dict[vac] = Bs
        chi_dict[vac]    = Chis
        cv_dict[vac]     = CVs
    end
    p = plot(layout=(2,2), size=(800,600), plot_margin=150mm)
    labels_for_subplots = ["a)", "b)", "c)", "d)"]
    plot!(p[1], T_list, magnet_dict[0.0],
          marker=:o, label="Vac=0.0",
          xlabel="T", ylabel="|m_s|")
    plot!(p[1], T_list, magnet_dict[0.1],
          marker=:d, label="Vac=0.1")
    plot!(p[1], T_list, magnet_dict[0.2],
          marker=:o, label="Vac=0.2")
    annotate!(p[1],
              (-0.18, 0.9),
              text(labels_for_subplots[1], 10),
              :relative)
    plot!(p[2], T_list, binder_dict[0.0],
          marker=:square, label="Vac=0.0",
          xlabel="T", ylabel="Binder U")
    plot!(p[2], T_list, binder_dict[0.1],
          marker=:diamond, label="Vac=0.1")
    plot!(p[2], T_list, binder_dict[0.2],
          marker=:square, label="Vac=0.2")
    annotate!(p[2],
              (-0.18, 0.9),
              text(labels_for_subplots[2], 10),
              :relative)
    plot!(p[3], T_list, chi_dict[0.0],
          marker=:hex, label="Vac=0.0",
          xlabel="T", ylabel="χ")
    plot!(p[3], T_list, chi_dict[0.1],
          marker=:diamond, label="Vac=0.1")
    plot!(p[3], T_list, chi_dict[0.2],
          marker=:hex, label="Vac=0.2")
    annotate!(p[3],
              (-0.17, 0.8),
              text(labels_for_subplots[3], 10),
              :relative)
    plot!(p[4], T_list, cv_dict[0.0],
          marker=:triangle, label="Vac=0.0",
          xlabel="T", ylabel="C_v")
    plot!(p[4], T_list, cv_dict[0.1],
          marker=:utriangle, label="Vac=0.1")
    plot!(p[4], T_list, cv_dict[0.2],
          marker=:triangle, label="Vac=0.2")
    annotate!(p[4],
              (-0.25, 0.8),
              text(labels_for_subplots[4], 10),
              :relative)
    savefig(p, joinpath("results", "advanced_observables_comparison_vacancies.png"))
end

"""
Run T=1..4 with vac=0.1, save snapshots.
"""
function ising_snapshots_with_vacancies()
    run_ising_simulation(size=32, 
                         T_min=1.0, 
                         T_max=4.0, 
                         T_step=1.0,
                         vacancy_fraction=0.1,
                         steps_per_temp=1000, 
                         equilibration_steps=500)
    println("Snapshots saved to 'ising_snapshots_with_vacancies.png' (vac=0.1).")
end

"""
Plot S(Q_x, π) for vac=0.0,0.1,0.2 at T=1,2,3 in one figure.
"""
function compare_structure_factor_linecut_allvac_oneplot()
    vacs = [0.0, 0.1, 0.2]
    Tvals = 1:3
    data_dict = Dict{Float64, 
                Tuple{Vector{Float64}, Vector{Matrix{Float64}}, Vector{Float64}, Vector{Float64}}}()
    for v in vacs
        temps, _, _, _, S_Q_hist, Qx, Qy = run_ising_simulation(
            size=64,
            T_min=1.0,
            T_max=3.0,
            T_step=1.0,
            vacancy_fraction=v,
            steps_per_temp=1000,
            equilibration_steps=500
        )
        data_dict[v] = (temps, S_Q_hist, Qx, Qy)
    end
    p = plot(layout=(1,3), size=(1400, 400), plot_margin=10mm)
    for (i, T) in enumerate(Tvals)
        for v in vacs
            (temps, S_Q_hist, Qx, Qy) = data_dict[v]
            idx_temp = findfirst(x -> isapprox(x, T; atol=1e-6), temps)
            S_Q_avg = S_Q_hist[idx_temp]
            j_pi = floor(Int, length(Qy)/2) + 1
            S_Q_line = S_Q_avg[:, j_pi]
            plot!(
                p[i],
                Qx, 
                S_Q_line,
                label = "Vac=$(v)",
                marker = :circle,
                markersize = 4,
                lw = 2
            )
        end
        plot!(
            p[i],
            xlim = (0, 2π),
            xticks = ([0, π/3, 2π/3, π, 4π/3, 5π/3, 2π],
                      ["0", "π/3", "2π/3", "π", "4π/3", "5π/3", "2π"]),
            xlabel = "Q_x",
            ylabel = "S(Q_x, π)",
            title  = "T = $(T).0"
        )
    end
    savefig(p, joinpath("results","structure_factor_linecut_vac_0.0_0.1_0.2_T123.png"))
    println("Created a single plot in 'results/structure_factor_linecut_vac_0.0_0.1_0.2_T123.png'")
end
