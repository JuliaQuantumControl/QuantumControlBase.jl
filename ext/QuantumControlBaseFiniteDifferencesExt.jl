module QuantumControlBaseFiniteDifferencesExt

using QuantumControlBase: _default_chi_via
using LinearAlgebra: axpby!
import FiniteDifferences

import QuantumControlBase: make_automatic_chi, make_automatic_grad_J_a


function make_automatic_chi(
    J_T,
    trajectories,
    ::Val{:FiniteDifferences};
    via=_default_chi_via(trajectories)
)

    function fdm_chi_via_phi!(χ, ϕ, trajectories; tau=nothing, τ=tau)
        function _J_T(Ψ...)
            -J_T(Ψ, trajectories)
        end
        fdm = FiniteDifferences.central_fdm(5, 1)
        ∇J = FiniteDifferences.grad(fdm, _J_T, ϕ...)
        for (k, ∇Jₖ) ∈ enumerate(∇J)
            ∇Jₖ = convert(typeof(χ[k]), ∇Jₖ)
            # |χₖ⟩ = ½ |∇Jₖ⟩  # ½ corrects for gradient vs Wirtinger deriv
            axpby!(0.5, ∇Jₖ, false, χ[k])
        end
    end

    function fdm_chi_via_tau!(χ, ϕ, trajectories; tau=nothing, τ=tau)
        if isnothing(τ)
            msg = "chi! returned by `make_chi` with `via=:tau` requires keyword argument tau/τ"
            throw(ArgumentError(msg))
        end
        function _J_T(τ...)
            -J_T(ϕ, trajectories; τ=τ)
        end
        fdm = FiniteDifferences.central_fdm(5, 1)
        ∇J = FiniteDifferences.grad(fdm, _J_T, τ...)
        for (k, ∇Jₖ) ∈ enumerate(∇J)
            ∂J╱∂τ̄ₖ = 0.5 * ∇Jₖ  # ½ corrects for gradient vs Wirtinger deriv
            # |χₖ⟩ = (∂J/∂τ̄ₖ) |ϕₖ⟩
            axpby!(∂J╱∂τ̄ₖ, trajectories[k].target_state, false, χ[k])
        end
    end

    if via ≡ :phi
        return fdm_chi_via_phi!
    elseif via ≡ :tau
        ϕ_tgt = [traj.target_state for traj in trajectories]
        if any(isnothing.(ϕ_tgt))
            error("`via=:tau` requires that all trajectories define a `target_state`")
        end
        τ_tgt = ComplexF64[1.0 for traj in trajectories]
        if abs(J_T(ϕ_tgt, trajectories) - J_T(nothing, trajectories; τ=τ_tgt)) > 1e-12
            error(
                "`via=:tau` in `make_chi` requires that `J_T`=$(repr(J_T)) can be evaluated solely via `τ`"
            )
        end
        return fdm_chi_via_tau!
    else
        msg = "`via` must be either `:phi` or `:tau`, not $(repr(via))"
        throw(ArgumentError(msg))
    end

end


function make_automatic_grad_J_a(J_a, tlist, ::Val{:FiniteDifferences})
    function automatic_grad_J_a!(∇J_a, pulsevals, tlist)
        func = pulsevals -> J_a(pulsevals, tlist)
        fdm = FiniteDifferences.central_fdm(5, 1)
        ∇J_a_fdm = FiniteDifferences.grad(fdm, func, pulsevals)[1]
        copyto!(∇J_a, ∇J_a_fdm)
    end
    return automatic_grad_J_a!
end


end
