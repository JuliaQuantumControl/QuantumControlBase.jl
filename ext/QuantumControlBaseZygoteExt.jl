module QuantumControlBaseZygoteExt

using QuantumControlBase: _default_chi_via
using LinearAlgebra: axpby!
import Zygote

import QuantumControlBase: make_automatic_chi, make_automatic_grad_J_a


function make_automatic_chi(
    J_T,
    trajectories,
    ::Val{:Zygote};
    via=_default_chi_via(trajectories)
)

    function zygote_chi_via_phi!(χ, ϕ, trajectories; tau=nothing, τ=tau)
        function _J_T(Ψ...)
            -J_T(Ψ, trajectories)
        end
        ∇J = Zygote.gradient(_J_T, ϕ...)
        for (k, ∇Jₖ) ∈ enumerate(∇J)
            ∇Jₖ = convert(typeof(χ[k]), ∇Jₖ)
            # |χₖ⟩ = ½ |∇Jₖ⟩  # ½ corrects for gradient vs Wirtinger deriv
            axpby!(0.5, ∇Jₖ, false, χ[k])
        end
    end

    function zygote_chi_via_tau!(χ, ϕ, trajectories; tau=nothing, τ=tau)
        if isnothing(τ)
            msg = "chi! returned by `make_chi` with `via=:tau` requires keyword argument tau/τ"
            throw(ArgumentError(msg))
        end
        function _J_T(τ...)
            -J_T(ϕ, trajectories; τ=τ)
        end
        ∇J = Zygote.gradient(_J_T, τ...)
        for (k, ∇Jₖ) ∈ enumerate(∇J)
            ∂J╱∂τ̄ₖ = 0.5 * ∇Jₖ  # ½ corrects for gradient vs Wirtinger deriv
            # |χₖ⟩ = (∂J/∂τ̄ₖ) |ϕₖ⟩
            axpby!(∂J╱∂τ̄ₖ, trajectories[k].target_state, false, χ[k])
        end
    end

    if via ≡ :phi
        return zygote_chi_via_phi!
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
        return zygote_chi_via_tau!
    else
        msg = "`via` must be either `:phi` or `:tau`, not $(repr(via))"
        throw(ArgumentError(msg))
    end

end


function make_automatic_grad_J_a(J_a, tlist, ::Val{:Zygote})
    function automatic_grad_J_a!(∇J_a, pulsevals, tlist)
        func = pulsevals -> J_a(pulsevals, tlist)
        ∇J_a_zygote = Zygote.gradient(func, pulsevals)[1]
        copyto!(∇J_a, ∇J_a_zygote)
    end
    return automatic_grad_J_a!
end

end
