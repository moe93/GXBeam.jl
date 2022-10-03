"""
    BodyState

Holds the state variables for the body frame

# Fields:
 - `u`: Linear deflection
 - `theta`: Angular deflection (Wiener-Milenkovic parameters)
 - `V`: Linear velocity
 - `Omega`: Angular velocity
"""
struct BodyState{TF}
    u::SVector{3, TF}
    theta::SVector{3, TF}
    V::SVector{3, TF}
    Omega::SVector{3, TF}
end

"""
    PointState

Holds the state variables for a point

# Fields:
 - `u`: Linear deflection
 - `theta`: Angular deflection (Wiener-Milenkovic parameters)
 - `V`: Linear velocity
 - `Omega`: Angular velocity
 - `F`: External forces
 - `M`: External moments
"""
struct PointState{TF}
    u::SVector{3, TF}
    theta::SVector{3, TF}
    V::SVector{3, TF}
    Omega::SVector{3, TF}
    F::SVector{3, TF}
    M::SVector{3, TF}
end

"""
    ElementState

Holds the state variables for an element

# Fields:
 - `u`: Linear deflection
 - `theta`: Angular deflection (Wiener-Milenkovic parameters)
 - `V`: Linear velocity
 - `Omega`: Angular velocity
 - `Fi`: Internal forces
 - `Mi`: Internal moments
"""
struct ElementState{TF}
    u::SVector{3, TF}
    theta::SVector{3, TF}
    V::SVector{3, TF}
    Omega::SVector{3, TF}
    Fi::SVector{3, TF}
    Mi::SVector{3, TF}
end

"""
    AssemblyState{TF, TP<:AbstractVector{PointState{TF}}, TE<:AbstractVector{ElementState{TF}}}

Struct for storing state variables for the points and elements in an assembly.

# Fields:
 - `points::TP`: Array of [`PointState`](@ref) for each point in the assembly
 - `elements::TE`: Array of [`ElementState`](@ref) for each element in the assembly
"""
struct AssemblyState{TF, TB<:BodyState{TF}, TP<:AbstractVector{PointState{TF}}, TE<:AbstractVector{ElementState{TF}}}
    body::TB
    points::TP
    elements::TE
end
Base.eltype(::AssemblyState{TF, TB, TP, TE}) where {TF, TB, TP, TE} = TF

"""
    AssemblyState(system, assembly, x = system.x; prescribed_conditions = Dict())

Post-process the system state given the solution vector `x`.  Return an object
of type `AssemblyState` that defines the state of the assembly for the time step.

If `prescribed_conditions` is not provided, all point state variables are assumed
to be displacements/rotations, rather than their actual identities as used in the
analysis.
"""
function AssemblyState(system, assembly, x = system.x; prescribed_conditions = Dict{Int,PrescribedConditions{Float64}}())

    body = extract_body_states(system, x)

    points = extract_point_states(system, assembly, x; prescribed_conditions)

    elements = extract_element_states(system, assembly, x; prescribed_conditions)

    return AssemblyState{eltype(x), typeof(body), typeof(points), typeof(elements)}(body, points, elements)
end

"""
    extract_body_states(system, assembly, ipoint, x = system.x;
        prescribed_conditions = Dict{Int,PrescribedConditions{Float64}}())

Return the state variables corresponding to the motion of the body frame, given the 
solution vector `x`.
"""
extract_body_states

function extract_body_states(system::StaticSystem, x = system.x)

    u = @SVector zeros(3)
    θ = @SVector zeros(3)
    V = @SVector zeros(3)
    Ω = @SVector zeros(3)

    return BodyState(u, θ, V, Ω)
end

function extract_body_states(system::Union{DynamicSystem,ExpandedSystem}, x = system.x)

    u, θ = body_displacement(x)
    V, Ω = body_velocity(x)

    return BodyState(u, θ, V, Ω)
end

"""
    extract_point_state(system, assembly, ipoint, x = system.x;
        prescribed_conditions = Dict{Int,PrescribedConditions{Float64}}())

Return the state variables corresponding to point `ipoint` (see [`PointState`](@ref))
given the solution vector `x`.

If `prescribed_conditions` is not provided, all point state variables are assumed
to be displacements/rotations, rather than their actual identities as used in the
analysis.
"""
extract_point_state

function extract_point_state(system::StaticSystem, assembly, ipoint, x = system.x;
    prescribed_conditions = Dict{Int,PrescribedConditions{Float64}}())

    @unpack force_scaling, indices, t = system

    # check state variable length
    @assert length(x) == length(system.x) "state vector length does not match with the provided system"

    # get current prescribed conditions
    pc = typeof(prescribed_conditions) <: AbstractDict ? prescribed_conditions : prescribed_conditions(t)

    # extract point state variables
    u, θ = point_displacement(x, ipoint, indices.icol_point, pc)
    F, M = point_loads(x, ipoint, indices.icol_point, force_scaling, pc)
    V, Ω = zero(u), zero(θ)
    
    # convert rotation parameter to Wiener-Milenkovic parameters
    scaling = rotation_parameter_scaling(θ)
    θ *= scaling

    # promote all variables to the same type
    u, θ, V, Ω, F, M = promote(u, θ, V, Ω, F, M)

    return PointState(u, θ, V, Ω, F, M)
end

function extract_point_state(system::DynamicSystem, assembly, ipoint, x = system.x;
    prescribed_conditions = Dict{Int,PrescribedConditions{Float64}}())

    @unpack force_scaling, indices, t = system

    # check state variable length
    @assert length(x) == length(system.x) "state vector length does not match with the provided system"

    # get current prescribed conditions
    pc = typeof(prescribed_conditions) <: AbstractDict ? prescribed_conditions : prescribed_conditions(t)

    # extract point state variables
    u, θ = point_displacement(x, ipoint, indices.icol_point, pc)
    F, M = point_loads(x, ipoint, indices.icol_point, force_scaling, pc)
    V, Ω = point_velocities(x, ipoint, indices.icol_point)

    # convert rotation parameter to Wiener-Milenkovic parameters
    scaling = rotation_parameter_scaling(θ)
    θ *= scaling

    # promote all variables to the same type
    u, θ, V, Ω, F, M = promote(u, θ, V, Ω, F, M)

    return PointState(u, θ, V, Ω, F, M)
end

function extract_point_state(system::ExpandedSystem, assembly, ipoint, x = system.x;
    prescribed_conditions = Dict{Int,PrescribedConditions{Float64}}())

    @unpack force_scaling, indices, t = system

    # check state variable length
    @assert length(x) == length(system.x) "state vector length does not match with the provided system"

    # get current prescribed conditions
    pc = typeof(prescribed_conditions) <: AbstractDict ? prescribed_conditions : prescribed_conditions(t)

    # extract point state variables
    u, θ = point_displacement(x, ipoint, indices.icol_point, pc)
    F, M = point_loads(x, ipoint, indices.icol_point, force_scaling, pc)
    V, Ω = point_velocities(x, ipoint, indices.icol_point)

    # rotate linear and angular velocities into the body frame
    C = get_C(θ)
    V = C'*V
    Ω = C'*Ω

    # convert rotation parameters to Wiener-Milenkovic parameters
    scaling = rotation_parameter_scaling(θ)
    θ *= scaling

    # promote all variables to the same type
    u, θ, V, Ω, F, M = promote(u, θ, V, Ω, F, M)

    return PointState(u, θ, V, Ω, F, M)
end

"""
    extract_point_states(system, assembly, x = system.x;
        prescribed_conditions = Dict{Int,PrescribedConditions{Float64}}())

Return the state variables corresponding to each point (see [`PointState`](@ref))
given the solution vector `x`.

If `prescribed_conditions` is not provided, all point state variables are assumed
to be displacements/rotations, rather than their actual identities as used in the
analysis.
"""
function extract_point_states(system, assembly, x = system.x; kwargs...)
    
    return [extract_point_state(system, assembly, ipoint, x; kwargs...) for ipoint in eachindex(assembly.points)]
end

"""
    extract_point_states!(points, system, assembly, x = system.x;
        prescribed_conditions = Dict{Int,PrescribedConditions{Float64}}())

Pre-allocated version of [`extract_point_states`](@ref)
"""
function extract_point_states!(points, system, assembly, x = system.x; kwargs...)

    for ipoint in eachindex(points)
        points[ipoint] = extract_point_state(system, assembly, ipoint, x; kwargs...)
    end

    return points
end

"""
    extract_element_state(system, assembly, ielem, x = system.x;
        prescribed_conditions = Dict{Int,PrescribedConditions{Float64}}())

Return the state variables corresponding to element `ielem` (see [`ElementState`](@ref))
given the solution vector `x`.
"""
extract_element_state

function extract_element_state(system::StaticSystem, assembly, ielem, x = system.x; 
    prescribed_conditions = Dict{Int,PrescribedConditions{Float64}}())

    # system variables
    @unpack force_scaling, indices, t = system

    # check state variable length
    @assert length(x) == length(system.x) "state vector length does not match with the provided system"

    # current prescribed conditions
    pc = typeof(prescribed_conditions) <: AbstractDict ? prescribed_conditions : prescribed_conditions(t)

    # linear and angular displacement
    u1, θ1 = point_displacement(x, assembly.start[ielem], indices.icol_point, pc)
    u2, θ2 = point_displacement(x, assembly.stop[ielem], indices.icol_point, pc)
    u = (u1 + u2)/2
    θ = (θ1 + θ2)/2

    # internal forces and moments
    F, M = element_loads(x, ielem, indices.icol_elem, force_scaling)

    # linear and angular velocity
    V = zero(u)
    Ω = zero(θ)

    # convert rotation parameter to Wiener-Milenkovic parameters
    scaling = rotation_parameter_scaling(θ)
    θ *= scaling

    # promote all variables to the same type
    u, θ, V, Ω, F, M = promote(u, θ, V, Ω, F, M)

    return ElementState(u, θ, V, Ω, F, M)
end

function extract_element_state(system::DynamicSystem, assembly, ielem, x = system.x; 
    prescribed_conditions = Dict{Int,PrescribedConditions{Float64}}())

    # system variables
    @unpack force_scaling, indices, t = system

    # check state variable length
    @assert length(x) == length(system.x) "state vector length does not match with the provided system"

    # current prescribed conditions
    pc = typeof(prescribed_conditions) <: AbstractDict ? prescribed_conditions : prescribed_conditions(t)

    # linear and angular displacement
    u1, θ1 = point_displacement(x, assembly.start[ielem], indices.icol_point, pc)
    u2, θ2 = point_displacement(x, assembly.stop[ielem], indices.icol_point, pc)
    u = (u1 + u2)/2
    θ = (θ1 + θ2)/2

    # internal forces and moments
    F, M = element_loads(x, ielem, indices.icol_elem, force_scaling)

    # linear and angular velocity
    V1, Ω1 = point_velocities(x, assembly.start[ielem], indices.icol_point)
    V2, Ω2 = point_velocities(x, assembly.stop[ielem], indices.icol_point)
    V = (V1 + V2)/2
    Ω = (Ω1 + Ω2)/2

    # convert rotation parameter to Wiener-Milenkovic parameters
    scaling = rotation_parameter_scaling(θ)
    θ *= scaling

    # promote all variables to the same type
    u, θ, V, Ω, F, M = promote(u, θ, V, Ω, F, M)

    return ElementState(u, θ, V, Ω, F, M)
end

function extract_element_state(system::ExpandedSystem, assembly, ielem, x = system.x; 
    prescribed_conditions = Dict{Int,PrescribedConditions{Float64}}())

    # system variables
    @unpack force_scaling, indices, t = system

    # check state variable length
    @assert length(x) == length(system.x) "state vector length does not match with the provided system"

    # current prescribed conditions
    pc = typeof(prescribed_conditions) <: AbstractDict ? prescribed_conditions : prescribed_conditions(t)

    # linear and angular displacement
    u1, θ1 = point_displacement(x, assembly.start[ielem], indices.icol_point, pc)
    u2, θ2 = point_displacement(x, assembly.stop[ielem], indices.icol_point, pc)
    u = (u1 + u2)/2
    θ = (θ1 + θ2)/2

    # element forces and moments
    F1, M1, F2, M2 = expanded_element_loads(x, ielem, indices.icol_elem, force_scaling)
    F = (F1 + F2)/2
    M = (M1 + M2)/2

    # linear and angular velocity
    V, Ω = expanded_element_velocities(x, ielem, indices.icol_elem)

    # rotate linear and angular velocites into the body frame
    C = get_C(θ)
    Cab = assembly.elements[ielem].Cab
    CtCab = C'*Cab
    V = CtCab*V
    Ω = CtCab*Ω

    # convert rotation parameter to Wiener-Milenkovic parameters
    scaling = rotation_parameter_scaling(θ)
    θ *= scaling

    # promote all variables to the same type
    u, θ, V, Ω, F, M = promote(u, θ, V, Ω, F, M)

    return ElementState(u, θ, V, Ω, F, M)
end

"""
    extract_element_states(system, assembly, x = system.x;
        prescribed_conditions = Dict{Int,PrescribedConditions{Float64}}())

Return the state variables corresponding to each element (see [`ElementState`](@ref))
given the solution vector `x`.
"""
function extract_element_states(system, assembly, x = system.x; kwargs...)

    return [extract_element_state(system, assembly, ielem, x; kwargs...) for ielem in eachindex(assembly.elements)]
end

"""
    extract_element_states!(elements, system, assembly, x = system.x;
        prescribed_conditions = Dict{Int,PrescribedConditions{Float64}}())

Pre-allocated version of [`extract_element_states`](@ref)
"""
function extract_element_states!(elements, system, assembly, x = system.x; kwargs...)

    for ielem in eachindex(elements)
        elements[ielem] = extract_element_state(system, assembly, ielem, x; kwargs...)
    end

    return elements
end