"""
    body_displacement(x)

Extract the linear and angular displacement of the body frame from the state vector.
"""
function body_displacement(x)

    ub = SVector(x[1], x[2], x[3])
    θb = SVector(x[4], x[5], x[6])

    return ub, θb
end

"""
    body_velocity(x)

Extract the linear and angular velocity of the body frame from the state vector.
"""
function body_velocity(x)

    vb = SVector(x[7], x[8], x[9])
    ωb = SVector(x[10], x[11], x[12])

    return vb, ωb
end

"""
    body_velocity(system, x=system.x; linear_velocity=zeros(3), angular_velocity=zeros(3))

Extract the linear and angular velocity of the body frame from the state vector or 
prescribed velocitys.
"""
function body_velocity(system::AbstractSystem, x=system.x; 
    linear_velocity=(@SVector zeros(3)), 
    angular_velocity=(@SVector zeros(3))
    )

    return body_velocity(x, system.indices.icol_indices, linear_velocity, angular_velocity)
end

"""
    body_velocity(x, icol, vb=zeros(3), ωb=zeros(3))

Extract the linear and angular velocity of the body frame from the state vector or 
prescribed velocities.
"""
function body_velocity(x, icol, vb=(@SVector zeros(3)), ωb=(@SVector zeros(3)))

    vb = SVector(
        iszero(icol[1]) ? vb[1] : x[7],
        iszero(icol[2]) ? vb[2] : x[8],
        iszero(icol[3]) ? vb[3] : x[9],
    )

    ωb = SVector(
        iszero(icol[4]) ? ωb[1] : x[10],
        iszero(icol[5]) ? ωb[2] : x[11],
        iszero(icol[6]) ? ωb[3] : x[12],
    )

    return vb, ωb
end

"""
    body_acceleration(system, x=system.x; linear_acceleration=zeros(3), angular_acceleration=zeros(3))

Extract the linear and angular acceleration of the body frame from the state vector or 
prescribed accelerations.
"""
function body_acceleration(system::AbstractSystem, x=system.x; 
    linear_acceleration=(@SVector zeros(3)), 
    angular_acceleration=(@SVector zeros(3))
    )

    return body_acceleration(x, system.indices.icol_indices, linear_acceleration, angular_acceleration)
end

function body_acceleration(x, icol, ab=(@SVector zeros(3)), αb=(@SVector zeros(3)))

    ab = SVector(
        iszero(icol[1]) ? ab[1] : x[icol[1]],
        iszero(icol[2]) ? ab[2] : x[icol[2]],
        iszero(icol[3]) ? ab[3] : x[icol[3]],
    )

    αb = SVector(
        iszero(icol[4]) ? αb[1] : x[icol[4]],
        iszero(icol[5]) ? αb[2] : x[icol[5]],
        iszero(icol[6]) ? αb[3] : x[icol[6]],
    )

    return ab, αb
end

"""
    steady_body_residual!(resid, x, indices, ub_p, θb_p, vb_p, ωb_p)

Calculate and insert the residual entries corresponding to the motion of the body frame 
for a steady state analysis into the system residual vector.
"""
function steady_body_residual!(resid, x, indices, ub_p, θb_p, vb_p, ωb_p)
    
    # extract body states
    ub, θb = body_displacement(x)
    vb, ωb = body_velocity(x)

    # construct residuals (all rigid body states are prescribed)
    ru = ub - ub_p
    rθ = θb - θb_p
    rv = vb - vb_p
    rω = ωb - ωb_p

    # insert residuals into the residual vector
    resid[1:3] = ru
    resid[4:6] = rθ
    resid[7:9] = rv
    resid[10:12] = rω
    
    return resid
end

const initial_body_residual! = steady_body_residual!

"""
    newmark_body_residual!(resid, x, indices, ubdot_init, θbdot_init, vbdot_init, ωbdot_init, dt)

Calculate and insert the residual entries corresponding to the motion of the body frame 
for a newmark-scheme time marching analysis into the system residual vector.
"""
function newmark_body_residual!(resid, x, indices, ab_p, αb_p, ubdot_init, θbdot_init, 
    vbdot_init, ωbdot_init, dt)
    
    # extract body states
    ub, θb = body_displacement(x)
    vb, ωb = body_velocity(x)
    ab, αb = body_acceleration(x, indices.icol_body, ab_p, αb_p)

    # extract body rates
    ubdot = 2/dt*ub - SVector{3}(ubdot_init)
    θbdot = 2/dt*θb - SVector{3}(θbdot_init)
    vbdot = 2/dt*vb - SVector{3}(vbdot_init)
    ωbdot = 2/dt*ωb - SVector{3}(ωbdot_init)

    # rotation parameter matrices
    C = get_C(θb)
    Qinv = get_Qinv(θb)

    # construct residuals
    ru = vb - ubdot
    rθ = Qinv*C*ωb - θbdot
    rv = ab - vbdot
    rω = αb - ωbdot

    # insert residuals into the residual vector
    resid[1:3] = ru
    resid[4:6] = rθ
    resid[7:9] = rv
    resid[10:12] = rω
    
    return resid
end

"""
    dynamic_body_residual!(resid, dx, x, indices, vb_p, ωb_p)

Calculate and insert the residual entries corresponding to the motion of the body frame 
for a dynamic analysis into the system residual vector.
"""
function dynamic_body_residual!(resid, dx, x, indices, vb_p, ωb_p)
    
    # extract body states
    ub, θb = body_displacement(x)
    vb, ωb = body_velocity(x)
    ab, αb = body_acceleration(x, indices.icol_body)

    # extract body rates
    ubdot, θbdot = body_displacement(dx)
    vbdot, ωbdot = body_velocity(dx)

    # rotation parameter matrices
    C = get_C(θb)
    Qinv = get_Qinv(θb)

    # construct residuals
    ru = vb - ubdot
    rθ = Qinv*C*ωb - θbdot
    rv = ab - vbdot
    rω = αb - ωbdot

    # overwrite residuals for velocities (if applicable)
    for i = 1:3
        # prescribed velocities
        if iszero(icol[i])
            rv = setindex(rv, vb[i] - vb_p[i], i)
        end
        # prescribed accelerations
        if iszero(icol[3+i])
            rω = setindex(rω, ωb[i] - ωb_p[i], i)
        end
    end

    # insert residuals into the residual vector
    resid[1:3] = ru
    resid[4:6] = rθ
    resid[7:9] = rv
    resid[10:12] = rω
    
    return resid
end

"""
    expanded_steady_body_residual!(resid, x, indices, ub_p, θb_p, vb_p, ωb_p, ab_p, αb_p)

Calculate and insert the residual entries corresponding to the prescribed body frame motion
of a constant mass matrix system into the system residual vector.
"""
const expanded_steady_body_residual! = steady_body_residual!

"""
    expanded_dynamic_body_residual!(resid, x, indices, ub_p, θb_p, vb_p, ωb_p, ab_p, αb_p)

Calculate and insert the residual entries corresponding to the motion of the body frame 
for a constant mass matrix system into the system residual vector.
"""
const expanded_dynamic_body_residual! = dynamic_body_residual!

"""
    steady_body_jacobian!(jacob, x, indices, ub_p, θb_p, vb_p, ωb_p)

Calculate and insert the jacobian entries corresponding to the motion of the body frame 
for a steady state analysis into the system jacobian matrix.
"""
function steady_body_jacobian!(jacob, x, indices, ub_p, θb_p, vb_p, ωb_p)

    for i = 1:12
        jacob[i,i] = 1 
    end
    
    return jacob
end

"""
    initial_body_jacobian!(jacob, x, indices, ub_p, θb_p, vb_p, ωb_p)

Calculate and insert the jacobian entries corresponding to the motion of the body frame 
for the initialization of a time domain analysis into the system jacobian vector.
"""
const initial_body_jacobian! = steady_body_jacobian!

"""
    newmark_body_jacobian!(jacob, x, indices, ab_p, αb_p, ubdot_init, θbdot_init, 
        vbdot_init, ωbdot_init)

Calculate and insert the jacobian entries corresponding to the motion of the body frame 
for a steady state analysis into the system jacobian matrix.
"""
function newmark_body_jacobian!(jacob, x, indices, vb_p, ωb_p, ubdot_init, θbdot_init, 
    vbdot_init, ωbdot_init, dt)

    # extract body states
    icol = indices.icol_body
    ub, θb = body_displacement(x)
    vb, ωb = body_velocity(x)
    ab, αb = body_acceleration(x, icol, ab_p, αb_p)

    # extract body rates
    ubdot_ub = 2/dt*I3
    θbdot_θb = 2/dt*I3
    vbdot_vb = 2/dt*I3
    ωbdot_ωb = 2/dt*I3

    # rotation parameter matrices
    C = get_C(θb)
    Qinv = get_Qinv(θb)
    C_θ1, C_θ2, C_θ3 = get_C_θ(C, θb)
    Qinv_θ1, Qinv_θ2, Qinv_θ3 = get_Qinv_θ(θb)

    # construct residuals
    ru_ub = -ubdot_ub
    ru_vb = I3
    
    rθ_θb = mul3(Qinv_θ1, Qinv_θ2, Qinv_θ3, C*ωb) + Qinv*mul3(C_θ1, C_θ2, C_θ3, ωb) - θbdot_θb
    rθ_ωb = Qinv*C

    rv_vb = -vbdot_vb
    rv_ab = I3

    rω_ωb = -ωbdot_ωb
    rω_αb = I3

    # insert jacobian entries into the jacobian matrix

    jacob[1:3,1:3] = ru_ub
    jacob[1:3,7:9] = ru_vb

    jacob[4:6,4:6] = rθ_θb
    jacob[4:6,10:12] = rθ_ωb

    jacob[7:9,7:9] = rv_vb
    for i = 1:3
        if !iszero(icol[i])
            @views jacob[7:9, icol[i]] .= rv_ab[:,i]
        end
    end

    jacob[10:12,10:12] = rω_ωb
    for i = 4:6
        if !iszero(icol[i])
            @views jacob[10:12, icol[i]] .= rω_αb[:,i-3]
        end
    end

    return jacob
end

function dynamic_body_jacobian!(jacob, dx, x, indices, ab_p, αb_p)

    # extract body states
    icol = indices.icol_body
    ub, θb = body_displacement(x)
    vb, ωb = body_velocity(x)
    ab, αb = body_acceleration(x, icol, ab_p, αb_p)

    # rotation parameter matrices
    C = get_C(θb)
    Qinv = get_Qinv(θb)
    C_θ1, C_θ2, C_θ3 = get_C_θ(C, θb)
    Qinv_θ1, Qinv_θ2, Qinv_θ3 = get_Qinv_θ(θb)

    # construct residuals
    ru_vb = I3
    
    rθ_θb = mul3(Qinv_θ1, Qinv_θ2, Qinv_θ3, C*ωb) + Qinv*mul3(C_θ1, C_θ2, C_θ3, ωb)
    rθ_ωb = Qinv*C

    rv_ab = I3

    rω_αb = I3

    ra_ab = I3
    rα_αb = I3

    # insert jacobian entries into the jacobian matrix

    jacob[1:3,7:9] = ru_vb

    jacob[4:6,4:6] = rθ_θb
    jacob[4:6,10:12] = rθ_ωb

    jacob[7:9,13:15] = rv_ab

    jacob[10:12,16:18] = rω_αb

    jacob[13:15,13:15] = ra_ab

    jacob[16:18,16:18] = rα_αb

    # add influence of prescribed acceleration state variables
    icol_accel[1] > 0 && setindex!(jacob, -1, 13, icol_accel[1])
    icol_accel[2] > 0 && setindex!(jacob, -1, 14, icol_accel[2])
    icol_accel[3] > 0 && setindex!(jacob, -1, 15, icol_accel[3])

    icol_accel[4] > 0 && setindex!(jacob, -1, 16, icol_accel[4])
    icol_accel[5] > 0 && setindex!(jacob, -1, 17, icol_accel[5])
    icol_accel[6] > 0 && setindex!(jacob, -1, 18, icol_accel[6])

    return jacob
end

const expanded_steady_body_jacobian! = steady_body_jacobian!

function expanded_dynamic_body_jacobian!(jacob, x, icol_accel, ab_p, αb_p)
    
    return dynamic_body_jacobian!(jacob, x, x, icol_accel, ab_p, αb_p)
end

function mass_matrix_body_jacobian!(jacob)
    
    # define body residuals
    jacob[1:3,1:3] = -I3
    jacob[4:6,4:6] = -I3
    jacob[7:9,7:9] = -I3
    jacob[10:12,10:12] = -I3
    
    return jacob
end