import taichi as ti
import numpy as np
ti.init(arch=ti.gpu)

# material id
WATER = 0
JELLY = 1
SNOW  = 2

n_particles, n_grid = 9000, 128
dx, inv_dx = 1 / n_grid, float(n_grid)
dt, p_vol, p_rho = 1e-4, (dx * 0.5)**2, 1
p_mass = p_vol * p_rho
E, nu = 0.1e4, 0.2
mu_0 = E / (2 * (1 + nu))
lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))

x = ti.Vector.field(2, dtype=float, shape=n_particles)
v = ti.Vector.field(2, dtype=float, shape=n_particles)
C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)
F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)
material = ti.field(dtype=int, shape=n_particles)
Jp = ti.field(dtype=float, shape=n_particles)

grid_v = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))


@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]
        h = ti.max(0.1, ti.min(5, ti.exp(10 * (1.0 - Jp[p]))))
        if material[p] == JELLY:
            h = 0.3
        mu, la = mu_0 * h, lambda_0 * h
        if material[p] == WATER:
            mu = 0.0
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            if material[p] == SNOW:
                new_sig = ti.min(ti.max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        if material[p] == WATER:
            new_F = ti.Matrix.identity(float, 2)
            new_F[0, 0] = J
            F[p] = new_F
        elif material[p] == SNOW:
            F[p] = U @ sig @ V.transpose()
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, 2) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] = (1 / grid_m[i, j]) * grid_v[i, j]
            grid_v[i, j][1] -= dt * 50
            if i < 3 and grid_v[i, j][0] < 0:
                grid_v[i, j][0] = 0
            if i > n_grid - 3 and grid_v[i, j][0] > 0:
                grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0:
                grid_v[i, j][1] = 0
            if j > n_grid - 3 and grid_v[i, j][1] > 0:
                grid_v[i, j][1] = 0
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]


@ti.kernel
def reset():
    group_size = n_particles // 3
    for i in range(n_particles):
        x[i] = [ti.random() * 0.2 + 0.3 + 0.10 * (i // group_size),
                 ti.random() * 0.2 + 0.05 + 0.32 * (i // group_size)]
        material[i] = i // group_size
        v[i] = [0, 0]
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        Jp[i] = 1
        C[i] = ti.Matrix.zero(float, 2, 2)


reset()
gui = ti.GUI("MLS-MPM", res=512, background_color=0x112F41)
while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == 'r':
            reset()
        elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            break
    for s in range(50):
        substep()
    colors = np.array([0x068587, 0xED553B, 0xEEEEF0], dtype=np.uint32)
    gui.circles(x.to_numpy(), radius=1.5,
                color=colors[material.to_numpy()])
    gui.show()
