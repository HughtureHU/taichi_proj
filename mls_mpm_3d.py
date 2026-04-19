import taichi as ti

ti.init(arch=ti.gpu)

# ============================================================
# 3D MPM Snow Simulation with Realistic Rendering
# 基于 Stomakhin et al. 2013 "A material point method for snow simulation"
# ============================================================

dim = 3
n_particles = 40000
n_grid = 64
dx = 1.0 / n_grid
inv_dx = float(n_grid)
dt = 1e-4

p_vol = (dx * 0.5) ** dim
p_rho = 1.0
p_mass = p_vol * p_rho

# 雪的本构参数
E = 1.0e5                                 # 杨氏模量
nu = 0.2                                  # 泊松比
mu_0 = E / (2 * (1 + nu))
lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))

# 雪的塑性参数
theta_c = 2.5e-2                          # 临界压缩
theta_s = 7.5e-3                          # 临界拉伸
xi = 10.0                                 # 硬化系数
gravity = 50.0

# 粒子场
x = ti.Vector.field(dim, dtype=float, shape=n_particles)
v = ti.Vector.field(dim, dtype=float, shape=n_particles)
C = ti.Matrix.field(dim, dim, dtype=float, shape=n_particles)
F = ti.Matrix.field(dim, dim, dtype=float, shape=n_particles)
Jp = ti.field(dtype=float, shape=n_particles)

# 网格场
grid_v = ti.Vector.field(dim, dtype=float, shape=(n_grid,) * dim)
grid_m = ti.field(dtype=float, shape=(n_grid,) * dim)

# 渲染用场
particle_colors = ti.Vector.field(4, dtype=float, shape=n_particles)
base_color = ti.Vector.field(4, dtype=float, shape=n_particles)


@ti.kernel
def substep():
    # 清空网格
    for I in ti.grouped(grid_m):
        grid_v[I] = ti.Vector.zero(float, dim)
        grid_m[I] = 0.0

    # Particle-to-Grid (P2G)
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2,
             0.75 - (fx - 1.0) ** 2,
             0.5 * (fx - 0.5) ** 2]

        # 更新形变梯度
        F[p] = (ti.Matrix.identity(float, dim) + dt * C[p]) @ F[p]

        # 硬化：压得越实的雪越硬
        h = ti.exp(xi * (1.0 - Jp[p]))
        mu = mu_0 * h
        la = lambda_0 * h

        # SVD 塑性投影
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(dim)):
            new_sig = ti.min(ti.max(sig[d, d], 1 - theta_c), 1 + theta_s)
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        F[p] = U @ sig @ V.transpose()

        # Cauchy 应力
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + \
                 ti.Matrix.identity(float, dim) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]

        # 散播到 27 个相邻网格点
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1] * w[k][2]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass

    # 网格运算：动量转速度 + 重力 + 边界
    for I in ti.grouped(grid_m):
        if grid_m[I] > 0:
            grid_v[I] = grid_v[I] / grid_m[I]
            grid_v[I][1] -= dt * gravity
            for d in ti.static(range(dim)):
                if I[d] < 3 and grid_v[I][d] < 0:
                    grid_v[I][d] = 0
                if I[d] > n_grid - 3 and grid_v[I][d] > 0:
                    grid_v[I][d] = 0

    # Grid-to-Particle (G2P)
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2,
             0.75 - (fx - 1.0) ** 2,
             0.5 * (fx - 0.5) ** 2]

        new_v = ti.Vector.zero(float, dim)
        new_C = ti.Matrix.zero(float, dim, dim)

        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = offset.cast(float) - fx
            g_v = grid_v[base + offset]
            weight = w[i][0] * w[j][1] * w[k][2]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)

        v[p] = new_v
        C[p] = new_C
        x[p] += dt * v[p]


@ti.kernel
def init():
    group_size = n_particles // 3
    for i in range(n_particles):
        g = i // group_size
        cx, cy, cz = 0.0, 0.0, 0.0
        if g == 0:
            cx, cy, cz = 0.22, 0.78, 0.42   # 最上面那坨
        elif g == 1:
            cx, cy, cz = 0.46, 0.55, 0.32   # 中间
        else:
            cx, cy, cz = 0.65, 0.35, 0.52   # 最下面

        x[i] = ti.Vector([cx + ti.random() * 0.18,
                          cy + ti.random() * 0.18,
                          cz + ti.random() * 0.18])
        v[i] = ti.Vector([0.0, 0.0, 0.0])
        F[i] = ti.Matrix.identity(float, dim)
        Jp[i] = 1.0
        C[i] = ti.Matrix.zero(float, dim, dim)

        # 每颗粒子的底色做轻微扰动，避免一片"塑料白"
        tint = ti.random() * 0.08
        base_color[i] = ti.Vector([0.94 - tint * 0.5,
                                   0.97 - tint * 0.2,
                                   1.00,
                                   1.0])


@ti.kernel
def update_colors():
    # 被压实的雪 -> 偏冷偏暗（模拟 AO + 冰的蓝色吸收）
    for p in range(n_particles):
        compression = ti.max(0.0, 1.0 - Jp[p])
        shade = ti.min(1.0, compression * 1.5)
        particle_colors[p][0] = base_color[p][0] * (1.0 - shade * 0.22)
        particle_colors[p][1] = base_color[p][1] * (1.0 - shade * 0.14)
        particle_colors[p][2] = base_color[p][2] * (1.0 - shade * 0.05)
        particle_colors[p][3] = 1.0


def main():
    init()

    window = ti.ui.Window("Realistic Snow - Taichi MPM", (1280, 800), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((0.03, 0.05, 0.10))   # 深蓝夜空
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(1.8, 1.1, 1.8)
    camera.lookat(0.5, 0.3, 0.5)
    camera.up(0, 1, 0)
    camera.fov(45)

    print("[提示] 按住鼠标右键 + WASD 移动视角，按 R 重置，ESC 退出")

    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == 'r':
                init()
            elif window.event.key == ti.ui.ESCAPE:
                break

        for _ in range(25):
            substep()
        update_colors()

        camera.track_user_inputs(window, movement_speed=0.03,
                                 hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        # 冷暖对比光照：暖色主光 + 冷色天光 + 偏冷环境光
        scene.ambient_light((0.25, 0.30, 0.40))
        scene.point_light(pos=(2.0, 2.5, 2.0),   color=(1.00, 0.95, 0.85))
        scene.point_light(pos=(-1.0, 2.0, -1.0), color=(0.30, 0.45, 0.70))

        scene.particles(x, radius=0.008, per_vertex_color=particle_colors)

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
