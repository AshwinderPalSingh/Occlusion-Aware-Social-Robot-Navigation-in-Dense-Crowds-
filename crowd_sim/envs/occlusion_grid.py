import numpy as np

"""
Builds the two occupancy grids used by Stage 2's occlusion-inference VAE
(sim.occlusion_method == 'vae'):

- label_grid: privileged ground truth, every human's true circle stamped in,
  visible or not. Used only as the Label_VAE's training target.
- sensor_grid: what the robot's own raycast (already computed each step by
  get_num_human_in_fov()'s ray_test(camera_fov=True) call) actually sees:
  free space, human hits, and an unknown/occluded shadow behind whatever
  blocked a ray that didn't hit a human.

Both grids are robot-centered and expressed in a "ray frame" where angle 0
points along the robot's heading (matching sim.cam_ray_angles, i.e.
world-frame ray direction = robot.theta + cam_ray_angles). This is a
different rotation convention than world_to_robot() (which bakes in a -90deg
offset for the spatial_edges network inputs) -- that's fine, this frame is
only ever consumed by Sensor_VAE/Label_VAE, never compared against
world_to_robot()'s outputs.

Values: 0 = free, 1 = occupied by a human, 0.5 = unknown/occluded (sensor_grid only).
Obstacles themselves are not encoded here -- the policy receives them through the
obstacle-only lidar point cloud (ob['point_clouds']); only humans need inference.

Known limitation: rays that hit nothing are marked free out to the grid's edge
(max_extent), not to the sensor range, so at sensor ranges below the grid half-width
(e.g. the 3 m and 5 m cells of the range sweep) unsensed cells read as free, not unknown.
"""

UNKNOWN = 0.5
OCCUPIED = 1.0
FREE = 0.0


def _world_to_ray_frame(sim, world_dx, world_dy):
    """Rotate a world-frame offset into the same frame cam_ray_angles is defined in."""
    dist = np.hypot(world_dx, world_dy)
    angle = np.arctan2(world_dy, world_dx) - sim.robot.theta
    return dist * np.cos(angle), dist * np.sin(angle)


def _to_grid_index(lx, ly, half_extent, grid_res, grid_size):
    col = np.floor((lx + half_extent) / grid_res).astype(np.int64)
    row = np.floor((ly + half_extent) / grid_res).astype(np.int64)
    valid = (col >= 0) & (col < grid_size) & (row >= 0) & (row < grid_size)
    return row, col, valid


def build_label_grid(sim):
    grid_res = sim.config.pas.grid_res
    grid_size = sim.config.pas.grid_size
    half_extent = grid_size * grid_res / 2.

    label_grid = np.zeros((grid_size, grid_size), dtype=np.float32)

    for human in sim.humans:
        lx, ly = _world_to_ray_frame(sim, human.px - sim.robot.px, human.py - sim.robot.py)
        if abs(lx) > half_extent + human.radius or abs(ly) > half_extent + human.radius:
            continue
        # rasterize the human's circle as a filled disk of cells
        r_cells = max(1, int(np.ceil(human.radius / grid_res)))
        center_row, center_col, valid = _to_grid_index(np.array([lx]), np.array([ly]), half_extent, grid_res,
                                                        grid_size)
        if not valid[0]:
            continue
        center_row, center_col = int(center_row[0]), int(center_col[0])
        row_lo, row_hi = max(0, center_row - r_cells), min(grid_size, center_row + r_cells + 1)
        col_lo, col_hi = max(0, center_col - r_cells), min(grid_size, center_col + r_cells + 1)
        rows, cols = np.meshgrid(np.arange(row_lo, row_hi), np.arange(col_lo, col_hi), indexing='ij')
        dist_cells = np.hypot(rows - center_row, cols - center_col) * grid_res
        mask = dist_cells <= human.radius
        label_grid[row_lo:row_hi, col_lo:col_hi][mask] = OCCUPIED

    return label_grid[np.newaxis, :, :]  # (1, grid_size, grid_size)


def build_sensor_grid(sim):
    grid_res = sim.config.pas.grid_res
    grid_size = sim.config.pas.grid_size
    half_extent = grid_size * grid_res / 2.
    max_extent = half_extent * np.sqrt(2)  # far enough to cover grid corners

    sensor_grid = np.full((grid_size, grid_size), UNKNOWN, dtype=np.float32)

    hit_dist = sim.closest_hit_dist  # (cam_ray_num,), true hit distance regardless of what was hit
    hit_is_human = sim.closest_hit_id != -1  # ray_test() only ever sets an id for human hits
    sensor_range = sim.robot.sensor_range
    angles = sim.cam_ray_angles

    no_hit = hit_dist >= sensor_range - 1e-3
    # everything that isn't a human hit and isn't "no hit at all" must have hit an obstacle/border
    hit_is_obstacle = (~hit_is_human) & (~no_hit)

    n_steps = int(np.ceil(max_extent / grid_res)) + 1
    steps = np.linspace(0., max_extent, n_steps)  # (n_steps,)

    cos_a = np.cos(angles)  # (ray_num,)
    sin_a = np.sin(angles)

    # local (x, y) of every sample point along every ray: (ray_num, n_steps)
    lx = steps[np.newaxis, :] * cos_a[:, np.newaxis]
    ly = steps[np.newaxis, :] * sin_a[:, np.newaxis]

    free_reach = np.where(no_hit, max_extent, hit_dist)  # (ray_num,), how far "free" extends
    is_free_sample = steps[np.newaxis, :] <= free_reach[:, np.newaxis]  # (ray_num, n_steps)

    rows, cols, valid = _to_grid_index(lx.ravel(), ly.ravel(), half_extent, grid_res, grid_size)
    rows, cols, valid = rows.reshape(lx.shape), cols.reshape(lx.shape), valid.reshape(lx.shape)

    free_mask = is_free_sample & valid
    sensor_grid[rows[free_mask], cols[free_mask]] = FREE

    # occupied cell at the exact hit point, for rays that hit a human
    hit_lx = hit_dist * cos_a
    hit_ly = hit_dist * sin_a
    hit_rows, hit_cols, hit_valid = _to_grid_index(hit_lx, hit_ly, half_extent, grid_res, grid_size)
    human_hit_mask = hit_is_human & hit_valid
    sensor_grid[hit_rows[human_hit_mask], hit_cols[human_hit_mask]] = OCCUPIED

    # cells beyond an obstacle hit stay UNKNOWN (already the array's default value);
    # nothing to do here since sensor_grid was initialized to UNKNOWN and only
    # free/occupied cells get overwritten above.
    del hit_is_obstacle  # documents intent; no separate write needed given the UNKNOWN default

    return sensor_grid[np.newaxis, :, :]  # (1, grid_size, grid_size)


def build_grids(sim):
    """Returns (sensor_grid, label_grid), each shape (1, grid_size, grid_size).

    Must be called after get_num_human_in_fov() but before any later ray_test() /
    ray_test_no_humans() call in the same step -- get_num_human_in_fov() populates
    self.closest_hit_dist/self.closest_hit_id via a camera_fov=True ray test (indexed by
    cam_ray_angles, which this module assumes); a later full-lidar-sweep ray_test() call
    (indexed by ray_angles) would overwrite both with different, angle-mismatched data.
    crowd_sim_tb2_obs.py's generate_ob() calls this right after get_num_human_in_fov(),
    before its own point_clouds ray_test() call, so this precondition holds.
    """
    return build_sensor_grid(sim), build_label_grid(sim)
