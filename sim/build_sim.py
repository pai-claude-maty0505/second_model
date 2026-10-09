"""Build the Gazebo (Harmonic) world for the SO-101 pick-and-place task (second model).

Differences from the first model
- The SO-101's original one-sided jaw (fixed jaw + one moving jaw) is replaced by a
  **parallel two-finger gripper**: both fingers slide open/closed symmetrically on
  prismatic joints (each finger has its own force-limited position controller).
- The block is a rigid body that **collides with the fingers**; it is held only by
  the friction produced by squeezing it (no DetachableJoint / no "grasp assist").
- Each finger has a 3 mm rubber pad (high friction) on its inner face; the finger body
  tapers in 10 small steps toward a rounded tip.

Steps
1. Patch the official SO-101 URDF: absolute mesh paths, fix base to world,
   remove the original jaw, add palm + two prismatic fingers.
2. Convert URDF -> SDF with `gz sdf -p`.
3. Post-process the SDF: collision bitmasks + friction, joint controllers, joint state publisher.
4. Write world.sdf: table, red block, blue tray, policy camera, video camera.

Usage: python3 build_sim.py --urdf so101_new_calib.urdf --out /content/pai_second/sim/gen
"""
import argparse, os, subprocess, xml.etree.ElementTree as ET

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
FINGERS = ["finger_left", "finger_right"]
HOME_ARM = [0.0, -1.5, 0.6, 1.6, 0.0]
HOME_FINGER = 0.02          # opening of each finger from the centre line (m); must match the task node
FINGER_MAX = 0.035
WORLD = "pick_place"
MODEL = "so101"

# Task geometry (metres, robot base at origin on the table surface)
BLOCK_SIZE = 0.025
BLOCK_MASS = 0.03
TRAY_XY = (0.10, 0.20)
TRAY_SIZE = 0.09
FRICTION = 5.0          # block and rubber pads (high-friction rubber)
FRICTION_BODY = 1.0     # palm / finger body (plastic)

# Gripper geometry in the gripper_link frame (its -z axis points from the wrist toward the fingertips;
# the original tool frame "gripper_frame_link" is at (-0.0079, 0, -0.0981) and is kept as the TCP).
CX = -0.0079                # centre line of the gripper (x) = TCP x
FINGER_Z = -0.081           # finger centre height  (fingers span z = -0.056 ... -0.106)
FINGER_LEN, FINGER_W = 0.050, 0.022   # finger length (z) and width (y)
PAD_T, PAD_W, PAD_LEN = 0.003, 0.020, 0.036   # rubber pad on the inner face: thickness, width, length
N_STEPS = 10                # finger body taper: number of thickness steps from root to tip


def _box_part(parent, xyz, size, material):
    for tag in ("visual", "collision"):
        e = ET.SubElement(parent, tag)
        ET.SubElement(e, "origin", xyz=" ".join(f"{v:.5f}" for v in xyz), rpy="0 0 0")
        g = ET.SubElement(e, "geometry")
        ET.SubElement(g, "box", size=" ".join(f"{v:.5f}" for v in size))
        if tag == "visual":
            ET.SubElement(e, "material", name=material)


def _geom(parent, tag, xyz, shape, rpy="0 0 0", material=None, name=None):
    e = ET.SubElement(parent, tag, **({"name": name} if name else {}))
    ET.SubElement(e, "origin", xyz=" ".join(f"{v:.5f}" for v in xyz), rpy=rpy)
    g = ET.SubElement(e, "geometry")
    kind, attrs = shape
    ET.SubElement(g, kind, **attrs)
    if material:
        ET.SubElement(e, "material", name=material)


def _add_finger(link, sign):
    """One finger in its joint frame (inner contact surface at x = 0, body extends to sign*x, tip toward -z).

    - rubber pad (3 mm, black, high friction) on the inner face; its long edges are rounded
    - finger body tapers in N_STEPS (10) steps toward the tip (10 -> 6 mm) and ends in a rounded tip
    Collisions are kept simple (pad box + body box) so that contacts stay stable.
    """
    L, W = FINGER_LEN, FINGER_W
    box = lambda sx, sy, sz: ("box", {"size": f"{sx:.5f} {sy:.5f} {sz:.5f}"})
    cyl = lambda r, l: ("cylinder", {"radius": f"{r:.5f}", "length": f"{l:.5f}"})
    pad_z = -L / 2 + 0.002 + PAD_LEN / 2                      # pad from 2 mm above the tip upward
    # --- collisions
    _geom(link, "collision", (sign * PAD_T / 2, 0, pad_z), box(PAD_T, PAD_W, PAD_LEN), name="rubber_pad")
    _geom(link, "collision", (sign * (PAD_T + 0.003), 0, 0), box(0.006, W, L), name="finger_body")
    # --- visuals: rubber pad with rounded long edges
    r = PAD_T / 2
    _geom(link, "visual", (sign * PAD_T / 2, 0, pad_z), box(PAD_T, PAD_W - 2 * r, PAD_LEN), material="rubber_black")
    for yy in (-(PAD_W / 2 - r), PAD_W / 2 - r):
        _geom(link, "visual", (sign * PAD_T / 2, yy, pad_z), cyl(r, PAD_LEN), material="rubber_black")
    # --- visuals: body tapering smoothly in N_STEPS steps (10 -> 6 mm, thickness measured outward from the pad)
    z0, z1 = L / 2, -L / 2 + 0.003
    segs = [(0.010 - 0.004 * i / (N_STEPS - 1), z0 + (z1 - z0) * i / N_STEPS, z0 + (z1 - z0) * (i + 1) / N_STEPS)
            for i in range(N_STEPS)]
    for t, z_top, z_bot in segs:
        _geom(link, "visual", (sign * (PAD_T + t / 2), 0, (z_top + z_bot) / 2), box(t, W, z_top - z_bot), material="finger_dark")
    # rounded tip (cylinder along y)
    _geom(link, "visual", (sign * (PAD_T + 0.003), 0, -L / 2 + 0.003), cyl(0.003, W), rpy="1.5708 0 0", material="finger_dark")


def patch_urdf(src, dst, mesh_dir):
    tree = ET.parse(src)
    root = tree.getroot()
    for m in root.iter("mesh"):
        fn = m.get("filename")
        m.set("filename", "file://" + os.path.join(mesh_dir, os.path.basename(fn)))
    for t in root.findall("transmission"):
        root.remove(t)
    # --- remove the original jaw: moving jaw link + its joint, and the follower body (fixed jaw) ---
    for j in root.findall("joint"):
        if j.get("name") == "gripper":
            root.remove(j)
    for l in root.findall("link"):
        if l.get("name") == "moving_jaw_so101_v1_link":
            root.remove(l)
    glink = [l for l in root.findall("link") if l.get("name") == "gripper_link"][0]
    for tag in ("visual", "collision"):
        for e in glink.findall(tag):
            m = e.find("geometry/mesh")
            if m is not None and "wrist_roll_follower" in m.get("filename"):
                glink.remove(e)
    # --- new parallel gripper: palm + rail on gripper_link, two prismatic fingers ---
    mat = ET.SubElement(root, "material", name="finger_dark")
    ET.SubElement(mat, "color", rgba="0.25 0.25 0.28 1")
    _box_part(glink, (-0.002, 0.0, -0.040), (0.060, 0.036, 0.016), "3d_printed")   # palm
    _box_part(glink, (CX, 0.0, -0.052), (0.082, 0.026, 0.008), "3d_printed")       # rail
    rub = ET.SubElement(root, "material", name="rubber_black")
    ET.SubElement(rub, "color", rgba="0.05 0.05 0.05 1")
    for name, sign in (("finger_left", -1.0), ("finger_right", 1.0)):
        link = ET.SubElement(root, "link", name=f"{name}_link")
        inert = ET.SubElement(link, "inertial")
        ET.SubElement(inert, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(inert, "mass", value="0.02")
        ET.SubElement(inert, "inertia", ixx="5e-6", ixy="0", ixz="0", iyy="5e-6", iyz="0", izz="1e-6")
        _add_finger(link, sign)
        j = ET.SubElement(root, "joint", name=name, type="prismatic")
        ET.SubElement(j, "origin", xyz=f"{CX} 0 {FINGER_Z}", rpy="0 0 0")
        ET.SubElement(j, "parent", link="gripper_link")
        ET.SubElement(j, "child", link=f"{name}_link")
        ET.SubElement(j, "axis", xyz=f"{sign} 0 0")       # positive = open (move outward)
        ET.SubElement(j, "limit", effort="20", velocity="0.5", lower="0", upper=str(FINGER_MAX))
    # --- fix base_link to the world ---
    ET.SubElement(root, "link", name="world")
    j = ET.SubElement(root, "joint", name="world_fixed", type="fixed")
    ET.SubElement(j, "parent", link="world")
    ET.SubElement(j, "child", link="base_link")
    ET.SubElement(j, "origin", xyz="0 0 0", rpy="0 0 0")
    tree.write(dst)


def plugin(parent, filename, name, **children):
    p = ET.SubElement(parent, "plugin", filename=filename, name=name)
    for k, v in children.items():
        e = ET.SubElement(p, k)
        e.text = str(v)
    return p


def _surface(col, bitmask, mu=None):
    surf = col.find("surface")
    if surf is None:
        surf = ET.SubElement(col, "surface")
    contact = surf.find("contact")
    if contact is None:
        contact = ET.SubElement(surf, "contact")
    ET.SubElement(contact, "collide_bitmask").text = bitmask
    if mu is not None:
        fr = ET.SubElement(surf, "friction")
        ode = ET.SubElement(fr, "ode")
        ET.SubElement(ode, "mu").text = str(mu)
        ET.SubElement(ode, "mu2").text = str(mu)


def post_process_sdf(sdf_in, sdf_out):
    tree = ET.parse(sdf_in)
    root = tree.getroot()
    model = root.find("model")
    model.set("name", MODEL)
    # Arm links (meshes): category 0x04, which nothing else accepts -> the arm never hits table/tray/block.
    # Gripper parts (boxes: palm, rail, fingers): 0x02 -> they DO collide with the block, the tray walls
    # and the table, so the block can only be lifted by squeezing it with friction.
    # Rubber pads (box 3 mm thick) get high friction; palm / finger bodies normal plastic friction.
    for col in model.iter("collision"):
        b = col.find("geometry/box/size")
        if b is not None:
            is_pad = abs(float(b.text.split()[0]) - PAD_T) < 1e-4
            _surface(col, "0x02", FRICTION if is_pad else FRICTION_BODY)
        else:
            _surface(col, "0x04")
    for j in model.iter("joint"):
        ax = j.find("axis")
        if ax is None or j.get("type") not in ("revolute", "prismatic"):
            continue
        dyn = ax.find("dynamics")
        if dyn is None:
            dyn = ET.SubElement(ax, "dynamics")
        ET.SubElement(dyn, "damping").text = "0.5" if j.get("type") == "revolute" else "2.0"
    for jn, q0 in zip(ARM_JOINTS, HOME_ARM):
        plugin(model, "gz-sim-joint-position-controller-system",
               "gz::sim::systems::JointPositionController",
               joint_name=jn, topic=f"/so101/{jn}/cmd",
               use_velocity_commands="true", p_gain=20, cmd_max=4, cmd_min=-4,
               initial_position=q0)
    # Fingers: force-controlled PID. Commanding "closed" (0) while the block is between the fingers
    # makes each finger push with at most cmd_max newtons -> squeeze force for friction grasping.
    for jn in FINGERS:
        plugin(model, "gz-sim-joint-position-controller-system",
               "gz::sim::systems::JointPositionController",
               joint_name=jn, topic=f"/so101/{jn}/cmd",
               p_gain=1500, i_gain=0, d_gain=10, cmd_max=15, cmd_min=-15,
               initial_position=HOME_FINGER)
    plugin(model, "gz-sim-joint-state-publisher-system",
           "gz::sim::systems::JointStatePublisher", topic="/so101/joint_states")
    tree.write(sdf_out)


def look_at(eye, target):
    """Return 'x y z roll pitch yaw' for a camera at eye looking at target (camera looks along +x)."""
    import math
    dx, dy, dz = (target[i] - eye[i] for i in range(3))
    yaw = math.atan2(dy, dx)
    pitch = math.atan2(-dz, math.hypot(dx, dy))
    return f"{eye[0]} {eye[1]} {eye[2]} 0 {pitch:.4f} {yaw:.4f}"


POLICY_CAM = ((0.40, 0.06, 0.30), (0.16, 0.07, 0.0))
VIDEO_CAM = ((0.45, -0.35, 0.32), (0.12, 0.06, 0.04))


def camera_sensor(name, w, h, rate, fov):
    return f"""
      <sensor name="{name}" type="camera">
        <camera><horizontal_fov>{fov}</horizontal_fov>
          <image><width>{w}</width><height>{h}</height><format>R8G8B8</format></image>
          <clip><near>0.02</near><far>5</far></clip></camera>
        <always_on>1</always_on><update_rate>{rate}</update_rate>
        <topic>/{name}/image</topic>
      </sensor>"""


def world_sdf(robot_sdf_path, img_w, img_h, fps):
    tx, ty = TRAY_XY
    s, wall_h, t = TRAY_SIZE, 0.02, 0.004
    walls = ""
    for i, (dx, dy, sx, sy) in enumerate([(s / 2, 0, t, s), (-s / 2, 0, t, s), (0, s / 2, s, t), (0, -s / 2, s, t)]):
        walls += f"""
        <collision name="w{i}"><pose>{dx} {dy} {wall_h/2} 0 0 0</pose><geometry><box><size>{sx} {sy} {wall_h}</size></box></geometry>
          <surface><contact><collide_bitmask>0x02</collide_bitmask></contact></surface></collision>
        <visual name="wv{i}"><pose>{dx} {dy} {wall_h/2} 0 0 0</pose><geometry><box><size>{sx} {sy} {wall_h}</size></box></geometry>
          <material><ambient>0.1 0.3 0.9 1</ambient><diffuse>0.1 0.3 0.9 1</diffuse></material></visual>"""
    b, m = BLOCK_SIZE, BLOCK_MASS
    inertia = m * b * b / 6
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <world name="{WORLD}">
    <physics name="1ms" type="ignored"><max_step_size>0.001</max_step_size><real_time_factor>1.0</real_time_factor></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
    <scene><ambient>0.6 0.6 0.6 1</ambient><background>0.85 0.88 0.92 1</background><shadows>false</shadows></scene>
    <light type="directional" name="sun"><cast_shadows>false</cast_shadows><pose>0 0 3 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse><specular>0.2 0.2 0.2 1</specular><direction>-0.3 0.2 -0.9</direction></light>

    <model name="table"><static>true</static>
      <link name="link">
        <collision name="c"><geometry><plane><normal>0 0 1</normal><size>2 2</size></plane></geometry>
          <surface><contact><collide_bitmask>0x03</collide_bitmask></contact>
          <friction><ode><mu>0.8</mu><mu2>0.8</mu2></ode></friction></surface></collision>
        <visual name="v"><geometry><plane><normal>0 0 1</normal><size>2 2</size></plane></geometry>
          <material><ambient>0.75 0.68 0.55 1</ambient><diffuse>0.75 0.68 0.55 1</diffuse></material></visual>
      </link></model>

    <model name="tray"><static>true</static><pose>{tx} {ty} 0 0 0 0</pose>
      <link name="link">
        <visual name="floor"><pose>0 0 0.001 0 0 0</pose><geometry><box><size>{s} {s} 0.002</size></box></geometry>
          <material><ambient>0.6 0.75 1 1</ambient><diffuse>0.6 0.75 1 1</diffuse></material></visual>{walls}
      </link></model>

    <!-- rigid block: held only by the friction of the two squeezing fingers -->
    <model name="block"><pose>0.2 0 {b/2} 0 0 0</pose>
      <link name="link">
        <inertial><mass>{m}</mass><inertia><ixx>{inertia:.3e}</ixx><iyy>{inertia:.3e}</iyy><izz>{inertia:.3e}</izz></inertia></inertial>
        <collision name="c"><geometry><box><size>{b} {b} {b}</size></box></geometry>
          <surface><contact><collide_bitmask>0x02</collide_bitmask></contact>
          <friction><ode><mu>{FRICTION}</mu><mu2>{FRICTION}</mu2></ode></friction></surface></collision>
        <visual name="v"><geometry><box><size>{b} {b} {b}</size></box></geometry>
          <material><ambient>0.9 0.1 0.1 1</ambient><diffuse>0.9 0.1 0.1 1</diffuse></material></visual>
      </link></model>

    <!-- policy camera: oblique view of workspace (input to the ACT policy) -->
    <model name="policy_cam"><static>true</static><pose>{look_at(*POLICY_CAM)}</pose>
      <link name="link">{camera_sensor("top_cam", img_w, img_h, fps, 1.05)}</link></model>
    <!-- presentation camera: only used for the submission video -->
    <model name="video_cam"><static>true</static><pose>{look_at(*VIDEO_CAM)}</pose>
      <link name="link">{camera_sensor("video_cam", 640, 480, fps, 1.0)}</link></model>

    <include><uri>file://{robot_sdf_path}</uri><name>{MODEL}</name><pose>0 0 0 0 0 0</pose></include>
  </world>
</sdf>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img_w", type=int, default=160)
    ap.add_argument("--img_h", type=int, default=120)
    ap.add_argument("--fps", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mesh_dir = os.path.join(os.path.dirname(os.path.abspath(a.urdf)), "assets")
    urdf2 = os.path.join(a.out, "so101_parallel.urdf")
    patch_urdf(a.urdf, urdf2, mesh_dir)
    raw_sdf = os.path.join(a.out, "so101_raw.sdf")
    with open(raw_sdf, "w") as f:
        subprocess.run(["gz", "sdf", "-p", urdf2], stdout=f, check=True)
    model_dir = os.path.join(a.out, "so101_model")
    os.makedirs(model_dir, exist_ok=True)
    post_process_sdf(raw_sdf, os.path.join(model_dir, "model.sdf"))
    with open(os.path.join(model_dir, "model.config"), "w") as f:
        f.write('<?xml version="1.0"?><model><name>so101</name><version>1.0</version><sdf version="1.9">model.sdf</sdf></model>')
    with open(os.path.join(a.out, "world.sdf"), "w") as f:
        f.write(world_sdf(model_dir, a.img_w, a.img_h, a.fps))
    print("wrote", os.path.join(a.out, "world.sdf"), "| robot urdf:", urdf2)


if __name__ == "__main__":
    main()
