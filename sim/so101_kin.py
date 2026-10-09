"""SO-101 forward / inverse kinematics (numpy only, URDF parsed directly).

Used by both the scripted expert (data collection) and the grasp-assist logic.
Only numpy is required so it runs under the ROS 2 system Python as well.
"""
import xml.etree.ElementTree as ET
import numpy as np

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
ALL_JOINTS = ARM_JOINTS + ["gripper"]


def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _axis_angle(axis, a):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def _T(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


class SO101Kinematics:
    def __init__(self, urdf_path, base_xyz=(0.0, 0.0, 0.0), tip_link="gripper_frame_link"):
        root = ET.parse(urdf_path).getroot()
        joints = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = [float(v) for v in o.get("xyz", "0 0 0").split()]
            rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
            ax = j.find("axis")
            axis = [float(v) for v in ax.get("xyz").split()] if ax is not None else [0, 0, 1]
            lim = j.find("limit")
            lo, hi = (float(lim.get("lower")), float(lim.get("upper"))) if lim is not None else (0, 0)
            joints[j.find("child").get("link")] = dict(
                name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"),
                T=_T(_rpy_to_R(*rpy), xyz), axis=axis, lo=lo, hi=hi)
        # chain from tip back to root
        chain, link = [], tip_link
        while link in joints:
            chain.append(joints[link])
            link = joints[link]["parent"]
        self.chain = chain[::-1]
        self.base_T = _T(np.eye(3), base_xyz)
        self.limits = {c["name"]: (c["lo"], c["hi"]) for c in self.chain if c["type"] == "revolute"}
        names = [c["name"] for c in self.chain if c["type"] == "revolute"]
        assert names == ARM_JOINTS, names

    def fk(self, q):
        """q: 5 arm joint angles -> 4x4 pose of the tool frame in world."""
        T = self.base_T.copy()
        i = 0
        for c in self.chain:
            T = T @ c["T"]
            if c["type"] == "revolute":
                T = T @ _T(_axis_angle(c["axis"], q[i]), [0, 0, 0])
                i += 1
        return T

    def ik(self, target_xyz, q0=None, down_weight=0.3, iters=300, tol=1e-3):
        """Damped-least-squares IK for position + 'tool pointing down' + wrist_roll = yaw.

        Returns (q, ok). The tool's approach axis (local z) is pushed to world -z,
        which gives a top-down grasp; wrist_roll is kept fixed (0 by default).
        """
        lo = np.array([self.limits[n][0] for n in ARM_JOINTS])
        hi = np.array([self.limits[n][1] for n in ARM_JOINTS])
        q = np.zeros(5) if q0 is None else np.array(q0, float)
        target = np.asarray(target_xyz, float)
        # seed pan toward the target
        q[0] = np.clip(np.arctan2(target[1] - self.base_T[1, 3], target[0] - self.base_T[0, 3]) * self._pan_sign(), lo[0], hi[0]) if q0 is None else q[0]
        active = [0, 1, 2, 3]  # wrist_roll stays fixed
        for _ in range(iters):
            e = self._err(q, target, down_weight)
            if np.linalg.norm(e[:3]) < tol and np.linalg.norm(e[3:]) < 0.05:
                break
            J = np.zeros((6, len(active)))
            for k, i in enumerate(active):
                dq = np.zeros(5); dq[i] = 1e-5
                J[:, k] = (self._err(q + dq, target, down_weight) - e) / -1e-5
            lam = 1e-3
            step = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
            q[active] = np.clip(q[active] + np.clip(step, -0.2, 0.2), lo[active], hi[active])
        pos_err = np.linalg.norm(self.fk(q)[:3, 3] - target)
        return q, pos_err < 5e-3

    def _err(self, q, target, w):
        T = self.fk(q)
        ep = target - T[:3, 3]
        approach = T[:3, 2]
        eo = (np.array([0, 0, -1.0]) - approach) * w
        return np.concatenate([ep, eo])

    def _pan_sign(self):
        # figure out whether +pan rotates toward +y (depends on URDF axis conventions)
        if not hasattr(self, "_ps"):
            p0 = self.fk(np.zeros(5))[:3, 3] - self.base_T[:3, 3]
            p1 = self.fk(np.array([0.1, 0, 0, 0, 0]))[:3, 3] - self.base_T[:3, 3]
            a0, a1 = np.arctan2(p0[1], p0[0]), np.arctan2(p1[1], p1[0])
            self._ps = 1.0 if a1 > a0 else -1.0
            self._pan_off = a0
        return self._ps
