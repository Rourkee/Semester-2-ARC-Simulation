##### robot_core.py
import math
import time
import numpy as np
import mujoco


class RobotCore:
    """
    Thin wrapper around MuJoCo model/data that exposes:
    - gripper, operational-space control, nullspace
    - simple EE path helpers (correct_to, home)
    - quaternion utilities
    - plate helpers: angle, geometry
    - min-jerk interpolation
    - site/geom helpers + knob target discovery
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, memory, K, qpos0):
        self.model = model
        self.data = data
        self.memory = memory
        self.K = list(K)
        self.qpos0 = list(qpos0)

    # ---------------- low-level control ----------------

    def gripper(self, open=True):
        """Open/close both finger actuators symmetrically."""
        open = bool(open)
        tgt = 0.04 if open else 0.0
        # guard if actuators are missing
        for a in ("pos_panda_finger_joint1", "pos_panda_finger_joint2"):
            try:
                self.data.actuator(a).ctrl = tgt
            except Exception:
                pass

    def control(self,
                xpos_d, xquat_d,
                *,
                qpref=None,            # optional: target joint angles (len 7)
                qmask=None,            # optional: 0/1 weights (len 7)
                null_k=20.0,           # nullspace stiffness
                null_d=2.0,            # nullspace damping
                null_mu=1e-6):         # DLS regularizer
        """Operational-space PD with nullspace posture."""
        xpos = self.data.body("panda_hand").xpos
        xquat = self.data.body("panda_hand").xquat

        # Jacobians
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        bodyid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "panda_hand")
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, bodyid)
        J = np.concatenate((jacp, jacr))          # (6 x nv)

        # Task-space errors
        error = np.zeros(6)
        error[:3] = np.asarray(xpos_d, float).reshape(3) - xpos
        res = np.zeros(3)
        # orientation error in local frame
        mujoco.mju_subQuat(res, xquat, np.asarray(xquat_d, float).reshape(4))
        mujoco.mju_rotVecQuat(res, res, xquat)
        error[3:] = -res

        v = J @ self.data.qvel

        # Task torques
        Kdiag = np.array(self.K, dtype=float)
        Ddiag = 2.0 * np.sqrt(Kdiag)
        tau_task = J.T @ (np.diag(Kdiag) @ error - np.diag(Ddiag) @ v)
        tau = np.zeros(self.model.nv)

        # start from bias
        for i in range(1, 8):
            tau[self.model.joint(f"panda_joint{i}").dofadr] = self.data.joint(f"panda_joint{i}").qfrc_bias

        tau[:7] += tau_task[:7]

        # Optional nullspace posture
        if qpref is not None:
            q = np.array([self.data.joint(f"panda_joint{i}").qpos[0] for i in range(1, 8)], dtype=float)
            dq = np.array([self.data.joint(f"panda_joint{i}").qvel[0] for i in range(1, 8)], dtype=float)
            qpref = np.asarray(qpref, dtype=float).reshape(7)
            if qmask is None:
                qmask = np.ones(7, dtype=float)
            else:
                qmask = np.asarray(qmask, dtype=float).reshape(7)

            e_q  = (qpref - q) * qmask
            de_q = dq * qmask

            JJt = J @ J.T
            invJJt = np.linalg.inv(JJt + null_mu * np.eye(6))
            N = np.eye(7) - J[:,:7].T @ invJJt @ J[:,:7]
            tau_null = N @ (null_k * e_q - null_d * de_q)
            tau[:7] += tau_null

        # Apply
        for i in range(1, 8):
            dofadr = self.model.joint(f"panda_joint{i}").dofadr
            try:
                self.data.actuator(f"panda_joint{i}").ctrl = tau[dofadr]
            except Exception:
                pass

    def correct_to(self, target_pos, xquat, threshold=0.01, steps=50):
        current_pos = self.data.body("panda_hand").xpos.copy()
        error = np.linalg.norm(np.asarray(target_pos, float).reshape(3) - current_pos)
        if error > threshold:
            path = np.linspace(current_pos, target_pos, steps)
            for pos in path:
                self.control(pos, xquat)
                mujoco.mj_step(self.model, self.data)
                if self.memory:
                    self.memory.trace_step(self.model, self.data, phase="correct_to")
                time.sleep(1e-4)

    def reset_joints_to_home(self, steps=500):
        current_qpos = np.array([self.data.joint(f"panda_joint{i+1}").qpos[0] for i in range(7)])
        target_qpos = np.array(self.qpos0)
        trajectory = np.linspace(current_qpos, target_qpos, steps)
        for step_qpos in trajectory:
            for i in range(7):
                self.data.joint(f"panda_joint{i+1}").qpos[0] = step_qpos[i]
            mujoco.mj_forward(self.model, self.data)
            if self.memory:
                self.memory.trace_step(self.model, self.data, phase="home")
            time.sleep(1e-3)

    # -------------- quaternion & math utils --------------

    @staticmethod
    def _unit_quat(q):
        q = np.array(q, dtype=float)
        return q / (np.linalg.norm(q) + 1e-12)

    @staticmethod
    def _apply_ypr(xquat_ref, *, yaw=0.0, pitch=0.0, roll=0.0):
        zaxis = np.array([0.0, 0.0, 1.0])
        yaxis = np.array([0.0, 1.0, 0.0])
        xaxis = np.array([1.0, 0.0, 0.0])
        qz = np.zeros(4); qy = np.zeros(4); qx = np.zeros(4)
        mujoco.mju_axisAngle2Quat(qz, zaxis, float(yaw))
        mujoco.mju_axisAngle2Quat(qy, yaxis, float(pitch))
        mujoco.mju_axisAngle2Quat(qx, xaxis, float(roll))
        tmp  = np.zeros(4)
        tmp2 = np.zeros(4)
        out  = np.zeros(4)
        mujoco.mju_mulQuat(tmp,  qy, qx)     # Ry * Rx
        mujoco.mju_mulQuat(tmp2, qz, tmp)    # Rz * (Ry*Rx)
        mujoco.mju_mulQuat(out,  tmp2, xquat_ref)
        return out

    @staticmethod
    def _slerp(q0, q1, t: float):
        q0 = np.asarray(q0, dtype=float); q0 /= (np.linalg.norm(q0) + 1e-12)
        q1 = np.asarray(q1, dtype=float); q1 /= (np.linalg.norm(q1) + 1e-12)
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        DOT = 0.9995
        t = float(t)
        if dot > DOT:
            out = q0 + t * (q1 - q0)
            return out / (np.linalg.norm(out) + 1e-12)
        theta_0 = math.acos(np.clip(dot, -1.0, 1.0))
        sin_0   = math.sin(theta_0)
        theta   = theta_0 * t
        s0 = math.sin(theta_0 - theta) / sin_0
        s1 = math.sin(theta) / sin_0
        return s0 * q0 + s1 * q1

    @staticmethod
    def _wrap_to_pi(a: float) -> float:
        return (float(a) + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _minjerk(a, b, n):
        n = int(max(2, n))
        t = np.linspace(0.0, 1.0, n)
        s = t**3 * (10 - 15*t + 6*t**2)
        a = np.asarray(a, dtype=float).reshape(-1)
        b = np.asarray(b, dtype=float).reshape(-1)
        return (1 - s)[:, None] * a + s[:, None] * b

    # -------------- plate helpers --------------

    def plate_geom_info(self):
        pc = self.data.body("spinning_plate").xpos.copy()
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "plate_geom")
        r_plate = float(self.model.geom_size[gid][0])
        half_h  = float(self.model.geom_size[gid][1])
        z_top   = pc[2] + half_h
        return pc, r_plate, z_top

    def plate_angle(self):
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "plate_rotation")
        if jid < 0:
            return 0.0
        qadr = int(self.model.jnt_qposadr[jid])
        return float(self.data.qpos[qadr])

    # -------------- site/geom helpers --------------

    def site_pos(self, name: str) -> np.ndarray | None:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            return None
        return self.data.site_xpos[sid].copy()

    def geom_pos(self, name: str) -> np.ndarray | None:
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            return None
        return self.data.geom_xpos[gid].copy()

    def has_site(self, name: str) -> bool:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0

    def has_geom(self, name: str) -> bool:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0

    # -------------- knob discovery --------------

    def find_knob_targets(self):
        """
        Return a dict with best grasp targets on the plate:
          {
            "primary": {"pos": (3,), "type": "site|geom", "name": "<name>"},
            "secondary": {...}  # optional
          }
        Preference: site 'knob_grasp' -> geom 'knob_handle' -> 'knob_stem'
                    and for secondary: 'knob2_grasp' -> 'knob2_handle' -> 'knob2_stem'
        """
        out = {}

        # primary
        for nm, kind in (("knob_grasp", "site"), ("knob_handle", "geom"), ("knob_stem", "geom")):
            pos = self.site_pos(nm) if kind == "site" else self.geom_pos(nm)
            if pos is not None:
                out["primary"] = {"pos": pos, "type": kind, "name": nm}
                break


        return out
