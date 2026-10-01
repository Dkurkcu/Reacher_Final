import os
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils.ezpickle import EzPickle
from collections import deque # <--- NEW: For N-step history

class ReacherV3Env(MujocoEnv, EzPickle):
    """
    Reacher with:
    - Blind Stall Detection (Proprioception only).
    - Sting Memory (Short-term pain buffer).
    - Dynamic Progress Reset.
    - Squared velocity penalty.
    - STATIC Box Obstacle (Blind Agent).
    """

    def __init__(self, xml_file="reacher_v3.xml", frame_skip=1, render_mode=None):
        EzPickle.__init__(self, xml_file, frame_skip, render_mode)
        fullpath = os.path.join(os.path.dirname(__file__), xml_file)

        # Bounds for min–max normalization
        qpos_low = np.array([-np.pi, -np.pi], dtype=np.float64)
        qpos_high = np.array([ np.pi, np.pi], dtype=np.float64)
        qvel_low = np.array([-3.0, -3.0], dtype=np.float64)
        qvel_high = np.array([ 3.0,  3.0], dtype=np.float64)
        r_tip = 1.2
        ft_low = np.array([-r_tip, -r_tip, 0.0], dtype=np.float64)
        ft_high = np.array([ r_tip, r_tip, 0.2], dtype=np.float64)
        tgt_low = np.array([-0.9, -0.9, 0.0], dtype=np.float64)
        tgt_high = np.array([ 0.9, 0.9, 0.2], dtype=np.float64)
        
        # Torque bounds [-3, 3]
        torque_low = np.array([-3.0, -3.0], dtype=np.float64)
        torque_high = np.array([ 3.0,  3.0], dtype=np.float64)

        self.max_stall_energy = 2.0
        self.relax_steps = 0
        # Sting bounds (0 to 1)
        sting_low = np.array([0.0], dtype=np.float64)
        sting_high = np.array([1.0], dtype=np.float64)

        # 13 features: qpos(2) + qvel(2) + ft(3) + tgt(3) + torque(2) + STING(1)
        self._feat_low_fixed = np.concatenate([qpos_low, qvel_low, ft_low, tgt_low, torque_low, sting_low])
        self._feat_high_fixed = np.concatenate([qpos_high, qvel_high, ft_high, tgt_high, torque_high, sting_high])

        self._E_min, self._E_max = -1.0, 1.0
        self._norm_eps = 1e-8

        MujocoEnv.__init__(
            self,
            model_path=fullpath,
            frame_skip=frame_skip,
            # Shape is 14 (12 features + 1 sting + 1 episode energy)
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(14,), dtype=np.float32), 
            render_mode=render_mode,
        )
        
        # Initialize Sting
        self.current_sting = 0.0

        # Success delay machinery
        self.success_delay_steps = 20
        self.current_delay = 0
        self.delay_pending_termination = False

        # Energy/work accounting
        self._prev_qpos = None
        self._prev_tau = None
        self.episode_energy = 0.0
        
        # ----------------- NEW: BLIND STALL TRACKERS -----------------
        # No more "contact_energy" (Cheating). Now we use Stall Energy.
        self.effort_history = deque(maxlen=5) # 5-step smoothing
        self.motion_history = deque(maxlen=5)
        
        self.episode_stall_energy = 0.0
        
        self._pulse_pending = False
        self._pulse_steps_left = 0
        self._disengage_steps_left = 0
        self._was_stalled = False # Replaces caution_zone flag
        self._steps_in_episode = 0
        # Cache MuJoCo IDs (Still needed for Box position in reset, even if blind)
        self.box_body_id = self.model.body("box_body").id 

        # Progress shaping state
        self.prev_dist = None

        # For stable logging
        self._last_act_penalty = 0.0
        self._last_progress = 0.0

        # RNG
        self.np_random = np.random.RandomState()

    # ---------- helpers ----------
    def _minmax_to_minus1_1(self, x, lo, hi):
        s01 = (x - lo) / (np.maximum(hi - lo, self._norm_eps))
        return np.clip(2.0 * s01 - 1.0, -1.0, 1.0)

    def _build_raw_obs(self):
        qpos = self.data.qpos[:2].astype(np.float64)
        qvel = self.data.qvel[:2].astype(np.float64)
        fingertip_pos = self.data.site_xpos[self.model.site("fingertip_site").id].astype(np.float64)
        target_pos = self.model.site("target").pos.astype(np.float64)
        torques = self.data.actuator_force[:2].astype(np.float64)
        
        # Get Sting as array
        sting = np.array([self.current_sting], dtype=np.float64)
        
        epE = np.array([self.episode_energy], dtype=np.float64)
        
        # Concatenate: 12 Feats + 1 Sting + 1 Energy = 14 Total
        return np.concatenate([qpos, qvel, fingertip_pos, target_pos, torques, sting, epE])

    def _get_normalized_obs(self):
        raw = self._build_raw_obs()
        # Normalize first 13 features (everything except energy)
        norm_fixed = self._minmax_to_minus1_1(raw[:13], self._feat_low_fixed, self._feat_high_fixed)
        E = float(raw[13]) 
        
        if np.isfinite(E):
            if E < self._E_min: self._E_min = E
            if E > self._E_max: self._E_max = E
        if abs(self._E_max - self._E_min) < 1e-6:
            self._E_max = self._E_min + 1e-6
        E_norm = self._minmax_to_minus1_1(np.array([E]), np.array([self._E_min]), np.array([self._E_max]))[0]
        
        return np.concatenate([norm_fixed, np.array([E_norm], dtype=np.float64)]).astype(np.float32)

    # ---------- Gymnasium API ----------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.RandomState(int(seed))
        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    def seed(self, seed_value=None):
        self.np_random = np.random.RandomState(seed_value)
        return [seed_value]

    def get_obs(self):
        return self._get_normalized_obs()

    def _get_obs(self):
        return self.get_obs()
    def reset_model(self):
        rng = self.np_random

        # ----------------- SAFER SPAWN LOGIC -----------------
        while True:
            # 1. Generate Target
            inner_radius = 0.2
            outer_radius = 0.9
            r2 = rng.uniform(inner_radius**2, outer_radius**2)
            r = np.sqrt(r2)
            theta = rng.uniform(0, 2*np.pi)
            offset = r * np.array([np.cos(theta), np.sin(theta)])
            target_pos = np.array([offset[0], offset[1], 0.1]) 

            # 2. Generate Box
            box_inner_radius = 0.15 
            box_outer_radius = 0.8
            box_r2 = rng.uniform(box_inner_radius**2, box_outer_radius**2)
            box_r = np.sqrt(box_r2)
            box_theta = rng.uniform(0, 2*np.pi)
            box_offset = box_r * np.array([np.cos(box_theta), np.sin(box_theta)])

            # 3. Check Distance
            dist_check = np.linalg.norm(offset - box_offset)
            if dist_check > 0.2: # 20cm buffer
                break
        # -----------------------------------------------------

        # Apply target position (Target is a site, so site_pos is still correct)
        self.model.site_pos[self.model.site("target").id][:] = target_pos

        # =====================================================
        # STATE GENERATION FOR FREEJOINT
        # Create zero arrays of the exact size MuJoCo wants
        # nq will be 9 (2 arm + 7 box). nv will be 8 (2 arm + 6 box).
        qpos_full = np.zeros(self.model.nq)
        qvel_full = np.zeros(self.model.nv)

        # 1. Set the Arm (Indices 0 and 1)
        qpos_full[0:2] = np.array([0.0, 0.0]) + rng.uniform(low=-0.1, high=0.1, size=2)
        qvel_full[0:2] = np.zeros(2) + rng.uniform(low=-0.005, high=0.005, size=2)

        # 2. Set the Freejoint Box Position (Indices 2, 3, 4 are X, Y, Z)
        # We NO LONGER use self.model.body_pos for the box! It lives in qpos now.
        qpos_full[2] = box_offset[0]  # Box X
        qpos_full[3] = box_offset[1]  # Box Y
        qpos_full[4] = 0.05           # Box Z

        # 3. Set the Freejoint Box Rotation (Quaternion: Indices 5, 6, 7, 8)
        # W, X, Y, Z for a standard, flat rotation is 1, 0, 0, 0
        qpos_full[5] = 1.0
        qpos_full[6] = 0.0
        qpos_full[7] = 0.0
        qpos_full[8] = 0.0
        # =====================================================

        # Set Arm & Box State
        self.set_state(qpos_full, qvel_full)
        
        self.do_simulation(np.zeros(self.model.nu), 1)
        
        # Reset trackers
        self.episode_energy = 0.0
        self.episode_stall_energy = 0.0 
        self.current_sting = 0.0 
        self._steps_in_episode = 0
        self._E_min, self._E_max = -1.0, 1.0

        # Reset History Buffers
        self.effort_history.clear()
        self.motion_history.clear()

        self._prev_qpos = self.data.qpos[:2].copy()
        self._prev_tau = np.zeros(2, dtype=np.float64)

        fingertip_pos = self.data.site_xpos[self.model.site("fingertip_site").id]
        current_target_pos = self.model.site("target").pos
        self.prev_dist = float(np.linalg.norm(fingertip_pos - current_target_pos))

        self.current_delay = 0
        self.delay_pending_termination = False
        self._last_act_penalty = 0.0
        self._last_progress = 0.0
        self._was_stalled = False 
        self._disengage_steps_left = 0
        self.relax_steps = 0

        return self.get_obs()
    def step(self, action):
        self._steps_in_episode += 1

        # 1. SUCCESS FREEZE
        if self.current_delay > 0:
            self.current_delay -= 1
            fingertip_pos = self.data.site_xpos[self.model.site("fingertip_site").id]
            target_pos = self.model.site("target").pos
            dist = float(np.linalg.norm(fingertip_pos - target_pos))
            stall_energy_limit_reached = self.episode_stall_energy >= self.max_stall_energy
            
            info = {
                "delay": True, "distance": dist, "success": dist < 0.05,
                "episode_energy": self.episode_energy,
                "act_penalty": float(self._last_act_penalty),
                "progress": float(self._last_progress),
                "episode_stall_energy": self.episode_stall_energy,
                "stall_energy_limit_reached": stall_energy_limit_reached,
                "qvel": np.zeros(2)
                
            }
            term = (self.current_delay == 0 and self.delay_pending_termination)
            return self.get_obs(), 0.0, term, False, info

        # 2. PHYSICS
        self.do_simulation(action, self.frame_skip)

        # 3. CALCULATE STATE
        qvel = self.data.qvel[:2]
        fingertip_pos = self.data.site_xpos[self.model.site("fingertip_site").id]
        target_pos = self.model.site("target").pos
        dist = float(np.linalg.norm(fingertip_pos - target_pos))

        # 4. SENSORS
        curr_effort = np.sum(np.square(self.data.ctrl))
        self.effort_history.append(curr_effort)
        curr_motion = np.sum(np.square(self.data.qvel))
        self.motion_history.append(curr_motion)
        
        avg_effort = np.mean(self.effort_history) if self.effort_history else 0.0
        avg_motion = np.mean(self.motion_history) if self.motion_history else 0.0

        # 5. STRICT STALL DETECTOR
        # Only safe if at target (dist < 0.05). Anywhere else = Danger.
        if self._steps_in_episode > 20 and dist > 0.1:
            is_stalled = (avg_effort > 0.6 ) and (avg_motion < 0.02)#0.6 ve 0.02
        else:
            is_stalled = False

        # 6. STING LOGIC (With Decay)
        self.current_sting *= 0.996 
        
        # If the sensor says stalled, Reset Sting to 1.0
        if is_stalled:
            self.current_sting = 1.0 
            if not self._was_stalled:
                self._was_stalled = True
        
        stall_penalty = 0.0
        if is_stalled:
            stall_damage = avg_effort * 0.1 * self.frame_skip
            self.episode_stall_energy += stall_damage
            stall_penalty = 0.1 * avg_effort 

        # 7. ENERGY
        qpos_now = self.data.qpos[:2].copy()
        dq = qpos_now - (self._prev_qpos if self._prev_qpos is not None else qpos_now)
        tau_prev = self._prev_tau if self._prev_tau is not None else np.zeros(2)
        energy_inc = float(np.dot(tau_prev, dq))
        self.episode_energy += energy_inc
        self._prev_qpos = qpos_now
        self._prev_tau = self.data.qfrc_actuator[:2].copy()

        # --- 8. THE DUAL-MODE REWARD SYSTEM ---
        
        # 1. Calculate Raw Progress FIRST (so it is available for logs later)
        raw_progress = float(self.prev_dist - dist)

        # 2. Check Panic Mode
        is_panicking = (self.current_sting > 0.5)

        if is_panicking:
            # === MODE A: PANIC (RUN AWAY) ===
            dist_reward = 0.0 
            
            # Use absolute distance change to calculate retreat reward
            if dist > self.prev_dist:
                progress_reward = (dist - self.prev_dist) * 150.0
            else:
                progress_reward = -0.05
                
        else:
            # === MODE B: HUNTER (CHASE TARGET) ===
            dist_reward = -dist 
            
            # Use the raw_progress we calculated above
            progress_reward = raw_progress * 50.0

        # --- COSTS ---
        time_penalty = 0.005
        energy_cost = 0.05 * abs(energy_inc)
        vel_penalty = 0.1 * float(np.dot(qvel, qvel))
        act_penalty = 0.01 * float(np.dot(action, action))

        # Re-Engagement Penalty (Keeps it honest during panic)
        sustain_penalty = 0.0
        if self.current_sting > 0.8 and avg_effort > 0.5:
            sustain_penalty = 1.0 * avg_effort  

        reward = dist_reward + progress_reward \
                 - time_penalty - energy_cost - vel_penalty - act_penalty \
                 - stall_penalty - sustain_penalty
        
        # 9. SUCCESS
        success = dist < 0.05
        if success:
            reward += 100.0 
            if dist <= 0.04: reward += 20.0
            if self.episode_stall_energy < 4.0: reward += 50.0
            
            self.current_delay = self.success_delay_steps
            self.delay_pending_termination = True
            
        # 10. COOLDOWN (CRITICAL: DO NOT DELETE)
        # Prevents sensor flicker so the Sting can actually decay.
        if avg_effort < 0.1: 
            self.relax_steps += 1
        else:
            self.relax_steps = 0
            
        if self.relax_steps > 5:
            self._was_stalled = False

        self.prev_dist = dist 
        self._last_act_penalty = act_penalty
        self._last_progress = progress_reward 

        truncated = False
        if self.episode_stall_energy > self.max_stall_energy: 
            truncated = True
            self.current_delay = 0 
            
        info = {
            "distance": dist, "success": success, "progress": raw_progress if not is_panicking else 0.0,
            "energy_inc": energy_inc, "episode_energy": self.episode_energy,
            "act_penalty": act_penalty, "sting": self.current_sting, 
            "is_stalled": is_stalled, "stall_penalty": stall_penalty,
            "episode_stall_energy": self.episode_stall_energy,
            "stall_energy_limit_reached": truncated,
            "qvel": qvel,
            
        }
       
        return self.get_obs(), reward, False, truncated, info
    
    def render(self):
        return super().render()

    def close(self):
        return super().close()









        