import os
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils.ezpickle import EzPickle


class ReacherV3Env(MujocoEnv, EzPickle):
    """
    Reacher with:
    - Custom Safety Zones (0.015J reward/0.02J penalty/0.04J truncation).
    - Dynamic Progress Reset after successful disengagement.
    - Squared velocity penalty for smoother motion.
    """

    def __init__(self, xml_file="reacher_v3.xml", frame_skip=1, render_mode=None):
        EzPickle.__init__(self, xml_file, frame_skip, render_mode)
        fullpath = os.path.join(os.path.dirname(__file__), xml_file)

        # Bounds for min–max normalization (first 10 features: qpos, qvel, ft_pos, tgt_pos)
        qpos_low = np.array([-np.pi, -np.pi], dtype=np.float64)
        qpos_high = np.array([ np.pi, np.pi], dtype=np.float64)
        qvel_low = np.array([-3.0, -3.0], dtype=np.float64)
        qvel_high = np.array([ 3.0,3.0], dtype=np.float64)
        r_tip = 1.2
        ft_low = np.array([-r_tip, -r_tip, 0.0], dtype=np.float64)
        ft_high = np.array([ r_tip, r_tip, 0.2], dtype=np.float64)
        tgt_low = np.array([-0.9, -0.9, 0.0], dtype=np.float64)
        tgt_high = np.array([ 0.9, 0.9, 0.2], dtype=np.float64)

        # 10 features: qpos(2) + qvel(2) + ft_pos(3) + tgt_pos(3)
        self._feat_low_fixed = np.concatenate([qpos_low, qvel_low, ft_low, tgt_low])
        self._feat_high_fixed = np.concatenate([qpos_high, qvel_high, ft_high, tgt_high])
        
        self._E_min, self._E_max = -1.0, 1.0
        self._norm_eps = 1e-8

        MujocoEnv.__init__(
            self,
            model_path=fullpath,
            frame_skip=frame_skip,
            # Shape is 11 (10 features + 1 episode energy)
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(11,), dtype=np.float32), 
            render_mode=render_mode,
        )

        # Success delay machinery (for clean termination after success)
        self.success_delay_steps = 20
        self.current_delay = 0
        self.delay_pending_termination = False

        # Energy/work accounting
        self._prev_qpos = None
        self._prev_tau = None
        self.episode_energy = 0.0
        
        # NEW: Obstacle and Safety Trackers
        self.contact_reward_limit = 0.018# Reward for dropping below this (Safe Zone boundary)
        self.contact_warning_limit = 0.02# Penalty trigger (Caution Zone Entry)
        self.contact_energy_limit = 0.07 # Hard limit for truncation (Failure Zone)
        self.episode_contact_energy = 0.0
        self._pulse_pending = False # NEW: Flag to trigger the corrective pulse
        self._pulse_steps_left = 0 # NEW: Counter for sustained pulse
        
        self._prev_contact_energy = 0.0 # Previous step's accumulated contact energy
        self._was_in_caution_zone = False # Tracks if the agent exceeded 0.02J during contact
        self._disengage_steps_left = 0 # Counter for temporary penalty reduction after disengage
        
        # Cache MuJoCo IDs for contact detection
        arm_geom_names = ["link0_geom", "link1_geom", "fingertip_geom"]
        self.arm_geom_ids = [self.model.geom(name).id for name in arm_geom_names]
        self.box_geom_id = self.model.geom("box_geom").id
        self.box_body_id = self.model.body("box_body").id 

        # Progress shaping state
        self.prev_dist = None

        # For stable logging in delay window
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
        
        epE = np.array([self.episode_energy], dtype=np.float64)
        # Returns 10 state features + 1 energy feature
        return np.concatenate([qpos, qvel, fingertip_pos, target_pos, epE])

    def _get_normalized_obs(self):
        raw = self._build_raw_obs()
        # Normalizing the first 10 state features
        norm_fixed = self._minmax_to_minus1_1(raw[:10], self._feat_low_fixed, self._feat_high_fixed)
        E = float(raw[10]) # Energy is at index 10
        
        # dynamic min–max for the energy channel (observability only)
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
        # Arm Start pose (first 2 qpos features)
        fixed_arm_qpos = np.array([0.0, 0.0], dtype=np.float64)
        
        rng = getattr(self, "np_random", np.random)
        
        # Random Target on donut (r ∈ [0.2, 0.9])
        inner_radius = 0.2
        outer_radius = 0.9
        r2 = rng.uniform(inner_radius**2, outer_radius**2)
        r = np.sqrt(r2)
        theta = rng.uniform(0, 2*np.pi)
        offset = r * np.array([np.cos(theta), np.sin(theta)])
        target_pos = np.array([offset[0], offset[1], 0.1])
        self.model.site_pos[self.model.site("target").id][:] = target_pos

        # NEW: Random Box Position (Box qpos is 7 features: 3 pos, 4 quat)
        box_inner_radius = 0.15 
        box_outer_radius = 0.8
        box_r2 = rng.uniform(box_inner_radius**2, box_outer_radius**2)
        box_r = np.sqrt(box_r2)
        box_theta = rng.uniform(0, 2*np.pi)
        box_offset = box_r * np.array([np.cos(box_theta), np.sin(box_theta)])
        
        box_qpos = np.zeros(7, dtype=np.float64)
        box_qpos[0] = box_offset[0] # x position
        box_qpos[1] = box_offset[1] # y position
        box_qpos[2] = 0.05 # z position
        box_qpos[3] = 1.0 # w component of identity quaternion

        # The full qpos is arm_qpos (2) + box_qpos (7) = 9
        full_qpos = np.concatenate([fixed_arm_qpos, box_qpos])
        full_qvel = np.zeros(self.model.nv, dtype=np.float64) 
        
        self.set_state(full_qpos, full_qvel)

        self.do_simulation(np.zeros(self.model.nu), 1)

        # Reset trackers
        self.episode_energy = 0.0
        self.episode_contact_energy = 0.0 
        self._E_min, self._E_max = -1.0, 1.0

        self._prev_qpos = self.data.qpos[:2].copy()
        self._prev_tau = np.zeros(2, dtype=np.float64)

        # initialize prev_dist for progress shaping
        fingertip_pos = self.data.site_xpos[self.model.site("fingertip_site").id]
        target_pos = self.model.site("target").pos
        self.prev_dist = float(np.linalg.norm(fingertip_pos - target_pos))

        # reset delay window state
        self.current_delay = 0
        self.delay_pending_termination = False

        # reset cached logging values
        self._last_act_penalty = 0.0
        self._last_progress = 0.0
        
        # NEW: Reset safety trackers for the start of the episode
        self._prev_contact_energy = 0.0
        self._was_in_caution_zone = False 
        self._disengage_steps_left = 0

        return self.get_obs()

    def step(self, action):
        # success delay window (unchanged)
        if self.current_delay > 0:
            self.current_delay -= 1
            fingertip_pos = self.data.site_xpos[self.model.site("fingertip_site").id]
            target_pos = self.model.site("target").pos
            dist = float(np.linalg.norm(fingertip_pos - target_pos))

            reward = 0.0
            terminated = False
            truncated = False
            info = {
                "delay": True,
                "distance": dist,
                "success": dist < 0.05,
                "episode_energy": self.episode_energy,
                "act_penalty": float(self._last_act_penalty),
                "progress": float(self._last_progress),
                "episode_contact_energy": self.episode_contact_energy,
                "contact_energy_limit_reached": False,
            }
            if self.current_delay == 0 and self.delay_pending_termination:
                terminated = True
                self.delay_pending_termination = False
            return self.get_obs(), reward, terminated, truncated, info

        # ------------------- CORRECTIVE PULSE OVERRIDE -------------------
        if self._pulse_steps_left > 0:
            # Override agent's action with sustained negative torque
            action = np.array([-3.0, -2.0], dtype=np.float64) 
            self._pulse_steps_left -= 1 # Decrement the counter
        # -----------------------------------------------------------------

        # Step simulation (applies the original action or the corrective pulse)
        self.do_simulation(action, self.frame_skip)

        # --- INITIALIZATION FOR MAIN STEP ---
        is_contact = False
        box_force_magnitude = 0.0
        contact_energy_inc = 0.0 # Step-wise energy increase (for logging)
        # --- END INITIALIZATION ---

        # -------------------- Contact and Safety Check --------------------
        
        # 1. Check for contact between arm and box
        for contact in self.data.contact:
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            arm_to_box_contact = (geom1_id in self.arm_geom_ids and geom2_id == self.box_geom_id) or \
                                 (geom2_id in self.arm_geom_ids and geom1_id == self.box_geom_id)
                                 
            if arm_to_box_contact:
                is_contact = True
                break
                
        # 2. Track contact-specific energy if contact occurs
        if is_contact:
            box_cfrc_ext = self.data.cfrc_ext[self.box_body_id]
            force_magnitude = np.linalg.norm(box_cfrc_ext[:3])
            box_force_magnitude = float(force_magnitude)
            
            contact_energy_inc = box_force_magnitude * self.model.opt.timestep * self.frame_skip
            self.episode_contact_energy += contact_energy_inc
        
        # 3. Check for First Breach of Caution Zone and set Pulse Flag
        # Trigger pulse only if E_contact crosses 0.02J and the flag hasn't been set yet (for persistent tracking)
        if self.episode_contact_energy >= self.contact_warning_limit and not self._was_in_caution_zone:
            # CAUTION ZONE BREACH DETECTED for the first time in this contact cycle
            self._was_in_caution_zone = True # Set flag for persistent reward/reset
            
            # TRIGGER THE CORRECTIVE PULSE FOR THE NEXT STEP
            self._pulse_steps_left = 10

        if not is_contact and self.episode_contact_energy > 1e-6:
            self.episode_contact_energy = 0.0
            self._prev_contact_energy = 0.0
        
            
        # ------------------------------------------------------------------

        # Mechanical work increment
        qpos_now = self.data.qpos[:2].copy()
        dq = qpos_now - (self._prev_qpos if self._prev_qpos is not None else qpos_now)
        tau_prev = self._prev_tau if self._prev_tau is not None else np.zeros(2, dtype=np.float64)
        energy_inc = float(np.dot(tau_prev, dq))
        self.episode_energy += energy_inc

        # Prepare next history tau_k
        self._prev_qpos = qpos_now
        self._prev_tau = self.data.qfrc_actuator[:2].copy()

        # Read state
        qvel = self.data.qvel[:2]
        fingertip_pos = self.data.site_xpos[self.model.site("fingertip_site").id]
        target_pos = self.model.site("target").pos
        dist = float(np.linalg.norm(fingertip_pos - target_pos))

        # -------------------- Dynamic Contact Energy Shaping (Strictly Conditional) --------------------
        contact_shaping_reward = 0.0
        # Positive if energy increased (pushing), negative if decreased (retreating)
        delta_contact_energy = self.episode_contact_energy - self._prev_contact_energy 

        # This logic governs the high-penalty/high-reward zone (E > 0.02 J)
        if self.episode_contact_energy > 1e-7: # Only run logic if there is contact energy > 0
            
            # Check if the energy is INCREASING (Pushing)
            if delta_contact_energy > 0:
                
                # APPLY HEAVY PENALTY ONLY IF CURRENT ENERGY IS ABOVE 0.02 J
                if self.episode_contact_energy > self.contact_warning_limit:
                    contact_shaping_reward = -75.0 * delta_contact_energy  # Pushing Penalty
            
            # Check if the energy is DECREASING (Retreating/Disengaging)
            elif delta_contact_energy < 0:
                
                # APPLY PERSISTENT REWARD IF THE CAUTION FLAG IS SET (meaning it has breached 0.02 J)
                if self._was_in_caution_zone:
                    # MAXIMIZED Retreat Reward
                    contact_shaping_reward = -150.0 * delta_contact_energy 
        
        # 3. Update history for the next step's calculation
        self._prev_contact_energy = self.episode_contact_energy
        # -------------------------------------------------------------------------------------------
        
        # ----------------- START VIRTUAL RESET WINDOW (Temp Penalty Reduction) -----------------
        temp_vel_penalty_boost = 0.0
        temp_act_penalty_boost = 0.0

        if self._disengage_steps_left > 0:
            # Set to 1.0 to eliminate penalties (1.0 - 1.0 = 0)
            temp_vel_penalty_boost = 1.0 
            temp_act_penalty_boost = 1.0
            self._disengage_steps_left -= 1 
        # ----------------- END VIRTUAL RESET WINDOW -----------------

        # ===================== reward (progress shaping; literals here) =====================
        progress= float(self.prev_dist - dist)
        time_penalty = 0.0001
        
        # SQUARED Velocity Penalty (with temporary reduction)
        vel_penalty_coeff = 0.001 * (1.0 - temp_vel_penalty_boost) 
        vel_penalty = vel_penalty_coeff * float(np.dot(qvel, qvel))
        
        # Action Penalty (with temporary reduction)
        act_penalty_coeff = 0.01 * (1.0 - temp_act_penalty_boost) 
        a = np.asarray(action, dtype=np.float64)
        act_penalty = act_penalty_coeff * float(np.dot(a, a))
        
        energy_cost = 0.05 * abs(energy_inc)
        
        # Base Collision penalty proportional to the force exerted on the box
        collision_penalty = 0.0
        if is_contact:
            collision_penalty = 1.0 * box_force_magnitude 

            # Inside step:
        if dist < 0.1:
            # Reduce movement costs by 95%
            vel_penalty *= 0.00
            act_penalty *= 0.00
            energy_cost *= 0.00
            

        # Final Reward calculation
        reward = progress - energy_cost - time_penalty - vel_penalty - act_penalty \
                 - collision_penalty + contact_shaping_reward
        
        # success reward + start delay window
        success = dist < 0.05
        if success:
            reward += 10.0
            if dist < 0.02:
                reward += 5.0
                
                
        
            # REWARD FOR SAFE SUCCESS: Bonus for success without entering caution zone (0.02J)
            if self.episode_contact_energy < self.contact_warning_limit: 
                reward += 5.0 
                
            self.current_delay = self.success_delay_steps
            self.delay_pending_termination = True
        # =====================================================================

        # ----------------- Virtual Reset Trigger (Pathfinding Reset) -----------------
        # The Reset MUST only trigger when E_contact returns to zero AND the agent was previously cautious.
        if self.episode_contact_energy < 1e-6: # Check if energy is effectively zero
            
            if self._was_in_caution_zone:
                # SUCCESSFUL RETREAT: Reset progress and start exploration window
                self.prev_dist = dist# Reset progress state
                self._disengage_steps_left = 40  # Start the exploration window (5 steps)
            
            # Reset the caution flag ONLY when energy hits zero, clearing the state for the next collision
            self._was_in_caution_zone = False 
            
        # ----------------- End Virtual Reset Trigger -----------------

        # update progress state and cached logging values (standard update)
        self.prev_dist = dist 
        self._last_act_penalty = act_penalty
        self._last_progress = progress

        # -------------------- Truncation Check (0.06 J) --------------------
        terminated = False
        truncated = False
        contact_energy_limit_reached = False
        
        if self.episode_contact_energy > self.contact_energy_limit: # 0.06 J
            truncated = True
            contact_energy_limit_reached = True
            self.current_delay = 0 

        # -------------------------------------------------------------------
        
        info = {
            "distance": dist,
            "success": success,
            "progress": progress,
            "energy_inc": energy_inc,
            "episode_energy": self.episode_energy,
            "act_penalty": act_penalty,
            # NEW INFO LOGGING
            "contact": is_contact,
            "box_force": box_force_magnitude,
            "collision_penalty": collision_penalty,
            "episode_contact_energy": self.episode_contact_energy,
            "contact_energy_limit_reached": contact_energy_limit_reached,
            "contact_energy_inc": contact_energy_inc, # Logging the step-wise change
        }

        return self.get_obs(), reward, terminated, truncated, info
    
    def render(self):
        return super().render()

    def close(self):
        return super().close()