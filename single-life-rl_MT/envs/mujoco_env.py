from collections import OrderedDict
import os
from typing import Optional

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e
        )
    )

DEFAULT_SIZE = 128


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(
            OrderedDict(
                [
                    (key, convert_observation_to_space(value))
                    for key, value in observation.items()
                ]
            )
        )
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float("inf"), dtype=np.float32)
        high = np.full(observation.shape, float("inf"), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments."""

    def __init__(self, model_path, frame_skip):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
            print("fullpath to xml", fullpath)
        if not path.exists(fullpath):
            raise OSError(f"File {fullpath} does not exist")
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            "render.modes": ["human", "rgb_array", "depth_array"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self._set_action_space()

        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        # pass  
        self.viewer.cam.azimuth = 15
        self.viewer.cam.distance = 0.75
        self.viewer.cam.elevation = -22.5
        self.viewer.cam.fixedcamid = -1
        # self.viewer.cam.lookat =  [3.80046184e-06, 0.00000000e+00, 2.00000000e-01]
        self.viewer.cam.type = 0
        print('Here')
        # ()+1     
    # -----------------------------

    def reset(self, seed: Optional[int] = None):
        super().reset(seed=seed)
        self.sim.reset()
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(
            old_state.time, qpos, qvel, old_state.act, old_state.udd_state
        )
        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        if np.array(ctrl).shape != self.action_space.shape:
            print(self.action_space.shape)
            raise ValueError("Action dimension mismatch")

        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def render(
        self,
        mode="rgb_array",
        width=512,
        height=512,
        camera_id=None,
        camera_name=None,
    ):
        
        if mode == "human":
            print('MODE')
            self.viewer = mujoco_py.MjViewer(self.sim)
            # self.viewer.render()
            return 

        # if mode == "rgb_array":
            # window size used for old mujoco-py:
        self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)
        self.viewer_setup()

        # data = self._get_viewer(mode).read_pixels(width, height, depth=True)[0]
        self.viewer.render(width = 640, height = 480)
        # print(dir(self._get_viewer(mode)))
        # return data
        data = self.viewer.read_pixels(width, height, depth=False)
        return data[::-1,:]
            
            
            # exit()
            # original image is upside-down, so flip it
            # return data[::-1, :, :]
        # elif mode == "depth_array":
        #     self._get_viewer(mode).render(width, height)
        #     # window size used for old mujoco-py:
        #     # Extract depth part of the read_pixels() tuple
        #     data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
        #     # original image is upside-down, so flip it
        #     return data[::-1, :]
        # elif mode == "human":
        #     self._get_viewer(mode).render()
        #     return self._get_viewer(mode).read_pixels(width, height, depth=False)

    def close(self):
        if self.viewer is not None:
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode, camera_id = -1):
        # print('whatisthis')
        # print(self.viewer)
        # self.viewer = self._viewers.get(mode)
        self.viewer = None
        if self.viewer is None:
            if mode == "human":
                print('MODE')
                self.viewer = mujoco_py.MjViewer(self.sim)

            elif mode == "rgb_array" or mode == "depth_array":
                # print('This is it')
                # print(camera_id)
                
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)
                # print(dir(self.viewer.cam))
                
                # camera = self.viewer.cam
                # print(camera.azimuth, camera.distance, camera.elevation, camera.fixedcamid, camera.lookat, camera.type, camera.trackbodyid)
                # self.viewer = mujoco_py.MjViewer(self.sim)
                # ()+1
            
                # self.viewer_setup()
                self.viewer.cam.azimuth = 0
                camera = self.viewer.cam
                print(camera.azimuth, camera.distance, camera.elevation, camera.fixedcamid, camera.lookat, camera.type, camera.trackbodyid)
                
                
                
                # ()+1
            self._viewers[mode] = self.viewer
            
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def state_vector(self):
        return np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat])
