from metadrive.constants import DEFAULT_AGENT
from metadrive.engine.logger import get_logger
from metadrive.manager.base_manager import BaseAgentManager
from metadrive.policy.AI_protect_policy import AIProtectPolicy
from metadrive.policy.idm_policy import TrajectoryIDMPolicy
from metadrive.policy.manual_control_policy import ManualControlPolicy, TakeoverPolicy, TakeoverPolicyWithoutBrake
from metadrive.policy.replay_policy import ReplayTrafficParticipantPolicy

from metadrive.manager.agent_manager import VehicleAgentManager
from develop.policy_utils import MacroTrajPolicy
from develop.vehicle_utils import MacroDefaultVehicle



class MacroAgentManager(VehicleAgentManager):
    def before_step(self, frame = 0, wps=None):
        step_infos = dict()
        for agent_id in self.active_agents.keys():
            policy = self.get_policy(self._agent_to_object[agent_id])
            if agent_id in wps.keys():
                waypoints = wps[agent_id]
            #self.get_agent(agent_id).before_macro_step(frame)
            action = policy.act(agent_id, frame, waypoints)
            #action = policy.act(agent_id)
            step_infos[agent_id] = policy.get_action_info()
            step_infos[agent_id].update(self.get_agent(agent_id).before_step(action))
        self._agents_finished_this_frame = dict()
        finished = set()
        for v_name in self._dying_objects.keys():
            self._dying_objects[v_name][1] -= 1
            if self._dying_objects[v_name][1] <= 0:  # Countdown goes to 0, it's time to remove the vehicles!
                v = self._dying_objects[v_name][0]
                self._remove_vehicle(v)
                finished.add(v_name)
        for v_name in finished:
            self._dying_objects.pop(v_name)
        return step_infos

    @property
    def agent_policy(self):
        """Get the agent policy class

        Make sure you access the global config via get_global_config() instead of self.engine.global_config

        Returns:
            Agent Policy class
        """
        # from metadrive.engine.engine_utils import get_global_config
        # # Takeover policy shares the control between RL agent (whose action is input via env.step)
        # # and external control device (whose action is input via controller).
        # if get_global_config()["agent_policy"] in [TakeoverPolicy, TakeoverPolicyWithoutBrake]:
        #     return get_global_config()["agent_policy"]
        # if get_global_config()["manual_control"]:
        #     if get_global_config().get("use_AI_protector", False):
        #         policy = AIProtectPolicy
        #     else:
        #         policy = ManualControlPolicy
        # else:
        #     policy = get_global_config()["agent_policy"]
        policy = MacroTrajPolicy
        return policy

    def _create_agents(self, config_dict: dict):
        from metadrive.component.vehicle.vehicle_type import random_vehicle_type, vehicle_type
        ret = {}
        v_type = MacroDefaultVehicle
        for agent_id, v_config in config_dict.items():
            # v_type = random_vehicle_type(self.np_random) if self.engine.global_config["random_agent_model"] else \
            #     vehicle_type[v_config["vehicle_model"] if v_config.get("vehicle_model", False) else "default"]

            obj_name = agent_id if self.engine.global_config["force_reuse_object_name"] else None
            obj = self.spawn_object(v_type, vehicle_config=v_config, name=obj_name)
            ret[agent_id] = obj
            policy_cls = self.agent_policy
            args = [obj, self.generate_seed()]
            if policy_cls == TrajectoryIDMPolicy or issubclass(policy_cls, TrajectoryIDMPolicy):
                args.append(self.engine.map_manager.current_sdc_route)
            self.add_policy(obj.id, policy_cls, *args)
        return ret