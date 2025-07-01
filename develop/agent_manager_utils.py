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
    def _get_policy(self, obj):
        policy = MacroTrajPolicy(obj, self.generate_seed())

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