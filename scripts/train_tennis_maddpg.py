import json
import torch
import numpy as np
import sys

sys.path.append("../")

from lib.helper import parse_config_for, extract_config_from
from lib.env.ParallelAgentsUnityEnvironment import ParallelAgentsUnityEnvironment
from lib.models.policy.DeterministicContinuousGaussianPolicy import DeterministicContinuousGaussianPolicy
from lib.models.function.StateActionValueFunction import StateActionValueFunction
from lib.RLAgentTrainer import RLAgentTrainer
from lib.agent.ddpg.MADDPGRLAgent import MADDPGRLAgent
from lib.log.WandbLogger import WandbLogger

if __name__ == "__main__":
    print(f"Found {torch._C._cuda_getDeviceCount()} GPU")

    args = parse_config_for(
        program_name='Tennis',
        config_objects={
            "gamma": 0.99,
            "epsilon": 1.0,
            "epsilon_decay": .999995,
            "epsilon_min": 0.05,
            "buffer_size": int(1e6),
            "min_buffer_size": int(1e4),
            "batch_size": 128,
            "tau": 1e-3,
            "update_every": 1,
            "learning_rate": 0.0001,
            "update_for": 4,
            "n_iterations": int(1e7),
            "max_t": 128,
            "enable_log": 0,
            "api_key": "",
            "grad_clip_max": None,
            "seed": int(np.random.randint(0, 1e10, 1)[0]),
            "agent_weights": None
        })

    env = ParallelAgentsUnityEnvironment(
        name="Tennis",
        target_reward=2.5,
        env_binary_path='../environments/Tennis_Linux_NoVis/Tennis.x86_64')

    policy = lambda: DeterministicContinuousGaussianPolicy(
        state_size=env.state_size, action_size=env.action_size,
        seed=int(np.random.randint(0, 1e10, 1)[0]), output_transform=lambda x: torch.tanh(x))
    value_function = lambda: StateActionValueFunction(
        state_size=env.state_size * env.num_agents,
        action_size=env.action_size * env.num_agents,
        seed=int(np.random.randint(0, 1e10, 1)[0]))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    agent = MADDPGRLAgent(
        get_actor=policy,
        get_critic=value_function,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        buffer_size=args.buffer_size,
        replay_min_size=args.min_buffer_size,
        batch_size=args.batch_size,
        tau=args.tau,
        lr=args.learning_rate,
        update_every=args.update_every,
        update_for=args.update_for,
        prioritized_exp_replay=False,
        device=device,
        action_size=env.action_size,
        state_size=env.state_size,
        grad_clip_max=args.grad_clip_max,
        n_agents=env.num_agents)

    if args.agent_weights is not None:
        agent.load(args.agent_weights)
        print(f"loaded agent weights from : {args.agent_weights}")

    if device != "cpu":
        torch.cuda.set_device(0)

    config = extract_config_from(env, policy, value_function,
                                 agent, {"n_iterations": args.n_iterations,
                                         "max_t": args.max_t,
                                         "seed": args.seed
                                         })

    print(f"initialized agent with config: \n {json.dumps(config, sort_keys=True, indent=4)}")

    logger = WandbLogger(
        wandb_project_name="udacity-drlnd-p3-tennis-maddpg-v2",
        run_name=None,
        entity="andrinburli",
        api_key=args.api_key,
        models=agent.models,
        config=config) if bool(args.enable_log) else None

    trainer = RLAgentTrainer(agent=agent, env=env, logger=logger, seed=args.seed)
    trainer.train(n_iterations=args.n_iterations, max_t=args.max_t,
                  intercept=True, t_max_episode=1024)

    env.dispose()
    logger.dispose()
