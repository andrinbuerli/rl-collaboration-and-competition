import json
import torch
import numpy as np
import sys

sys.path.append("../")

from lib.helper import parse_config_for, extract_config_from
from lib.env.DualParallelAgentsUnityEnvironment import DualParallelAgentsUnityEnvironment
from lib.models.policy.DeterministicDiscretePolicy import DeterministicDiscretePolicy
from lib.models.function.StateActionValueFunction import StateActionValueFunction
from lib.RLAgentTrainer import RLAgentTrainer
from lib.agent.ddpg.DiscreteMADDPGRLAgent import DiscreteMADDPGRLAgent
from lib.log.WandbLogger import WandbLogger

if __name__ == "__main__":
    print(f"Found {torch._C._cuda_getDeviceCount()} GPU")

    args = parse_config_for(
        program_name='Tennis',
        config_objects={
            "gamma": 0.99,
            "epsilon": 1.0,
            "epsilon_decay": .9995,
            "epsilon_min": 0.05,
            "buffer_size": int(1e6),
            "min_buffer_size": int(1e3),
            "batch_size": 64,
            "tau": 1e-3,
            "update_every": 1,
            "learning_rate": 0.0005,
            "update_for": 16,
            "n_iterations": int(1e8),
            "max_t": 32,
            "enable_log": 0,
            "api_key": "",
            "grad_clip_max": None,
            "seed": int(np.random.randint(0, 1e10, 1)[0]),
            "agent_weights": "agents/Soccer-DiscreteMADDPG-7750676833-latest"
        })

    env = DualParallelAgentsUnityEnvironment(
        name="Soccer",
        target_reward=2.5,
        env_binary_path='../environments/Soccer_Linux_NoVis/Soccer.x86_64')

    policy = lambda s, a: DeterministicDiscretePolicy(
        state_size=s,
        action_size=a,
        seed=int(np.random.randint(0, 1e10, 1)[0]))
    value_function = lambda s, a: StateActionValueFunction(
        state_size=s * env.num_agents,
        action_size=sum(env.action_size.values()),
        seed=int(np.random.randint(0, 1e10, 1)[0]))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    agent = DiscreteMADDPGRLAgent(
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
        action_size=list(env.action_size.values()),
        state_size=list(env.state_size.values()),
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
        wandb_project_name="udacity-drlnd-p3-soccer-maddpg-v2",
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
