"""Re-run the reference DreamerV3 config with a different seed / step budget."""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/home/ubuntu/dreamerv3")

from functools import partial as bind  # noqa: E402

import elements  # noqa: E402
import embodied  # noqa: E402
import portal  # noqa: E402
import ruamel.yaml as yaml  # noqa: E402

from dreamerv3.main import (  # noqa: E402
    make_agent,
    make_env,
    make_logger,
    make_replay,
    make_stream,
)

seed = int(sys.argv[1])
steps = int(sys.argv[2])
logdir = sys.argv[3]

raw = yaml.YAML(typ="safe").load(
    open("/home/ubuntu/logdir/jax-dmc/walker_walk-seed0/config.yaml").read()
)
raw["seed"] = seed
raw["logdir"] = logdir
raw["run"]["steps"] = float(steps)
raw["run"]["save_every"] = 1e9
raw["run"]["log_every"] = 120
raw["jax"]["profiler"] = False
if len(sys.argv) > 4 and sys.argv[4] == "noprealloc":
    raw["jax"]["prealloc"] = False
config = elements.Config(raw)

elements.Path(logdir).mkdir()
config.save(elements.Path(logdir) / "config.yaml")


def init():
    elements.timer.global_timer.enabled = config.logger.timer


portal.setup(errfile=False, initfns=[init], ipv6=config.ipv6)

args = elements.Config(
    **config.run,
    replica=config.replica,
    replicas=config.replicas,
    logdir=config.logdir,
    batch_size=config.batch_size,
    batch_length=config.batch_length,
    report_length=config.report_length,
    consec_train=config.consec_train,
    consec_report=config.consec_report,
    replay_context=config.replay_context,
)

embodied.run.train(
    bind(make_agent, config),
    bind(make_replay, config, "replay"),
    bind(make_env, config),
    bind(make_stream, config),
    bind(make_logger, config),
    args,
)
