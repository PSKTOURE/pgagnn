"""Exercise the CRIANN launcher with local stand-ins for Slurm commands."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def executable(path, content):
    path.write_text(content)
    path.chmod(0o755)


@pytest.fixture
def cluster(tmp_path):
    project = tmp_path / 'project with spaces'
    (project / 'src').mkdir(parents=True)
    (project / 'src/version.txt').write_text('original')
    (project / 'experiments/qm9').mkdir(parents=True)
    for name in ['train_qm9.sl', 'submit_criann.sh']:
        shutil.copy(ROOT / 'experiments/qm9' / name, project / 'experiments/qm9' / name)
    for name in ['pyproject.toml', 'uv.lock']:
        (project / name).write_text('fixture')
    data = tmp_path / 'dataset'
    (data / 'processed').mkdir(parents=True)
    (data / 'processed/data.pt').write_text('fixture')
    commands = tmp_path / 'commands'
    commands.mkdir()
    executable(commands / 'module', '#!/bin/bash\nexit 0\n')
    executable(commands / 'srun', '#!/bin/bash\nexec "$@"\n')
    executable(commands / 'scontrol', '#!/bin/bash\nprintf "%s\\n" "$*" >> "$REQUEUE_LOG"\nexit "${REQUEUE_STATUS:-0}"\n')
    executable(commands / 'sbatch', '#!/usr/bin/env python3\nimport json,os,sys\nopen(os.environ["SUBMIT_LOG"],"w").write(json.dumps(sys.argv[1:]))\n')
    python = commands / 'training-python'
    executable(python, '''#!/usr/bin/env python3
import json,os,pathlib,sys
args=sys.argv[1:]
def value(flag): return args[args.index(flag)+1]
path=pathlib.Path(value('--runs_dir'))/value('--target')/value('--model_name')
path.mkdir(parents=True,exist_ok=True)
(path/'last_model.pt').write_text('completed epoch fixture')
open(os.environ['ARGUMENT_LOG'],'w').write(json.dumps(args))
raise SystemExit(int(os.environ.get('TRAIN_STATUS','0')))
''')
    env = dict(os.environ, PATH=str(commands) + os.pathsep + os.environ['PATH'],
               QM9_PROJECT_DIR=str(project), SLURM_SUBMIT_DIR=str(project),
               SLURM_JOB_ID='12345', QM9_PYTHON=str(python), QM9_DATA_DIR=str(data),
               QM9_SCRATCH_ROOT=str(tmp_path / 'scratch'),
               REQUEUE_LOG=str(tmp_path / 'requeue.log'),
               SUBMIT_LOG=str(tmp_path / 'submit.json'), ARGUMENT_LOG=str(tmp_path / 'args.json'))
    return project, env


def launch(project, env, target='all'):
    return subprocess.run(['bash', str(project / 'experiments/qm9/train_qm9.sl'),
                           'pgagnn', 'test_run', target, '3'], env=env, text=True,
                          capture_output=True, timeout=30)


def test_array_requeue_and_resume_use_persistent_checkpoint_and_frozen_source(cluster):
    project, env = cluster
    env.update(SLURM_ARRAY_JOB_ID='12300', SLURM_ARRAY_TASK_ID='7', TRAIN_STATUS='3')
    result = launch(project, env)
    assert result.returncode == 0, result.stderr
    assert Path(env['REQUEUE_LOG']).read_text().strip() == 'requeue 12300_7'
    args = json.loads(Path(env['ARGUMENT_LOG']).read_text())
    assert args[args.index('--target') + 1] == 'U0'
    assert args[args.index('--weight_decay') + 1] == '5e-7'
    assert args[args.index('--checkpoint_interval') + 1] == '1'
    expected = project / 'runs/qm9/test_run/U0/pgagnn/last_model.pt'
    assert expected.is_file()
    assert '--resume_from' not in args
    (project / 'src/version.txt').write_text('edited after first launch')
    env.update(TRAIN_STATUS='0', SLURM_RESTART_COUNT='1')
    result = launch(project, env)
    assert result.returncode == 0, result.stderr
    args = json.loads(Path(env['ARGUMENT_LOG']).read_text())
    assert args[args.index('--resume_from') + 1] == str(expected)
    assert (project / 'runs/qm9/test_run/source/src/version.txt').read_text() == 'original'
    assert Path(env['REQUEUE_LOG']).read_text().count('requeue') == 1


def test_requeue_failure_is_not_reported_as_success(cluster):
    project, env = cluster
    env.update(TRAIN_STATUS='3', REQUEUE_STATUS='19')
    result = launch(project, env, target='mu')
    assert result.returncode == 19
    assert Path(env['REQUEUE_LOG']).read_text().strip() == 'requeue 12345'


def test_training_failure_does_not_requeue(cluster):
    project, env = cluster
    env['TRAIN_STATUS'] = '7'
    assert launch(project, env, target='Cv').returncode == 7
    assert not Path(env['REQUEUE_LOG']).exists()


def test_submit_all_targets_uses_bounded_array(cluster):
    project, env = cluster
    result = subprocess.run(['bash', str(project / 'experiments/qm9/submit_criann.sh'),
                             'test_run', 'all', '3'], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    args = json.loads(Path(env['SUBMIT_LOG']).read_text())
    assert args == ['--array=0-11%3', 'experiments/qm9/train_qm9.sl', 'pgagnn', 'test_run', 'all', '3']
    assert (project / 'slurm/qm9').is_dir()
