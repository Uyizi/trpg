"""Build a local wheelhouse from already installed dependencies; never download.
Run with the existing CPU Python containing Pydantic 2 and setuptools/wheel.
"""
import importlib.metadata as metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

PROJECT=Path(__file__).resolve().parents[1]
WHEELS=PROJECT/'outputs/debugger-wheelhouse'
ENV=PROJECT/'.venv-debugger'
DEPENDENCIES=['pydantic','pydantic-core','typing-extensions','typing-inspection','annotated-types']


def main():
    WHEELS.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'PIP_NO_INDEX':'1','PIP_DISABLE_PIP_VERSION_CHECK':'1','PYTHONDONTWRITEBYTECODE':'1'}
    for name in DEPENDENCIES:
        dist=metadata.distribution(name)
        with tempfile.TemporaryDirectory(prefix='qdebug-wheel-') as temporary:
            stage=Path(temporary)
            for item in dist.files or []:
                relative=Path(str(item))
                if relative.is_absolute() or '..' in relative.parts or '__pycache__' in relative.parts or relative.suffix=='.pyc':continue
                source=dist.locate_file(item)
                if not source.is_file():continue
                target=stage/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
            subprocess.run([sys.executable,'-m','wheel','pack',str(stage),'-d',str(WHEELS)],check=True,env=env)
    subprocess.run([sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation','--no-index',str(PROJECT),'-w',str(WHEELS)],check=True,env=env)
    if not ENV.exists():subprocess.run([sys.executable,'-m','venv',str(ENV)],check=True,env=env)
    subprocess.run([str(ENV/'bin/python'),'-m','pip','install','--force-reinstall','--no-index','--find-links',str(WHEELS),'qwen-behavior-debugger==1.0.0'],check=True,env=env)
    print('Core environment:',ENV)
    print('Local wheels only; compiled dependencies target this Python/OS, not arbitrary machines.')


if __name__=='__main__':main()
