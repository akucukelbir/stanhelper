# Stanhelper

Stanhelper is a lightweight wrapper around [cmdStan](http://mc-stan.org).

## Installation

To install from pip, run
```{bash}
pip install stanhelper
```

## Development

To begin:
```{bash}
(venv)$ pip install pip-tools
```

To keep environment updated:
```{bash}
(venv)$ pip-compile --output-file requirements.txt requirements.in
(venv)$ pip-sync
```

To install stanhelper in develop mode:
```{bash}
(venv)$ python setup.py develop
```

## Testing

Make sure you compile the Stan files under `tests` first. Then run
```{bash}
(venv)$ pytest --flake8
```
