# Stanhelper

Stanhelper is a lightweight wrapper around [cmdStan](http://mc-stan.org).

## Installation

To install from pip, run
```{bash}
pip install stanhelper
```

## Testing

Make sure you compile the Stan files under `tests` first. Then run
```{bash}
pytest tests
pytest --pep8
```
