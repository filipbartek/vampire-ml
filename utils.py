#!/usr/bin/env python3.7


def option_value(options, option_name, default=None):
    try:
        option_index = list(reversed(options)).index(option_name)
    except ValueError:
        return default
    if option_index == 0:
        # The option name was the last argument.
        return default
    assert option_index >= 1
    return options[-option_index]
