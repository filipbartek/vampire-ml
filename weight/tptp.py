import os
import re


file_name_pattern = r'^(?P<name>(?P<domain>[A-Z]{3})(?P<number>[0-9]{3})(?P<form_symbol>[-+^=_])(?P<version>[1-9])(?P<size_parameters>[0-9]*(\.[0-9]{3})*))(?:\.[pg])?$'


def problem_path(name, tptp_path):
    path = name
    m = re.match(file_name_pattern, name)
    if m:
        return os.path.join(tptp_path, 'Problems', m['domain'], m['name'] + '.p')
    return path
