import re

# Map types to (regex_pattern, converter)

TYPE_PATTERNS = {
    "int":   (r"\(-?\d+\)", lambda x: int(x[1:-1])),
    "bool":  (r"\((true)|(false)\)", lambda x: x[1:-1] == 'true'),
    "float": (r"\(-?(?:\d+(?:\.\d*)?|\.\d+)\)", lambda x: float(x[1:-1])),
    "str":   (r"\([^\)*]\)", lambda x: str(x[1:-1])),  # non-greedy: will stop at next literal in the template
}

def compile_template(template: str) -> re.Pattern:
    """
    Convert a template into matching regex.
    """
    pattern_parts = []
    converters = {}
    i = 0
    L = len(template)

    while i < L:
        if template[i] == '{':
            j = template.index('}', i)
            inside = template[i+1:j]
            if ':' not in inside:
                name = inside.strip()
                type_name = 'str'
            else:
                name, type_name = inside.split(':', 1)
                name = name.strip()
                type_name = type_name.strip()

            if type_name not in TYPE_PATTERNS:
                raise ValueError(f"Unknown type {type_name!r} in {{{inside}}}")

            type_pattern, converter = TYPE_PATTERNS[type_name]
            pattern_parts.append(fr"(?P<_{name}>{type_pattern})")
            converters[name] = converter

            i = j + 1
        else:
            # collect a run of literal characters
            start = i
            while i < L and template[i] != '{':
                i += 1
            pattern_parts.append(re.escape(template[start:i]))

    regex = re.compile("^" + "".join(pattern_parts) + "$")
    return regex, converters


def match_template(template: str, text: str):
    regex, converters = compile_template(template)
    m = regex.match(text)
    if not m:
        return None

    result = {}
    for name, value in m.groupdict().items():
        name = name[1:]
        conv = converters.get(name, str)
        result[name] = conv(value)
    return result

