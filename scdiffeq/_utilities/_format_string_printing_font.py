

def _pystring_format_effects(key=None):

    _pystring_format_effects_dict = {
        "PURPLE": "\033[95m",
        "CYAN": "\033[96m",
        "DARKCYAN": "\033[36m",
        "BLUE": "\033[94m",
        "GREEN": "\033[92m",
        "YELLOW": "\033[93m",
        "RED": "\033[91m",
        "BOLD": "\033[1m",
        "UNDERLINE": "\033[4m",
        "END": "\033[0m",
    }

    if key != None and key in _pystring_format_effects_dict.keys():
        effect = _pystring_format_effects_dict[key]
        return effect
    else:
        print(
            "\033[91m\033[1m{}\033[0m is not a valid effect.\n\nPlease choose from:\n\n{}".format(
                key, [_key for _key in _pystring_format_effects_dict.keys()]
            )
        )


def _format_string_printing_font(string, formatting_effect=None):

    """
    Function formats and returns a string using ANSI color codes.
    Parameters:
    -----------
    string (required)
    formatting_effect (optional)
        default: None
        type:    list
    Returns:
    --------
    formatted_string
        applies desired formatting to the printed string
    """

    if type(formatting_effect) != list:
        formatting_effect = [formatting_effect]

    formatting_effects = "".join(
        [_pystring_format_effects(formatting) for formatting in formatting_effect]
    )
    formatted_string = formatting_effects + string + _pystring_format_effects("END")

    return formatted_string