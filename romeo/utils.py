import re


def formula_parser(formula: str):
    r"""Extrapolate the target, fixed and random effect terms.

     Parameters
     ----------
     formula: str
         Dataframe with variables in columns.

     Returns
     -------
     dict : dict
         dictionary containing info necessary to construct problem formulation

    Examples
     ----------
     >>> formula = "y ~  x1 + x2 + x3 + (1 | c1) + (1 | c2)"
     >>> dict_formula = formula_parser(formula=formula)

    """

    # split between DV and fixed + random effect
    formula_fmt = [x.replace(" ", "") for x in formula.split("~")]
    lhs = formula_fmt[0]  # left hand side equation
    rhs = formula_fmt[1]  # right hand side equation

    # from rhs discern between fixed and random effect
    fixed_effect = []
    random_effect = []

    for elem in rhs.split("+"):
        # check is elem has is contained in parenthesis "()"
        if re.compile(r"\((.+)\)").search(elem) is not None:
            # append random effect element in the form "(1|group)"
            random_effect.append(re.match(r"\((.+)\)", elem).group(0))
        else:
            # append fixed effect element in the form
            fixed_effect.append(elem)

    clusters = [re.match(r"^.*\|(.*)\).*$", x).group(1) for x in random_effect]

    return {
        "target": lhs,
        "fixed_effect": fixed_effect,
        "random_effect": random_effect,
        "clusters": clusters,
    }
