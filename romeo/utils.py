# Import seaborn
import re

import seaborn as sns

# Load an example dataset
data = sns.load_dataset("iris")


formula = "species ~ sepal_length + sepal_width + petal_length + petal_width + (1 | type) + (1|store)"

formula_fmt = [x.replace(" ", "") for x in formula.split("~")]

lhs = formula_fmt[0]  # left hand side equation
rhs = formula_fmt[1]  # right hand side equation

fixed_effect = []
random_effect = []

for elem in rhs.split("+"):
    if re.compile("\((.+)\)").search(elem) is not None:
        random_effect.append(re.match("\((.+)\)", elem).group(0))
    else:
        fixed_effect.append(elem)

print(fixed_effect)
print(random_effect)
