from docplex.mp.model import Model

# Data


i = [(a, b, c, d) for a in r for b in r for c in r for d in r]

print(i)

mdl = Model(name='model')

# decision variables
x = mdl.binary_var_dict(i, name="x")
print(x)
