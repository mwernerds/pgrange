import numpy as np;
import pgrange;
import matplotlib.pyplot as plt;

# Query
q = pgrange.performQuery(
    "dbname = points user = postgres password = postgres hostaddr = 127.0.0.1 port = 5432",
    545390.445795515,
    5800605.7953092,
    64,
    4,
    32632)

# print the output numpy array of your query
print(q)

# reshape the data, so that zmin, zavg, zmax values are in the same dimension
q = q.reshape(128, 128, 3)

# stack zmin, zavg, zmax into one image
image = np.hstack([q[..., 0], q[..., 1], q[..., 2]])

# save the result as .png image
plt.imshow(image);
plt.savefig("out.png",dpi=600);
