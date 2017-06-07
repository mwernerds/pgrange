import numpy as np;
import pgrange;
import matplotlib.pyplot as plt;


q = pgrange.performQuery(
    "dbname = points user = postgres password = postgres hostaddr = 127.0.0.1 port = 5432",
    545390.445795515,
    5800605.7953092)

print(q)

means = np.array([x[0] for x in q]).reshape(128,128);
mins = np.array([x[1] for x in q]).reshape(128,128);
maxs = np.array([x[2] for x in q]).reshape(128,128);

image = np.hstack([means,mins,maxs]);
plt.imshow(image);
plt.savefig("out.png",dpi=600);
