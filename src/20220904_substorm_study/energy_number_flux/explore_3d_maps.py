# Source - https://stackoverflow.com/a/23914781
# Posted by pelson, modified by community. See post 'Timeline' for change history
# Retrieved 2026-09-15, License - CC BY-SA 3.0

# TODO: Google "mplot3d plot change animate view angle rotation"

import itertools

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.path import Path
import numpy as np

import cartopy.feature
from cartopy.mpl.path import shapely_to_path
import cartopy.crs as ccrs


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', xlim=[-120, -50], ylim=[45, 90])
ax.set_zlim(bottom=0)


concat = lambda iterable: list(itertools.chain.from_iterable(iterable))


def to_path_list(path_or_paths):
    # cartopy.shapely_to_path may return one Path or an iterable of Paths.
    if isinstance(path_or_paths, Path):
        return [path_or_paths]
    return list(path_or_paths)

target_projection = ccrs.PlateCarree()

feature = cartopy.feature.NaturalEarthFeature('physical', 'land', '110m')
geoms = feature.geometries()

geoms = [target_projection.project_geometry(geom, feature.crs)
         for geom in geoms]

paths = concat(to_path_list(shapely_to_path(geom)) for geom in geoms)

polys = concat(path.to_polygons() for path in paths)

lc = PolyCollection(polys, edgecolor='black',
                    facecolor='green', closed=False)

ax.add_collection3d(lc)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Height')

# plt.show()

# Source - https://stackoverflow.com/a/48298662
# Posted by pelson
# Retrieved 2026-09-15, License - CC BY-SA 3.0

# import cartopy.crs as ccrs
# import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import numpy as np


def f(x,y):
    x, y = np.meshgrid(x, y)
    return (1 - x / 2 + x**5 + y**3 + x*y**2) * np.exp(-x**2 -y**2)

nx, ny = 256, 512
X = np.linspace(-99, -101, nx)
Y = np.linspace(45, 90, ny)
Z = f(np.linspace(0, 100, nx), np.linspace(0, 100, ny))

# Rendering every cell in 3D is expensive; stride controls the speed/detail tradeoff.
stride = 4
X_plot = X[::stride]
Y_plot = Y[::stride]
Z_plot = Z[::stride, ::stride]


# fig = plt.figure()
# ax3d = fig.add_axes([0, 0, 1, 1], projection='3d')

# Make an axes that we can use for mapping the data in 2d.
proj_ax = plt.figure().add_axes([0, 0, 1, 1], projection=ccrs.Mercator())
XX, YY = np.meshgrid(X_plot, Y_plot)
qm = proj_ax.pcolormesh(XX, YY, Z_plot,  # Or use plot_surface
                        transform=ccrs.PlateCarree(),
                        shading='auto',
                        cmap='viridis',
                        alpha=0.3)

# Force facecolor expansion so each pcolormesh cell has its own RGBA color.
proj_ax.figure.canvas.draw()

trans_to_proj = qm.get_transform() - proj_ax.transData
paths = qm.get_paths()
facecolors = qm.get_facecolors()
zvals = np.ravel(Z_plot)

verts3d_all = []
facecolors_all = []

for idx, path in enumerate(paths):
    projected_path = trans_to_proj.transform_path(path)
    verts = projected_path.vertices
    if verts.shape[0] < 4:
        continue

    z = float(zvals[idx])
    verts3d = np.column_stack([verts[:, 0],
                               verts[:, 1],
                               np.full(verts.shape[0], z)])

    verts3d_all.append(verts3d)
    facecolors_all.append(facecolors[idx])

pc = Poly3DCollection(verts3d_all, linewidths=0)
pc.set_facecolor(facecolors_all)
ax.add_collection3d(pc)

ax.set_zlim(float(np.nanmin(Z)), float(np.nanmax(Z)))

proj_ax.autoscale_view()

ax.set_xlim(*proj_ax.get_xlim())
ax.set_ylim(*proj_ax.get_ylim())
ax.set_zlim(Z.min(), Z.max())


plt.close(proj_ax.figure)
plt.show()
