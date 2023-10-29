import matplotlib.pyplot as plt
import glacierml as gl
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import gaussian_kde
import math

# glac = gl.load_training_data(RGI_input = 'y')
# arch = gl.list_architectures(parameterization = '4')
# df = pd.read_pickle('quick_pick2')
# pr_list = []
# for i in range(0,25,1):
#     pr = str(i)
#     pr_list.append(pr)
    
# boot_var = df[pr_list].var(axis = 1)
# df = pd.DataFrame()
# for architecture in tqdm(arch['layer architecture'].unique()):
# #     print(architecture)
#     df_glob = gl.load_global_predictions(parameterization = '4', architecture = architecture)
#     df = pd.concat([df, df_glob])
    
# dfr = df[[
#         'RGIId','0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
#         '11','12','13','14','15','16','17','18','19','20','21',
#         '22','23','24',
# ]]

# glathida_estimates = pd.merge(glac, dfr, how = 'inner', on = 'RGIId')


print('assembling data...')
# df = pd.read_hdf(
#     'predicted_thicknesses/compiled_raw_' + 'first' + '_' + '4' + '.h5',
#     key = 'compiled_raw', mode = 'a'
# )


df = pd.read_pickle('quick_pick2')


def find_glacier_resid(df):

    dfr = pd.DataFrame()
    for i in (range(0,25,1)):
        x = pd.DataFrame(
                pd.Series(
                    df[str(i)] - df['Thickness'],
                    name = 'Residual'
            )
        )
        y = pd.DataFrame(
            pd.Series(
                df['Thickness'],
                name = 'Thickness'
            )
        )
        l = pd.DataFrame(
            pd.Series(
                df['Lmax_x'],
                name = 'Lmax'
            )
        )
        a = pd.DataFrame(
            pd.Series(
                df['Area_x'],
                name = 'Area'
            )
        )

        s = pd.DataFrame(
            pd.Series(
                df['Slope_x'],
                name = 'Slope'
            )
        )

        e = pd.DataFrame(
            pd.Series(
                df['Zmin_x'],
                name = 'Zmin'
            )
        )

        r = pd.DataFrame(
            pd.Series(
                df['RGIId'],
                name = 'RGIId'
            )
        )
        dft = x.join(y)
        dft = dft.join(l)
        dft = dft.join(a)
        dft = dft.join(s)
        dft = dft.join(e)
        dft = dft.join(r)
        dfr = pd.concat([dfr, dft])
    return dfr

# collect all residuals
dfr = find_glacier_resid(df)

# collect residuals for models from each glacier
res_min = pd.DataFrame()
res_max = pd.DataFrame()
for i in tqdm(df['RGIId'].unique()):
    dft = df[df['RGIId'] == i]
    f = find_glacier_resid(dft)
    rmin = pd.DataFrame(f.min()).T
    rmax = pd.DataFrame(f.max()).T
    res_min = pd.concat([res_min, rmin])
    res_max = pd.concat([res_max, rmax])

    

def findlog(x):
    if x > 0:
        log = math.log(x)
    elif x < 0:
        log = math.log(x*-1)*-1
    elif x == 0:
        log = 0
    return log

dfr['log residual'] = dfr['Residual'].apply(
    lambda row: findlog(row)
)

df['log thickness'] = np.log(df['Thickness'])

print('data assembled')
fig, ax = plt.subplots(2,2,figsize = (10,8))

    
    
feat = 'Area'
res_min = res_min.sort_values(feat, ascending = True)
res_max = res_max.sort_values(feat, ascending = True)
x = res_min[feat].astype(float)
y1 = res_min['Residual'].astype(float)
y2 = res_max['Residual'].astype(float)
dfr['log ' + feat] = np.log(dfr[feat])
xy = np.vstack(
    [
      dfr['log ' + feat], dfr['log residual']
    ]
)
print('calculating area residual density...')
z = gaussian_kde(xy)(xy)
ax[0][0].scatter(
    dfr[feat],
    dfr['Residual'],
    marker = '.',
    c = z,
    cmap = 'viridis'
)
model1 = np.poly1d(np.polyfit(x, y1, 2))
model2 = np.poly1d(np.polyfit(x, y2, 2))
ax[0][0].set_xscale('log')
ax[0][0].set_xlabel('Area (km$^2$)')
ax[0][0].plot(x,model1(x),color = 'r')
ax[0][0].plot(x,model2(x),color = 'r')


feat = 'Lmax'
res_min = res_min.sort_values(feat, ascending = True)
res_max = res_max.sort_values(feat, ascending = True)
x = res_min[feat].astype(float)
y1 = res_min['Residual'].astype(float)
y2 = res_max['Residual'].astype(float)
dfr['log ' + feat] = np.log(dfr[feat])
xy = np.vstack(
    [
      dfr['log ' + feat], dfr['log residual']
    ]
)
print('calculating lmax residual density...')
z = gaussian_kde(xy)(xy)
ax[0][1].scatter(
    dfr[feat],
    dfr['Residual'],
    marker = '.',
    c = z,
    cmap = 'viridis'
)
model1 = np.poly1d(np.polyfit(x, y1, 2))
model2 = np.poly1d(np.polyfit(x, y2, 2))
ax[0][1].plot(x,model1(x),color = 'r')
ax[0][1].plot(x,model2(x),color = 'r')
ax[0][1].set_xlabel('Max Length (m)')
ax[0][1].set_xscale('log')



feat = 'Slope'
res_min = res_min.sort_values(feat, ascending = True)
res_max = res_max.sort_values(feat, ascending = True)
x = res_min[feat].astype(float)
y1 = res_min['Residual'].astype(float)
y2 = res_max['Residual'].astype(float)
dfr['log ' + feat] = np.log(dfr[feat])
xy = np.vstack(
    [
      dfr['log ' + feat], dfr['log residual']
    ]
)
print('calculating slope residual density...')

z = gaussian_kde(xy)(xy)
ax[1][0].scatter(
    dfr[feat],
    dfr['Residual'],
    marker = '.',
    c = z,
    cmap = 'viridis'
)
model1 = np.poly1d(np.polyfit(x, y1, 2))
model2 = np.poly1d(np.polyfit(x, y2, 2))
ax[1][0].plot(x,model1(x),color = 'r')
ax[1][0].plot(x,model2(x),color = 'r')
ax[1][0].set_xlabel('Slope (degrees)')



feat = 'Zmin'
res_min = res_min.sort_values(feat, ascending = True)
res_max = res_max.sort_values(feat, ascending = True)
x = res_min[feat].astype(float)
y1 = res_min['Residual'].astype(float)
y2 = res_max['Residual'].astype(float)
dfr['log ' + feat] = np.log(dfr[feat])
xy = np.vstack(
    [
      dfr['log ' + feat], dfr['log residual']
    ]
)
print('calculating zmin residual density...')

z = gaussian_kde(xy)(xy)
ax[1][1].scatter(
    dfr[feat],
    dfr['Residual'],
    marker = '.',
    c = z,
    cmap = 'viridis'
)
model1 = np.poly1d(np.polyfit(x, y1, 2))
model2 = np.poly1d(np.polyfit(x, y2, 2))
ax[1][1].plot(x,model1(x),color = 'r')
ax[1][1].plot(x,model2(x),color = 'r')
ax[1][1].set_xlabel('Min Elevation(m)')

fig.supylabel('Residual (m)')
plt.tight_layout()
plt.colorbar(label = 'Log Density')

plt.savefig('feature_residuals.png')




# df = pd.DataFrame()
# for i in range(0,25,1):
#     x = pd.DataFrame(
#             pd.Series(
#                 glathida_estimates[str(i)] - glathida_estimates['Thickness'],
#                 name = 'Residual'
#         )
#     )
#     y = pd.DataFrame(
#         pd.Series(
#             glathida_estimates['Thickness'],
#             name = 'GlaThiDa Survey Thickness'
#         )
#     )
#     dft = x.join(y)
#     df = pd.concat([df, dft])
    
# df = df.dropna()

print('Computing thickness residual density...')

xy = np.vstack(
    [
      df['log thickness'], df['log residual']
    ]
)
z = gaussian_kde(xy)(xy)
print('density computed')
x = dfr['Thickness']
y = dfr['Residual']

# b, m = np.polyfit(x,y,1)
print('fitting line')
model = np.poly1d(np.polyfit(x, y, 2))


polyline = np.arange(x.min(), x.max(), 1)


# fig, ax = plt.subplots(1,1,figsize = (10,10))

print('plotting')
plt.scatter(
    x,
    y,
    marker = '.',
    c = z,
    cmap = 'viridis'
)
plt.ylabel('Residual (m)')
plt.xlabel('GlaThiDa Thickness (m)')
# plt.plot(x, b + m * x, '-')

plt.plot(polyline, model(polyline), color = 'red')
plt.colorbar(label = 'Log Density')

# plt.ylim(-10,550)
# ax.set_xlim(-10,550)
# plt.ylim(0,500)
# plt.title('Model Residuals as a function of GlaThiDa Survey Thickness\n' + str(model))

print('saving')
plt.savefig('residual_thickness.png')