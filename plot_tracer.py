def fields_for_plotting(ds,all_tracers,grid):
    import xarray as xr
    import numpy as np
    drF=-ds.Zp1.diff('Zp1')
    layers_depth_pac = (grid.cumsum((ds.LaHs1RHO*ds.dxG).mean('time').where(ds.XC<140).sum('XC')/ds.dxG.where(ds.XC<140).sum('XC'), '1RHO',   boundary='fill').load())
    layers_depth_atl=(grid.cumsum((ds.LaHs1RHO*ds.dxG).mean('time').where(ds.XC>140).sum('XC')/ds.dxG.where(ds.XC>140).sum('XC'), '1RHO',  boundary='fill').load())
    layers_depth=grid.cumsum(ds.LaHs1RHO.mean('time').mean('XC'), '1RHO', boundary='fill').load()
    ROC_pac=grid.cumsum((ds.LaVH1RHO[:,:,:,0:139]*ds.dxG[:,0:139]).mean('time').sum('XC'),'1RHO',boundary='fill').load()
    ROC_atl=grid.cumsum((ds.LaVH1RHO[:,:,:,140:215]*ds.dxG[:,140:215]).mean('time').sum('XC'),'1RHO',boundary='fill').load()
    ROC_tot=grid.cumsum((ds.LaVH1RHO*ds.dxG).mean('time').sum('XC'),'1RHO',boundary='fill').load()
    layersC=grid.interp(grid.cumsum(ds.LaHs1RHO.mean('time'),'1RHO',boundary='fill'),'Y',boundary='fill')
    z, rubbish = xr.broadcast(ds.Zp1, all_tracers.mean('Z').mean('time'))
    drK=xr.zeros_like(ds.Zp1)
    drK[1:ds['Zp1'].size]=drF.values
    #drK=ds.drF.rename({'Z':'Zp1'})
    #drK['Zp1']=ds['Zp1'][0:ds['Zp1'].size-1]
    z2=z+drK
    tracerb=xr.full_like(all_tracers.mean('time').mean('Z')*layersC,np.nan)
    for i in range(0, ds['layer_1RHO_bounds'].size):
        allin=z>=-layersC[i,:,:]
        notallin=z2>=-layersC[i,:,:]
        allin1=allin[1:33,:,:,:,:].rename({'Zp1':'Z'})
        allin1['Z']=ds['Z']
        notallin1=notallin[1:33,:,:,:,:].rename({'Zp1':'Z'})
        notallin1['Z']=ds['Z']
        tracerb=tracerb.load()
        tracerb[:,:,:,:,i]=((allin1*(all_tracers*ds.drF)).mean('time').sum('Z')
                +((allin1 != notallin1)*(all_tracers)).mean('time').sum('Z')
                            *(layersC.isel(layer_1RHO_bounds=i)-(allin1*ds.drF).sum('Z')))
    #tracerb=tracerb.rename({'layer_1RHO_center':'layer_1RHO_bounds'})
    #tracerb.coords['layer_1RHO_bounds']=ds['layer_1RHO_bounds'][1:ds['layer_1RHO_bounds'].size]
    tracerb=tracerb.diff('layer_1RHO_bounds')/layersC.diff('layer_1RHO_bounds')
    all_tracers_mean_pac=tracerb[:,:,:,0:139].where(ds.XC<140).mean(dim=('XC')).load()
    all_tracers_mean_atl=tracerb[:,:,:,140:215].where(ds.XC>140).mean(dim=('XC')).load()
    return(all_tracers_mean_pac,all_tracers_mean_atl,layers_depth_atl,layers_depth_pac,layers_depth,ROC_pac,ROC_atl,ROC_tot,tracerb)



def plot_tracers3(ds,all_tracers_mean_pac,layers_depth_pac,layers_depth,ROC_pac,ROC_tot,grid):
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    the_tracers=all_tracers_mean_pac[:,0,:,:]
    layer_1RHO_center_new = xr.DataArray(layers_depth_pac, dims=['layer_1RHO_bounds','YG'], coords={'layer_1RHO_bounds': ds.layer_1RHO_bounds,'YG':ds.YG})
    layer2D,_=(xr.broadcast(-9.81*(ds.layer_1RHO_bounds-35)/1035, ds.YG))
    layer2D.coords['depth']=(-layer_1RHO_center_new)
    b = ROC_pac.copy()
    b[:,0:35]=ROC_tot[:,0:35].copy()
    the_tracers.coords['depth']=-grid.interp(layers_depth_pac,'Y',boundary='fill')
    p = the_tracers.plot(figsize=(17, 6), col='case', col_wrap=3,x='YC',y='depth', vmin=0, vmax=1,rasterized=True,cbar_kwargs={'orientation':"horizontal",'fraction':0.07})
    layers_depth_pac[:,0:35]=layers_depth[:,0:35]
    b.coords['depth']=-layers_depth_pac
    x=[r'$\kappa_{redi}=50m^2/s$', r'$\kappa_{redi}=500m^2/s$', r'$\kappa_{redi}=5000m^2/s$', r'$\kappa_v=2 \times 10^{-5}m^2/s$, $\kappa_{redi}=500m^2/s$', r'$\kappa_v=2 \times 10^{-4}m^2/s$, $\kappa_{redi}=500m^2/s$', r'$\kappa_v=2 \times 10^{-3}m^2/s$, $\kappa_{redi}=500m^2/s$']
    i=0
    for ax in p.axes.flat:
        layer2D.plot.contour(x='YG',y='depth',ax=ax,levels=np.arange(0,0.04,0.002),colors='black',linewidths=0.5)
        (b/10**6).plot.contour(x='YG',y='depth',ax=ax,vmin=0,vmax=25,xlim=(-70,70),ylim=(-4000,0),levels=np.arange(0,25,2),colors='white',linewidths=0.5)
        (b/10**6).plot.contour(x='YG',y='depth',ax=ax,vmin=-25,vmax=0,xlim=(-70,70),ylim=(-4000,0),levels=np.arange(-25,0,2),colors='white', 
                               linestyles='dashed',linewidths=0.5)
        ax.axvline(-36,color='yellow')
        ax.axvline(-52.5,color='yellow')
        plt.xlim(-70,70)
        ax.set_title(x[i])
        i=i+1
    return p


def plot_tracerspac(ds,all_tracers_mean_pac,layers_depth_pac,layers_depth,ROC_pac,ROC_tot,grid):
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    the_tracers=all_tracers_mean_pac[:,1,:,:]
    layer_1RHO_center_new = xr.DataArray(layers_depth_pac, dims=['layer_1RHO_bounds','YG'], coords={'layer_1RHO_bounds': ds.layer_1RHO_bounds,'YG':ds.YG})
    layer2D,_=(xr.broadcast(-9.81*(ds.layer_1RHO_bounds-35)/1035, ds.YG))
    layer2D.coords['depth']=(-layer_1RHO_center_new)
    b = ROC_pac.copy()
    b[:,0:35]=ROC_tot[:,0:35].copy()
    the_tracers.coords['depth']=-grid.interp(layers_depth_pac,'Y',boundary='fill')
    p = the_tracers.plot(figsize=(17, 6), col='case', col_wrap=3,x='YC',y='depth', vmin=0, vmax=1,rasterized=True,cbar_kwargs={'orientation':"horizontal",'fraction':0.07})
    layers_depth_pac[:,0:35]=layers_depth[:,0:35]
    b.coords['depth']=-layers_depth_pac
    x=[r'$\kappa_{redi}=50m^2/s$', r'$\kappa_{redi}=500m^2/s$', r'$\kappa_{redi}=5000m^2/s$', r'$\kappa_v=2 \times 10^{-5}m^2/s$, $\kappa_{redi}=500m^2/s$', r'$\kappa_v=2 \times 10^{-4}m^2/s$, $\kappa_{redi}=500m^2/s$', r'$\kappa_v=2 \times 10^{-3}m^2/s$, $\kappa_{redi}=500m^2/s$']
    i=0
    for ax in p.axes.flat:
        layer2D.plot.contour(x='YG',y='depth',ax=ax,levels=np.arange(0,0.04,0.002),colors='black',linewidths=0.5)
        (b/10**6).plot.contour(x='YG',y='depth',ax=ax,vmin=0,vmax=25,xlim=(-70,70),ylim=(-4000,0),levels=np.arange(0,25,2),colors='white',linewidths=0.5)
        (b/10**6).plot.contour(x='YG',y='depth',ax=ax,vmin=-25,vmax=0,xlim=(-70,70),ylim=(-4000,0),levels=np.arange(-25,0,2),colors='white', 
                               linestyles='dashed',linewidths=0.5)
        ax.axvline(-36,color='yellow')
        ax.axvline(-52.5,color='yellow')
        plt.xlim(-70,70)
        ax.set_title(x[i])
        i=i+1
    return p

def plot_tracerschannel(ds,all_tracers_mean_pac,layers_depth_pac,layers_depth,ROC_pac,ROC_tot,grid):
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    the_tracers=all_tracers_mean_pac[:,8,:,:]+all_tracers_mean_pac[:,9,:,:]
    layer_1RHO_center_new = xr.DataArray(layers_depth_pac, dims=['layer_1RHO_bounds','YG'], coords={'layer_1RHO_bounds': ds.layer_1RHO_bounds,'YG':ds.YG})
    layer2D,_=(xr.broadcast(-9.81*(ds.layer_1RHO_bounds-35)/1035, ds.YG))
    layer2D.coords['depth']=(-layer_1RHO_center_new)
    b = ROC_pac.copy()
    b[:,0:35]=ROC_tot[:,0:35].copy()
    the_tracers.coords['depth']=-grid.interp(layers_depth_pac,'Y',boundary='fill')
    p = the_tracers.plot(figsize=(17, 6), col='case', col_wrap=3,x='YC',y='depth', vmin=0, vmax=1,rasterized=True,cbar_kwargs={'orientation':"horizontal",'fraction':0.07})
    layers_depth_pac[:,0:35]=layers_depth[:,0:35]
    b.coords['depth']=-layers_depth_pac
    x=[r'$\kappa_{redi}=50m^2/s$', r'$\kappa_{redi}=500m^2/s$', r'$\kappa_{redi}=5000m^2/s$', r'$\kappa_v=2 \times 10^{-5}m^2/s$, $\kappa_{redi}=500m^2/s$', r'$\kappa_v=2 \times 10^{-4}m^2/s$, $\kappa_{redi}=500m^2/s$', r'$\kappa_v=2 \times 10^{-3}m^2/s$, $\kappa_{redi}=500m^2/s$']
    i=0
    for ax in p.axes.flat:
        layer2D.plot.contour(x='YG',y='depth',ax=ax,levels=np.arange(0,0.04,0.002),colors='black',linewidths=0.5)
        (b/10**6).plot.contour(x='YG',y='depth',ax=ax,vmin=0,vmax=25,xlim=(-70,70),ylim=(-4000,0),levels=np.arange(0,25,2),colors='white',linewidths=0.5)
        (b/10**6).plot.contour(x='YG',y='depth',ax=ax,vmin=-25,vmax=0,xlim=(-70,70),ylim=(-4000,0),levels=np.arange(-25,0,2),colors='white', 
                               linestyles='dashed',linewidths=0.5)
        ax.axvline(-36,color='yellow')
        ax.axvline(-52.5,color='yellow')
        plt.xlim(-70,70)
        ax.set_title(x[i])
        i=i+1
    return p

def plot_tracers4(ds,all_tracers_mean_pac,layers_depth_pac,layers_depth,ROC_pac,ROC_tot,grid):
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    the_tracers=all_tracers_mean_pac[:,10,:,:]
    layer_1RHO_center_new = xr.DataArray(layers_depth_pac, dims=['layer_1RHO_bounds','YG'], coords={'layer_1RHO_bounds': ds.layer_1RHO_bounds,'YG':ds.YG})
    layer2D,_=(xr.broadcast(-9.81*(ds.layer_1RHO_bounds-35)/1035, ds.YG))
    layer2D.coords['depth']=(-layer_1RHO_center_new)
    b = ROC_pac.copy()
    b[:,0:35]=ROC_tot[:,0:35].copy()
    the_tracers.coords['depth']=-grid.interp(layers_depth_pac,'Y',boundary='fill')
    p = the_tracers.plot(figsize=(17, 5), col='case', col_wrap=3,x='YC',y='depth',rasterized=True,cmap='viridis',vmin=0)
    layers_depth_pac[:,0:35]=layers_depth[:,0:35]
    b.coords['depth']=-layers_depth_pac
    for ax in p.axes.flat:
        layer2D.plot.contour(x='YG',y='depth',ax=ax,levels=np.arange(0,0.04,0.002),colors='black')
        (b/10**6).plot.contour(x='YG',y='depth',ax=ax,vmin=0,vmax=25,xlim=(-70,70),ylim=(-4000,0),levels=25/2,colors='white',linewidths=0.5)
        (b/10**6).plot.contour(x='YG',y='depth',ax=ax,vmin=-25,vmax=0,xlim=(-70,70),ylim=(-4000,0),levels=25/2,colors='white', 
                               linestyles='dashed',linewidths=0.5)
        ax.axvline(-36,color='yellow')
        ax.axvline(-52.5,color='yellow')
        plt.xlim(-70,70)
    return p