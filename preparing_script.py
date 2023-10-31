import re
import warnings
warnings.filterwarnings("ignore")
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter1d
import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA #мультфакторная анова
from statsmodels.formula.api import ols #линейные модели 


import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D



from scipy.stats import f_oneway as ANOVA
import scipy


################БЛОК ПО АГРОХИМИИ###################
def aov_for_nominal(df, agrochem_property, by_points = 0):    
    """
    Считает многофакторный дисперсионный анализ по переменным 
    input - dataframe by agrochemical_proprerty
    output - resul of MANOVA
    """
    dispers = df.copy()
    agrochem_property_ = re.sub(r'[\(),.%№ /-]', '' , agrochem_property)
    dispers.columns = dispers.columns.str.replace(r'[\(),.%№ /-]', '', )
    if by_points == 0:
        reg = ols('{} ~ Глубина  *  Типобработки'.format(agrochem_property_) ,  data=dispers).fit()
    else:
        reg = ols('{} ~ Глубина  *  GPS'.format(agrochem_property_) ,  data=dispers).fit()
    aov = sm.stats.anova_lm(reg)
    aov = aov[["PR(>F)"]]
    aov.columns = ['p-value']
    aov = aov.iloc[:4,:]
    aov['p-value'] = round(aov['p-value'] , 3)
    return aov

def anova(df, varible, agrochem_property):
    """
    Считает дисперсионный анализ по глубинам



    """
    stat_test_df = pd.DataFrame()
    for ind,depth in enumerate(df["Глубина"].unique()): 
        df_by_depth = df[df["Глубина"] == depth]
        sample_list = []
        for value  in df[varible].unique():
            sample = df_by_depth[agrochem_property][df[varible] == value].values
            sample_list.append(sample) 
        
        p_val = ANOVA(*sample_list)[1]
        local_df = pd.DataFrame({"глубина":depth, 'p-value' : p_val}, index = [ind])
        stat_test_df = stat_test_df.append(local_df)
    stat_test_df["p-value"] = round(stat_test_df["p-value"],3)
    return(stat_test_df)
    
def ploting( df, hue,  agrochem_property , aov ,stat_test_df, to_lable = "по обработкам" ):
    fig = plt.figure(figsize=(7,7))
    ax1 = plt.subplot2grid((3,2), (0, 0), colspan=2, rowspan = 2)
    ax2 = plt.subplot2grid((3,2), (2, 0))
    ax3 = plt.subplot2grid((3,2), (2, 1))
    if len(df[hue].unique()) == 2:
        pal =  "prism_r"
    else:
        pal =  "tab10"
    sns.pointplot(data = df,
        x = "Глубина",
        y = agrochem_property,
        hue = hue,
        palette = pal,
        scale = 1.2,
        ci = 95,
        dodge= 0.5,
        join = False,
        capsize = .05,
        ax = ax1)
    ax1.set_title('Сравнение по {}'.format(to_lable))
    ax1.legend().set_title('Технология')

    ax2.axis('off')
    ax2.axis('tight')
    ax2.table(aov.values,rowLabels=aov.index , colLabels = aov.columns ,loc='center')
    ax2.set_title('MANOVA \nпо {} '.format(to_lable),  y=0.75 , x = 0.5)

    ax3.axis('off')
    ax3.axis('tight')
    ax3.table(stat_test_df.values, colLabels = stat_test_df.columns ,loc='center')
    ax3.set_title('ANOVA по глубинам \n {} '.format(to_lable),  y=0.75 , x = 0.5)
    plt.show(block=True)
    return fig 

def data_processing_agrochem(df, type_ , agrochem_property , to_lable = "1", by_points = 0 ):
    stats_type = df.groupby([type_,'Глубина']).agg({ np.mean,  np.std, scipy.stats.variation})
    if by_points == 0:
        features = [ 'Тип обработки', 'Глубина']
    else:
        features = ['GPS №', 'Глубина']
    features.append(agrochem_property) #добавление фичи
    df = df[features]
    stats = df.groupby([type_,'Глубина']).agg({ np.mean,  np.std, scipy.stats.variation})

    aov = aov_for_nominal(df, agrochem_property, by_points)
    stat_test_df = anova(df, type_, agrochem_property)
    fig = ploting(df, type_, agrochem_property, aov,stat_test_df,to_lable  )

    return(stats , aov, stat_test_df,fig )

####################################################################################################
################БЛОК ПО МОРФОЛОГИИ##################################################################

labs = ["CO\u2083\u00B2\u207B и HCO\u2083\u207B" , 
    "Cl\u207B",
    "SO\u2084\u00B2\u207B",
    "Ca\u00B2\u207A",
    "Mg\u00B2\u207A",
    "Na\u207A",
    "K\u207A"]
def profile_plot(sample,plot_name , horizonts = None,depth = None, colors = None, ):
    sns.set_theme(style="white", palette=None)
    agrochem_features =[
    "Сумма поглощенных оснований по Каппену, ммоль/100 г",
    'М.д. содержания  гипса (по Хитрову), %',
    'Массовая доля общего содержания карбонатов (по Козловскому), % (CO2)',
    'Органический углерод, %'
    ]

    agrochem_features_labs = [
        "СПО, ммоль/100 г",
        "Гипс, %",
        "Карботнаты, %",
        "Орг. Углерод, %" ]
    limits = [
        (20,30),
        (0,9), 
        (0,5),
        (0,2)

    ]
    hr = {'height_ratios': [ 2,4,2,2,2,2,1]}
    
    fig, ax  = plt.subplots(7,1, figsize = (10,15), gridspec_kw=hr )

    profile = sns.lineplot(data =sample, y ='Органический углерод, %' , x = "depth" ,alpha = 0,ci=None, ax = ax[-1])
    ax[-1].set( ylabel ='Горизонт', yticks =[], xlim=(0,200)  )
    ax[-1].set_xticklabels(ax[-1].get_xticks(), rotation = 90)
    ax[-1].set_xlabel('глубина', rotation = 180)
    ax[-1].set_title('№ скважины {}'.format(plot_name), fontsize = 16, rotation = 90, x = -0.1, y = 2)
    


    if horizonts != None:
        for hor in range(len(horizonts)):
            ax[-1].fill_between(x =depth[hor],y1 = 2, color=colors[hor],  hatch = ['+'])
            text_position = (depth[hor][1] - depth[hor][0])/2 + depth[hor][0]
            ax[-1].text(x = text_position , y = 0.6, s = horizonts[hor],size = 16,weight='bold' , rotation = 90)
            ax[-1].axvline(depth[hor][1],color =  "black", alpha = 0.5, linestyle = "--")

    ax_num = [2,3,4,5]
    for num, prop in enumerate(agrochem_features):

        property = gaussian_filter1d(sample[prop].astype('float'), sigma = 0.75)
        sns.lineplot(y = property , x = sample["depth"] ,color = 'black', ci=None, ax = ax[ax_num[num]])
        ax[ax_num[num]].set(xticks =[], xlabel=None, ylabel =agrochem_features_labs[num] , ylim=limits[num], xlim=(0,200))
        ax[ax_num[num]].set_yticklabels(ax[ax_num[num]].get_yticks(), rotation = 90)
        if horizonts != None:
            for line in depth:
               ax[ax_num[num]].axvline(line[1],color =  "black", alpha = 0.5, linestyle = "--")

    #соли

    kations = ['Массовая доля кальция (водорастворимая форма), ммоль/100 г почвы',
        'Массовая доля магния (водорастворимая форма), ммоль/100 г почвы',
        'Массовая доля натрия (водорастворимая форма), мг•экв на 100 г почвы',
        'Массовая доля калия (водорастворимая форма), мг•экв на 100 г почвы'
    ]
    anions = ['Карбонат и бикарбонат-ионы, ммоль/100 г', 
        'Массовая доля иона хлорида, ммоль/100 г',
        'Массовая доля иона сульфата, ммоль/100 г'
    ]
    sample[anions] = sample[anions] * -1 


    anions.extend(kations)
    salts = anions.copy()
    anions.extend(["Массовая доля плотного остатка водной вытяжки, %", "depth"])
    salt_sample = sample[anions]

    colors = ['#FF0000', "#FFF300", "#13FF00", "#00FFFB","#0000FF","#C500FF", "#FF0068" ]
    labs = ["CO\u2083\u00B2\u207B и HCO\u2083\u207B" , 
    "Cl\u207B",
    "SO\u2084\u00B2\u207B",
    "Ca\u00B2\u207A",
    "Mg\u00B2\u207A",
    "Na\u207A",
    "K\u207A"]
    legend_list = []
    for i in range(len(labs)):
        legend_list.append(mpatches.Patch(color=colors[i], label= labs[i],alpha=0.5))


    sns.lineplot(y =gaussian_filter1d(salt_sample["Массовая доля плотного остатка водной вытяжки, %"], sigma = 0.75) , x = salt_sample["depth"], color = 'black' ,ci=None, ax = ax[0])
    ax[0].invert_xaxis()
    ax[0].set(xticks=[], xlabel=None, ylabel ='Плотн.ост, %' )
    ax[0].set(ylim=(0, 2), label = '', xlim=(0,200))
    ax[0].set_yticklabels(ax[0].get_yticks(), rotation = 90)

    for line in depth:
        ax[0].axvline(line[1],color =  "black", alpha = 0.5, linestyle = "--")


    for num, sal in enumerate(salts):
        sns.lineplot( y =gaussian_filter1d(salt_sample[sal] , sigma = 0.75) , x = salt_sample["depth"] ,color = colors[num], ci=None, ax = ax[1])
        ax[1].fill_between(x =salt_sample["depth"],y1 = 0, y2 = gaussian_filter1d(salt_sample[sal] , sigma = 0.75), color= colors[num], alpha=0.5)

    ax[1].set(ylim=(-5, 5), ylabel ='Концентрация растворимых солей\nммоль/100 г', xlim=(0,200))
    ax[1].axhline(0,color =  "black")
    ax[1].set(xticks =[])
    ax[1].set_xlabel('', rotation = 180)
    ax[1].set_yticklabels(ax[1].get_yticks(), rotation = 90)
    for line in depth:
        ax[1].axvline(line[1],color =  "black", alpha = 0.5, linestyle = "--")

    ax[-1].set_title('№ скважины {}'.format(plot_name), fontsize = 16, rotation = 90, x = -0.1, y = 2)
    plt.legend(handles=legend_list, title='Растворимые соли', loc='lower left', mode = 'expand',ncol = 3, bbox_to_anchor=(1, 0.5, 0.5, 0.5))


#   plt.savefig('{}_профиль.png'.format(plot_name)) 

def gransostav_plot(gran_sostav, point):
    sample = gran_sostav[gran_sostav["GPS №"] == point]
    sns.set(rc={ 'figure.facecolor':'white'})
    small_labs = sample.columns[1:7]
    big_labs = sample.columns[7:9]
    fig, ax  = plt.subplots(2,1, figsize = (4,8))

    ax[0].pie(x = sample[big_labs].values.reshape(-1),autopct="%.1f%%",explode=[0.05]  * 2 ,wedgeprops={'edgecolor': 'black'}, labels = big_labs, shadow=True)
    ax[1].pie(x = sample[small_labs].values.reshape(-1),autopct="%.1f%%",explode=[0.05]  * 6 , wedgeprops={'edgecolor': 'black'}, labels = small_labs, shadow=True)
    ax[0].set_title("Гранулометрический состав {}".format(point))

    plt.show()


    #####################################################
    ###############Физика################################
def agrofiz_plot(data, proprety):
        sns.set(font_scale = 14)

        sns.set_theme(style="white", palette=None)
        fig = plt.figure(figsize=(7,7))
        plot = sns.pointplot(data = data,
                        x = "Тип обработки",
                        y = proprety,
                        hue = "Тип обработки",
                        palette = "prism",
                        scale = 1.2,
                        ci = 95,
                        join = False,
                        capsize = .05,)
        plot.set_title('Сравнение по обработкам')
        plot.legend().set_title(('Технология'))

        samp = data[["Тип обработки",proprety ]]
        stats = samp.groupby(["Тип обработки"]).agg({ np.mean,  np.std, scipy.stats.variation})
        a = samp[proprety][samp["Тип обработки"] == 'ТТ'].values
        b = samp[proprety][samp["Тип обработки"] != 'ТТ'].values
        AOV = ANOVA(a,b)

        return(stats,AOV,plot )

def SVD_vis(agrofiz):
    SVD_by_agregates = agrofiz[[
        'Тип обработки',
        'СВД водоустойчивые >10, мм',
        'СВД водоустойчивые 10-7, мм',
        'СВД водоустойчивые 7-5, мм' ,
        'СВД водоустойчивые 5-3, мм']]


    agregate_list = [
            'СВД водоустойчивые >10, мм',
            'СВД водоустойчивые 10-7, мм',
            'СВД водоустойчивые 7-5, мм' ,
            'СВД водоустойчивые 5-3, мм']

    SVD_by_agregates = SVD_by_agregates.melt(
        id_vars = 'Тип обработки' ,
        value_vars = agregate_list
        , var_name = 'Агрегаты, мм'
        , value_name='Размер, мм')

    SVD_by_agregates['Агрегаты, мм'] = SVD_by_agregates['Агрегаты, мм'].str.split(' ', expand=True)[2].str.replace(',','')

    

    anova = pd.DataFrame({"агрегаты" : agregate_list})

    p_val = []
    for i in agregate_list:
        df = agrofiz[["Тип обработки", i]].copy()

        a = df[i][df["Тип обработки"] == 'ТТ'].values
        b = df[i][df["Тип обработки"] != 'ТТ'].values
        AOV = ANOVA(a,b)[1]
        p_val.append(AOV)

    anova['P-val'] = p_val
    
    agregate_list.append("Тип обработки")
    stats = agrofiz[agregate_list].groupby(["Тип обработки"]).agg({ np.mean,  np.std, scipy.stats.variation})
    
    
    sns.set_theme(style="white", palette=None)
    fig = plt.figure(figsize=(3,3))
    plot = sns.pointplot(data = SVD_by_agregates,
                    x = "Агрегаты, мм",
                    y = "Размер, мм",
                    hue = "Тип обработки",
                    palette = "prism",
                    scale = 1,
                    dodge = 0.5,
                    ci = 95,
                    join = False,
                    capsize = .05,)
    plot.set_title('Сравнение по обработкам')
    plot.legend().set_title('Технология')
    plt.show()
    return(stats,anova,plot)


def Kvu(agrofiz):
        Kvu_by_agregates = agrofiz[[
            'Тип обработки',
            'Кву >10, мм',
            'Кву 10-7, мм',
            'Кву 7-5, мм' ,
            'Кву 5-3, мм']]
        agregate_list = [
            'Кву >10, мм',
            'Кву 10-7, мм',
            'Кву 7-5, мм' ,
            'Кву 5-3, мм']

        Kvu_by_agregates = Kvu_by_agregates.melt(
            id_vars = 'Тип обработки' ,
            value_vars = agregate_list
            , var_name = 'Агрегаты, мм'
            , value_name='Размер, мм')

        Kvu_by_agregates['Агрегаты, мм'] = Kvu_by_agregates['Агрегаты, мм'].str.split(' ', expand=True)[1].str.replace(',','')

        

        anova = pd.DataFrame({"агрегаты" : agregate_list})

        p_val = []
        for i in agregate_list:
            df = agrofiz[["Тип обработки", i]].copy()

            a = df[i][df["Тип обработки"] == 'ТТ'].values
            b = df[i][df["Тип обработки"] != 'ТТ'].values
            AOV = ANOVA(a,b)[1]
            p_val.append(AOV)

        anova['P-val'] = p_val
        
        agregate_list.append("Тип обработки")
        stats = agrofiz[agregate_list].groupby(["Тип обработки"]).agg({ np.mean,  np.std, scipy.stats.variation})
        
        
        sns.set_theme(style="white", palette=None)
        fig = plt.figure(figsize=(3,3))
        plot = sns.pointplot(data = Kvu_by_agregates,
                        x = "Агрегаты, мм",
                        y = "Размер, мм",
                        hue = "Тип обработки",
                        palette = "prism",
                        scale = 1.2,
                        dodge = 0.5,
                        ci = 95,
                        join = False,
                        capsize = .05,)
        plot.set_title('Сравнение по обработкам')
        plt.show()
        return(stats,anova,plot)


def ob_ves_plot(ob_ves):
        fig= plt.figure(figsize=(7,5))
        plot = sns.pointplot(data = ob_ves,
                x = "GPS №",
                y = "Объемный вес",
                hue = "GPS №",
                palette = "tab10",
                scale = 1.2,
                ci = 95,
                dodge= 0.5,
                join = False,
                capsize = .05,
        )
        plot.set(title =  'Объемный вес')
        plt.show()


        aov_list = []
        for i in ob_ves['GPS №'].unique():
            lst = list(ob_ves["Объемный вес"][ob_ves['GPS №'] == i].values)
            aov_list.append(lst)
        aov = ANOVA(*aov_list)

        stats = ob_ves[["Объемный вес", "GPS №"]].groupby(["GPS №"]).agg({ np.mean,  np.std, scipy.stats.variation})

        return(stats,aov,plot)



##### скрипт для лепестковой диаграммы 


def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)


        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)


                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def radar_plot(test_df):
    rebuild = test_df[["Тип обработки","GPS №",'Размер агрегатов, мм',"Вес, г", "Вес монолита, г"]].groupby(["Тип обработки",'Размер агрегатов, мм']).agg(['mean']).round(2).reset_index()
    rebuild = rebuild.droplevel(1, axis = 1)
    rebuild['соотношение'] = (rebuild['Вес, г'] / rebuild['Вес монолита, г'] * 100).round(2)
    PP = rebuild[rebuild['Тип обработки']  == 'ПП']["соотношение"].to_list()
    TT = rebuild[rebuild['Тип обработки']  != 'ПП']["соотношение"].to_list()


    size_cats = ['>10','10-7','7-5', '5-3', '3-2', '2-1', '1-0.5', '    0.5-0.25', '<0.25']


    data = [size_cats, ('тип обработки', [PP, TT])]

    theta = radar_factory(len(size_cats), frame='circle')



    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(top=0.85, bottom=0.05)

    line = ax.plot(size_cats, TT, color = '#df204c')
    ax.fill(size_cats, TT, color = '#df204c' ,alpha=0.25)

    line = ax.plot(size_cats, PP, color = '#43cc1d')
    ax.fill(size_cats, PP, color = '#43cc1d' ,alpha=0.25)



    ax.set_varlabels(size_cats)
    return(ax)

##### барплоты чтобы сравнивать по две группы

def suhoe_stats_barplot(test_df):
    sns.set(font_scale=1.1)
    sns.set_style("ticks")

    ax = sns.barplot(test_df,
                y="Содержание агрегатов, %",
                x="Размер агрегатов, мм",
                hue="Тип обработки",
                palette = "prism",
                edgecolor = "black",
                errcolor = "black",
                errorbar = 'se',
                capsize = 0.2,
                errwidth = 1.5)

    for i, patch in enumerate(ax.patches):
            if i < len(ax.patches)/2:
                patch.set_hatch("//")
    ax.legend(loc='best', fontsize = 16, title='Тип обработки')


    rp = radar_plot(test_df)


    stats = test_df[['Тип обработки', "Размер агрегатов, мм",'Содержание агрегатов, %']].groupby(['Тип обработки', "Размер агрегатов, мм"]).agg(['mean','std','sem'])\
    
    return(ax,rp, stats.round(2))



#### Для мокрого просеивания 

def mokroe_stats_barplot(test_df):
    fig, ax  = plt.subplots(2,2, figsize = (11,11) )


    test_df.sort_values(by = ["Тип обработки","Размер сухого агрегата","Размеры фракций, мм"], inplace= True)


    sns.set(font_scale=1.1)
    sns.set_style("ticks")
    for n,i in enumerate([[">10","10-7"],["7-5","5-3"]]):
        for k in range(len(i)):
            test_df
            df = test_df[test_df['Размер сухого агрегата'] == i[k]]
            df['Размеры фракций, мм'] = df['Размеры фракций, мм'].astype(str)
            plot = sns.barplot(df,
                        y="Содержание агрегатов, %",
                        x='Размеры фракций, мм',
                        hue="Тип обработки",
                        palette = "prism",
                        edgecolor = "black",
                        errcolor = "black",
                        errorbar = 'se',
                        capsize = 0.2,
                        errwidth = 1.5,
                        ax = ax[n,k]
            ) 


            for d, patch in enumerate(plot.patches):
                if d < len(plot.patches)/2:
                    patch.set_hatch("//")
            ax[n,k].get_legend().set_visible(False)
            ax[n,k].set(title='Размер сухого агрегата {}'.format(i[k]))
    ax[0,0].legend(loc='best', fontsize = 16, title='Тип обработки')
    return(fig)

def mokroe_stats(test_df):
    test_df_for_stats = test_df[[
        'Тип обработки',
        "Размер сухого агрегата",
        "Размеры фракций, мм",
        'Содержание агрегатов, %']]
    stats = test_df_for_stats.groupby(['Тип обработки', "Размер сухого агрегата","Размеры фракций, мм",]).agg(['mean','std','sem']).reset_index().dropna(axis = 0)
    return(stats)

def radar_chart(test_df):
        test_df_for_stats = test_df[[
                'Тип обработки',
                "Размер сухого агрегата",
                "Размеры фракций, мм",
                "Содержание агрегатов, %"]]
        rebuild = test_df_for_stats.groupby(['Тип обработки',"Размер сухого агрегата","Размеры фракций, мм"]).agg(['mean']).reset_index().dropna(axis = 0)

        rebuild = rebuild.droplevel(1, axis = 1)


        for n,i in enumerate([[">10","10-7"],["7-5","5-3"]]):
                for k in range(len(i)):
                
                        df = rebuild[rebuild['Размер сухого агрегата'] == i[k]]
                        size_cats = df['Размеры фракций, мм'].unique().astype(str)
                        PP = df[df['Тип обработки']  == 'ПП']["Содержание агрегатов, %"].to_list()
                        TT = df[df['Тип обработки']  != 'ПП']["Содержание агрегатов, %"].to_list()
                        theta = radar_factory(len(size_cats), frame='circle')
                        fig, ax = plt.subplots( figsize=(5,5),subplot_kw=dict(projection='radar'))


                        fig.subplots_adjust(top=0.85, bottom=0.05)
                
                        line = ax.plot(size_cats, TT, color = '#df204c')
                        ax.fill(size_cats, TT, color = '#df204c' ,alpha=0.25)

                        line = ax.plot(size_cats, PP, color = '#43cc1d')
                        ax.fill(size_cats, PP, color = '#43cc1d' ,alpha=0.25)
                        ax.set_varlabels(size_cats)
                        ax.set(title='Размер сухого агрегата {}'.format(i[k]))

def mokroe_all(test_df):
    mokroe_stats_barplot(test_df)
    stats = (mokroe_stats(test_df))
    radar_chart(test_df)
    return(stats)