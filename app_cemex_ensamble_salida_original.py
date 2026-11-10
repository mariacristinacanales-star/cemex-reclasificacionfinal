"""CEMEX: limpieza (notebook 1), outliers + transformación (notebook 2),
clasificación por ensamble Soft Voting de seis modelos (notebook _5modelos_CEMEX). Los parámetros optimizados se tomaron de los outputs guardados. No es un modelo preentrenado.
"""
import io
import hashlib
import re
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title='Reclasificación CEMEX', page_icon='📄')
st.title('Reclasificación automática de costos CEMEX – By UDEM')
st.write('Carga el archivo .xlsx de un mes de registros de costos de “Otros” de transporte propio descargado de SAP y obtén la reclasificación de los registros.')

FEATURES = [
    'Denom.clase de coste_num', 'Cl.coste_num',
    'Costo Estandar_num', 'Descrip.clases coste_num'
]
ENCODINGS = [
    'Segmento', 'Cl.coste', 'Denom.clase de coste', 'Denom.obj.',
    'Descrip.clases coste', 'Usuario', 'Cl.doc', 'Clasificación',
    'Jerarquía FOP', 'Nombre UO', 'Plaza', 'Costo Estandar',
    'Zona', 'Cia', 'Llave', 'Tipo de registro', 'Categoría de costo'
]
REQUIRED = [
    'Actual', 'Denom.obj.', 'Descrip.clases coste', 'Nº doc.',
    'Usuario', 'Nombre UO', 'Plaza', 'Cl.doc', 'Denom.obj.interl.',
    'Material', '    Ctd.reg.', 'UM', 'Costo Estandar',
    'Denom.clase de coste', 'Cl.coste', 'Jerarquía FOP'
]
DROPS_1 = [
    'Database OG ', 'Texto doc.', 'Ref.anul.', 'Nº pers.', 'Año',
    'Fecha contable', 'MonT', 'Doc.compr.', 'Texto de pedido',
    'Ord. partner', 'Denominación', 'Texto breve de material'
]


def contains(df, col, pattern):
    return df[col].astype('string').str.contains(pattern, case=False, na=False, regex=True)


def etiquetar_segun_reglas(df):
    """Mismo orden y precedencia de las celdas 62 a 71 del notebook 1."""
    df['Categoría de costo'] = pd.Series(pd.NA, index=df.index, dtype='object')
    def set_cat(condition, name):
        df.loc[condition, 'Categoría de costo'] = name

    set_cat(df['Denom.clase de coste'].isin([
        '5604 MATERIAS PRIMAS','5605 SUMINISTROS PRO','5610 REVENTA DE PROD'
    ]), 'Materiales y materias primas')
    set_cat(contains(df,'Descrip.clases coste',r'Aditivos|Agregado|Cemento|Material|Papeleria|Papelería|Oficina'), 'Materiales y materias primas')
    set_cat(contains(df,'Costo Estandar',r'Elem Seg') | contains(df,'Descrip.clases coste',r'segur'), 'Elementos de seguridad')
    set_cat(df['Denom.clase de coste'].isin(['5681 COMBUSTIBLES','5603 ENERGETICOS','5602 ENERGIA ELECTR.']), 'Combustibles y energía')
    set_cat(contains(df,'Costo Estandar',r'Gasolina|Combustible|Diesel|Energia|Energía'), 'Combustibles y energía')
    set_cat(contains(df,'Descrip.clases coste',r'Diesel'), 'Combustibles y energía')
    set_cat(df['Denom.clase de coste'].isin(['5619 MANTENIMIENTO','5619 MAINTENANCE']), 'Mantenimiento y gastos operativos')
    set_cat(contains(df,'Costo Estandar',r'Mtto|Mantenimiento'), 'Mantenimiento y gastos operativos')
    set_cat(contains(df,'Descrip.clases coste',r'Cambio de Inventarios|Gastos Complementarios Mineros'), 'Mantenimiento y gastos operativos')
    set_cat(df['Denom.clase de coste'].isin(['5689 EQPS/SERVS.TELE','5673 SERVICIOS','5617 OTROS SERVICIOS','5688 EQPO.COMPUTO/SW']), 'Servicios y tecnología')
    set_cat(contains(df,'Descrip.clases coste',r'GPS|telecom|celular|software|comput'), 'Servicios y tecnología')
    set_cat(df['Denom.clase de coste'].isin([
        '5630 SLDOS./SALARIOS','5630 OTS.SUELDOS/SAL','5630 OTS.PRESTAC.SOC',
        '5630 SEGURIDAD SOCIA','5630 INDEMNIZACIONES','5630 PREST.NO.GARANT',
        '5672 HONOR.PERS.NATU','5671 HONOR.PERS.JURI'
    ]), 'Personal y prestaciones')
    set_cat(contains(df,'Costo Estandar',r'Sueldos|Personal'), 'Personal y prestaciones')
    set_cat(df['Denom.clase de coste'].eq('5676 GASTOS DE VIAJE'), 'Viajes y movilidad')
    set_cat(contains(df,'Costo Estandar',r'Gastos de Viaje'), 'Viajes y movilidad')
    set_cat(contains(df,'Descrip.clases coste',
        r'Diversos|Cafeteria|Cafetería|Comisiones y Regalias|Reuniones de Trabajo|Aseo y Limpieza|'
        r'Relac. Publicas|Relac. Públicas|Festejos y Regalos|Stand Eventos y Convenciones|'
        r'Afiliaciones|Gastos Legales|Articulos Promocionales|Suscripciones y Publicaciones|'
        r'Indemnizaciones por Calidad|Alimentos y Gastos Zootecnia'),
        'Gastos administrativos y generales')
    set_cat(df['Denom.clase de coste'].isin(['5678 SEGUROS/FIANZAS','5679 IMPUESTOS']), 'Seguros, impuestos y permisos')
    set_cat(contains(df,'Descrip.clases coste',r'Tramites y Licencias|Trámites y Licencias|Multas y actualiz.de impuestos'), 'Seguros, impuestos y permisos')
    set_cat(df['Denom.clase de coste'].isin([
        '5675 ARRENDAMIENTOS','5675 INTERCO ARREND','5683 AMORTIZACION',
        '5683 AMORTIZATION','5683 PC-AMORTIZATION','5684 PROVISIONES'
    ]), 'Arrendamientos, amortizaciones y provisiones')
    set_cat(contains(df,'Costo Estandar',r'Leasing|Alq|Amort|Provision|Provisión'), 'Arrendamientos, amortizaciones y provisiones')
    return df


