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


def limpiar(df):
    # Notebook 1: 0 -> NaN en todas las columnas, filtro de faltantes, documentos y montos
    df = df.copy()
    df['Fila_excel_original'] = np.arange(len(df)) + 2
    df = df.replace([0, '0'], np.nan)
    df.drop(columns=DROPS_1, errors='ignore', inplace=True)
    df = df.dropna(subset=['Actual','Denom.obj.','Descrip.clases coste','Nº doc.',
                           'Usuario','Nombre UO','Plaza']).copy()
    validos = ['KI','KV','KY','R1','SA','WA','WE']
    cond = df['Cl.doc'].isna() & df['Denom.obj.interl.'].isin(validos)
    df.loc[cond, 'Cl.doc'] = df.loc[cond, 'Denom.obj.interl.'].astype(str)
    df = df.dropna(subset=['Cl.doc']).copy()
    df.drop(columns=['Denom.obj.interl.'], errors='ignore', inplace=True)
    df['Material'] = pd.to_numeric(df['Material'], errors='coerce').astype('Int64')
    cond_drop = (df['Material'].notna() & df['    Ctd.reg.'].isna()) | (df['    Ctd.reg.'].notna() & df['UM'].isna())
    df = df.loc[~cond_drop].copy()
    cond1 = df['Material'].notna() & df['    Ctd.reg.'].notna() & df['UM'].notna()
    cond2 = df['Material'].isna() & df['    Ctd.reg.'].notna() & df['UM'].notna()
    df['Tipo de registro'] = np.select([cond1,cond2],['Material','Producto no material'], default='Servicio u otro')
    df.drop(columns=['Material','    Ctd.reg.','UM'], errors='ignore', inplace=True)
    df['Costo Estandar'] = df['Costo Estandar'].fillna('No especificado')
    df['Actual'] = pd.to_numeric(df['Actual'], errors='coerce')
    df = df.loc[df['Actual'].ge(0)].copy()
    if df.empty: raise ValueError('Ningún registro sobrevivió a los filtros de limpieza del notebook 1.')
    return etiquetar_segun_reglas(df)


def transformar(df):
    # Notebook 2: IQR 1.5, imputación de outliers con la mediana, eliminación y LabelEncoder
    df = df.copy()
    for col in ['Mes','Actual','Ce.coste','Área func.','Nº doc.','CC','CeBe']:
        if col not in df.columns: continue
        s = pd.to_numeric(df[col], errors='coerce')
        q1, q3 = s.quantile(.25), s.quantile(.75)
        iqr = q3-q1
        mask = (s < q1-1.5*iqr) | (s > q3+1.5*iqr)
        df.loc[mask, col] = s.median()
    df.drop(columns=['Mes','Ce.coste','Área func.','Nº doc.'], errors='ignore', inplace=True)
    map_names = {'Denom.obj.':'Denom.obj._num'}
    for col in ENCODINGS:
        if col not in df.columns: continue
        encoded_col = map_names.get(col, col+'_num')
        values = df[col].astype('string').fillna('nan').astype(str)
        df[encoded_col] = LabelEncoder().fit_transform(values)
    if not set(FEATURES).issubset(df.columns):
        raise ValueError('Faltan columnas transformadas para el modelo: '+', '.join(set(FEATURES)-set(df.columns)))
    return df


def clasificar(df):
    """Ensamble Soft Voting del notebook _5modelos_CEMEX.

    Cinco estimadores usan sus mejores parámetros guardados en el notebook.
    El SVM del ensamble original es el SVC RBF del diccionario `modelos`,
    con probability=True (NO el LinearSVC optimizado de la celda anterior).
    """
    df = df.copy()
    marcadas = df['Categoría de costo'].notna()
    X = df.loc[marcadas, FEATURES].apply(pd.to_numeric, errors='coerce')
    y = df.loc[marcadas, 'Categoría de costo'].astype(str)
    datos_ok = X.notna().all(axis=1)
    X, y = X.loc[datos_ok], y.loc[datos_ok]
    if y.nunique() < 2:
        raise ValueError('Las reglas generaron menos de dos categorías para entrenar.')
    if (y.value_counts() < 3).any():
        raise ValueError('Hay categorías con menos de 3 ejemplos. Se necesita suficiente información para separar el 80/20 y aplicar SMOTE.')

    # En el notebook original: train/test 80/20, stratify, random_state=0.
    x_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    cuentas = pd.Series(y_train_encoded).value_counts()
    if cuentas.min() < 2:
        raise ValueError('Hay categorías con muy pocos registros en entrenamiento para aplicar SMOTE.')
    vecinos = min(5, int(cuentas.min()) - 1)
    x_train_smote, y_train_smote = SMOTE(
        random_state=0, k_neighbors=vecinos
    ).fit_resample(x_train, y_train_encoded)

    # Los parámetros de estos cinco modelos se recuperaron de los outputs
    # de RandomizedSearchCV en el notebook proporcionado.
    rf = RandomForestClassifier(
        max_depth=10, min_samples_split=5, min_samples_leaf=1,
        n_estimators=200, random_state=0, n_jobs=1
    )
    xgb = XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.2,
        random_state=0, eval_metric='mlogloss', n_jobs=1
    )
    lr = LogisticRegression(C=0.1, max_iter=2000, random_state=0)
    svm = make_pipeline(
        StandardScaler(), SVC(kernel='rbf', probability=True, random_state=0)
    )
    nb = GaussianNB(var_smoothing=1e-12)
    dt = DecisionTreeClassifier(
        max_depth=20, min_samples_split=2, min_samples_leaf=1,
        criterion='gini', random_state=0
    )

    ensamble = VotingClassifier(
        estimators=[
            ('rf', rf), ('xgb', xgb), ('lr', lr),
            ('svm', svm), ('nb', nb), ('dt', dt)
        ], voting='soft', n_jobs=1
    )
    ensamble.fit(x_train_smote, y_train_smote)

    todas = df[FEATURES].apply(pd.to_numeric, errors='coerce')
    predecibles = todas.notna().all(axis=1)
    df['Categoría reclasificada'] = pd.NA
    if predecibles.any():
        clases_num = ensamble.predict(todas.loc[predecibles])
        df.loc[predecibles, 'Categoría reclasificada'] = le.inverse_transform(
            clases_num.astype(int)
        )
    df['Origen categoría entrenamiento'] = np.where(
        marcadas, 'Regla del notebook 1', 'Sin regla aplicable'
    )
    difiere = (
        marcadas & predecibles &
        df['Categoría de costo'].astype('string').ne(
            df['Categoría reclasificada'].astype('string')
        ).fillna(False)
    )
    df['Revisar cambio'] = np.where(difiere, 'Sí', '')
    return df


# Reconoce diferencias inocuas de escritura sin confundir columnas diferentes.
# No inventa datos ni rellena columnas faltantes.
ALIASES = {
    'Denom.obj.': ['Denom obj', 'Denom.obj', 'Denominación objeto',
                   'Denominacion objeto', 'Denominación del objeto'],
    '    Ctd.reg.': ['Ctd.reg.', 'Ctd reg', 'Ctd.reg',
                       'Cantidad registrada', 'Cantidad registro'],
    'Nº doc.': ['N° doc.', 'Nº doc', 'No doc', 'Número de documento'],
    'Costo Estandar': ['Costo Estándar'],
}


def normalizar_nombre(nombre):
    nombre = unicodedata.normalize('NFKD', str(nombre))
    nombre = ''.join(c for c in nombre if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9]+', '', nombre.casefold())


def detectar_columnas(df):
    # Solo se asigna automáticamente si hay una coincidencia única.
    existentes = list(df.columns)
    usados = set()
    renombrar = {}
    for canonico in REQUIRED:
        if canonico in existentes:
            usados.add(canonico)
            continue
        variantes = [canonico] + ALIASES.get(canonico, [])
        claves = {normalizar_nombre(v) for v in variantes}
        encontrados = [c for c in existentes
                       if c not in usados and normalizar_nombre(c) in claves]
        if len(encontrados) == 1:
            renombrar[encontrados[0]] = canonico
            usados.add(encontrados[0])
    return renombrar


# Los datos del modelo se procesan en una copia; la descarga preserva el Excel original.
st.subheader('Sube tu Excel original (.xlsx)')
archivo = st.file_uploader('Upload: Sube tu Excel original (.xlsx)', type=['xlsx'])

if archivo is not None:
    try:
        raw = archivo.getvalue()
        input_id = hashlib.sha256(raw).hexdigest()
        original = pd.read_excel(io.BytesIO(raw))
        if st.session_state.get('input_id') != input_id or 'conteos' not in st.session_state:
            st.write(f'Cantidad de registros base original: **{len(original):,}**')

        # Reiniciar resultados anteriores cuando el usuario suba otro Excel.
        if st.session_state.get('input_id') != input_id:
            st.session_state.pop('xlsx_reclasificado', None)
            st.session_state.pop('conteos', None)

        renombres = detectar_columnas(original)
        pendientes = [c for c in REQUIRED
                      if c not in original.columns and c not in renombres.values()]
        opciones_usadas = set(renombres) | {c for c in REQUIRED if c in original.columns}
        manual = {}
        if pendientes:
            # Solo mostrar opciones adicionales cuando la equivalencia sea ambigua.
            with st.expander('Relacionar columnas faltantes', expanded=True):
                for canonico in pendientes:
                    disponibles = ['— Seleccionar columna —'] + [
                        str(c) for c in original.columns if c not in opciones_usadas
                    ]
                    elegido = st.selectbox(
                        f'¿Qué columna corresponde a «{canonico}»?',
                        disponibles, key='map_' + canonico
                    )
                    if elegido != '— Seleccionar columna —':
                        manual[elegido] = canonico
                        opciones_usadas.add(elegido)
        faltan = [c for c in REQUIRED
                  if c not in original.columns
                  and c not in renombres.values()
                  and c not in manual.values()]

        if st.button('Reclasificar', type='primary', disabled=bool(faltan)):
            st.session_state.pop('xlsx_reclasificado', None)
            st.session_state.pop('conteos', None)
            try:
                with st.spinner('Limpiando, transformando y aplicando el modelo de ensamble...'):
                    # Se renombran columnas SOLO para preparar las variables del modelo.
                    preparado = original.rename(columns={**renombres, **manual})
                    limpio = limpiar(preparado)
                    transformado = transformar(limpio)
                    clasificado = clasificar(transformado)

                    # La salida sale de la base ORIGINAL, no de la transformada.
                    resultado_excel = original.copy()
                    if 'Reclasificación' in resultado_excel.columns:
                        raise ValueError('El Excel ya contiene una columna llamada Reclasificación. Renómbrala antes de procesar para no sobrescribir datos.')
                    resultado_excel['Reclasificación'] = pd.Series(
                        pd.NA, index=resultado_excel.index, dtype='object'
                    )
                    preds = clasificado['Categoría reclasificada'].dropna()
                    resultado_excel.loc[preds.index, 'Reclasificación'] = preds.to_numpy()

                    cantidad_clasificada = int(resultado_excel['Reclasificación'].notna().sum())
                    cantidad_sin_clasificar = len(resultado_excel) - cantidad_clasificada
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl',
                                        datetime_format='YYYY-MM-DD HH:MM:SS',
                                        date_format='YYYY-MM-DD') as writer:
                        resultado_excel.to_excel(writer, sheet_name='Base reclasificada', index=False)
                    st.session_state['xlsx_reclasificado'] = buffer.getvalue()
                    st.session_state['conteos'] = (
                        len(resultado_excel), cantidad_sin_clasificar, cantidad_clasificada
                    )
                    st.session_state['input_id'] = input_id
            except Exception as err:
                st.error(f'No se pudo reclasificar el archivo: {err}')

        if st.session_state.get('input_id') == input_id and 'conteos' in st.session_state:
            total, sin_clasificar, reclasificados = st.session_state['conteos']
            st.write(f'Cantidad de registros base original: **{total:,}**')
            st.write(f'Cantidad de registros sin reclasificar: **{sin_clasificar:,}**')
            st.write(f'Cantidad de registros reclasificados (predicciones): **{reclasificados:,}**')
            st.write('Archivo clasificado. Descarga el .xlsx')
            st.download_button(
                'Descargar base reclasificada .xlsx',
                data=st.session_state['xlsx_reclasificado'],
                file_name='CEMEX_base_reclasificada.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                type='primary'
            )
    except Exception as err:
        st.error(f'No se pudo leer el Excel: {err}')

st.caption('Solo se emparejan automáticamente columnas equivalentes sin ambigüedad. '
           'Si falta una columna, indica su correspondencia real. '
           'Las predicciones son propuestas y requieren validación; '
           'los registros que no puedan clasificarse permanecen en el archivo con la reclasificación vacía.')
