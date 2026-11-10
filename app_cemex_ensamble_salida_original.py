

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
