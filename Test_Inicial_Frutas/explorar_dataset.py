import os
import sys
import yaml

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

RUTA_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.yaml')
EXTENSIONES_IMAGEN = ('.jpg', '.jpeg', '.png', '.bmp')


def resolver_carpeta(ruta_relativa, carpeta_base):
    """Busca la carpeta de imágenes tal cual la indica el yaml, y si no
    existe (rutas tipo '../train/images'), prueba junto al propio data.yaml."""
    candidatos = [
        os.path.normpath(os.path.join(carpeta_base, ruta_relativa)),
        os.path.normpath(os.path.join(carpeta_base, os.path.basename(os.path.dirname(ruta_relativa)), os.path.basename(ruta_relativa))),
    ]
    for candidato in candidatos:
        if os.path.isdir(candidato):
            return candidato
    return candidatos[0]


def contar_imagenes(carpeta):
    if not os.path.isdir(carpeta):
        return None
    return sum(1 for f in os.listdir(carpeta) if f.lower().endswith(EXTENSIONES_IMAGEN))


def main():
    with open(RUTA_YAML, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    carpeta_base = os.path.dirname(RUTA_YAML)

    print('Resumen del dataset')
    print('=' * 40)

    for clave, etiqueta in (('train', 'Train'), ('val', 'Valid')):
        ruta_relativa = data.get(clave)
        if not ruta_relativa:
            continue
        carpeta = resolver_carpeta(ruta_relativa, carpeta_base)
        total = contar_imagenes(carpeta)
        if total is None:
            print(f'{etiqueta}: carpeta no encontrada ({carpeta})')
        else:
            print(f'{etiqueta}: {total} imágenes ({carpeta})')

    nombres = data.get('names', [])
    print('=' * 40)
    print(f'Clases configuradas ({len(nombres)}):')
    for i, nombre in enumerate(nombres):
        print(f'  {i}: {nombre}')


if __name__ == '__main__':
    main()
