# Configuración de Pre-commit y Herramientas de Calidad de Código

Este documento describe cómo instalar y configurar las herramientas de
calidad de código para el proyecto.

## Herramientas Incluidas

- **pre-commit**: Framework para gestionar hooks de git
- **black**: Formateador automático de código Python
- **isort**: Organizador de imports
- **flake8**: Linter para verificar estilo y errores
- **mypy**: Verificador de tipos estáticos

## Configuración de Líneas de Código

Todas las herramientas están configuradas para **79 caracteres máximo
por línea**, siguiendo la PEP 8.

## Instalación

### 1. Activar tu entorno virtual

```bash
source venv/bin/activate  # En Linux/Mac
# o
venv\Scripts\activate  # En Windows
```

### 2. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### 3. Instalar los hooks de pre-commit

```bash
pre-commit install
```

## Uso

### Ejecutar pre-commit manualmente

Para ejecutar todos los hooks en todos los archivos:

```bash
pre-commit run --all-files
```

Para ejecutar en archivos específicos:

```bash
pre-commit run --files src/pipeline/preprocesing.py
```

### Ejecución automática

Una vez instalado con `pre-commit install`, los hooks se ejecutarán
automáticamente cada vez que hagas un `git commit`. Si algún hook
falla, el commit será rechazado hasta que se corrijan los problemas.

### Ejecutar herramientas individualmente

#### Black (formateador)
```bash
black src/
black --check src/  # Solo verificar sin modificar
```

#### isort (ordenar imports)
```bash
isort src/
isort --check-only src/  # Solo verificar sin modificar
```

#### flake8 (linter)
```bash
flake8 src/
```

#### mypy (type checking)
```bash
mypy src/
```

## Archivos de Configuración

- `.pre-commit-config.yaml`: Configuración de pre-commit hooks
- `pyproject.toml`: Configuración de black, isort y mypy
- `.flake8`: Configuración de flake8

## Saltar Pre-commit (No Recomendado)

Si por alguna razón necesitas hacer un commit sin ejecutar los hooks:

```bash
git commit --no-verify -m "mensaje"
```

**Nota**: Esto no es recomendado, ya que puede introducir código que
no cumple con los estándares de calidad.

## Actualizar Hooks

Para actualizar los hooks a las últimas versiones:

```bash
pre-commit autoupdate
```

## Solución de Problemas

### Error: "command not found: pre-commit"

Asegúrate de haber activado tu entorno virtual e instalado las
dependencias.

### Los hooks modifican archivos

Algunos hooks (como black e isort) modifican archivos automáticamente.
Después de que se ejecuten, necesitas volver a agregar los archivos
modificados:

```bash
git add .
git commit -m "mensaje"
```

### Conflictos entre black e isort

La configuración está ajustada para que black e isort sean
compatibles usando `--profile black` en isort.
