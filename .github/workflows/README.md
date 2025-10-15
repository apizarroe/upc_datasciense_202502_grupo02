# GitHub Actions - Workflows de Calidad de Código

## 📋 code-quality.yml

Este workflow valida automáticamente la calidad del código usando las
mismas herramientas configuradas en pre-commit local.

### 🔄 Cuándo se ejecuta

#### En Push (commits directos):
- ✅ Se ejecuta en **TODAS las ramas**
- ❌ **EXCEPTO** en `feature/ghg/proyecto_tf` (rama experimental)

Esto te permite trabajar libremente en tu rama experimental sin
restricciones.

#### En Pull Requests:
- ✅ Se ejecuta **SIEMPRE** en todos los PRs
- ✅ Incluye PRs desde `feature/ghg/proyecto_tf` hacia otras ramas

Esto garantiza que cuando quieras integrar código de la rama
experimental, se valide antes del merge.

### 🛠️ Herramientas que ejecuta

El workflow tiene 2 jobs principales:

#### 1. Pre-commit Validation
Ejecuta todos los hooks configurados en `.pre-commit-config.yaml`:
- trailing-whitespace
- end-of-file-fixer
- check-yaml, check-json, check-toml
- check-added-large-files
- check-merge-conflicts
- debug-statements

#### 2. Code Quality Tools
Ejecuta las herramientas de calidad de código:

**Black** (Formateador de código)
```bash
black --check --line-length=79 src/
```

**isort** (Ordenador de imports)
```bash
isort --check-only --profile black --line-length=79 src/
```

**flake8** (Linter)
```bash
flake8 src/ --max-line-length=79 --extend-ignore=E203,W503
```

**mypy** (Type checker)
```bash
mypy src/ --ignore-missing-imports --no-strict-optional
```

### 📊 Resultados

Los resultados se muestran en:
1. La pestaña **Actions** del repositorio en GitHub
2. En el **Pull Request** si es aplicable
3. Un resumen automático al final del workflow

### ✅ Estado de los checks

GitHub mostrará:
- ✅ Check verde si todo pasa
- ❌ Check rojo si algo falla
- 🟡 Check amarillo si está en progreso

### 🚫 Bloqueo de merges

Por defecto, los checks son **requeridos** pero no bloquean el merge.

Para **bloquear merges** si fallan los checks:
1. Ve a Settings > Branches en GitHub
2. Agrega una Branch Protection Rule para `main`
3. Marca "Require status checks to pass before merging"
4. Selecciona "Pre-commit Validation" y "Code Quality Tools"

### 🔧 Modificar el workflow

Para editar qué se valida o cuándo se ejecuta:
```bash
.github/workflows/code-quality.yml
```

### 💡 Agregar más ramas experimentales

Para agregar más ramas que se salten en push, edita la sección `on`:

```yaml
push:
  branches:
    - '**'
    - '!feature/ghg/proyecto_tf'
    - '!otra-rama-experimental'
    - '!draft/*'  # Patrón para múltiples ramas
```

### 🐛 Troubleshooting

**El workflow no se ejecuta:**
- Verifica que el archivo esté en `.github/workflows/`
- El nombre debe terminar en `.yml` o `.yaml`
- Debe estar en la rama principal del repo

**Falla en mypy:**
- MyPy tiene `continue-on-error: true`
- No bloquea el workflow si falla
- Es solo informativo

**Cache de pre-commit:**
- Se cachea para acelerar ejecuciones
- Se invalida si cambia `.pre-commit-config.yaml`
