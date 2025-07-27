@echo off
REM Virtual Agora - Windows batch script
REM A structured multi-agent AI discussion platform

setlocal

REM Get the directory of this script
set SCRIPT_DIR=%~dp0
set SRC_DIR=%SCRIPT_DIR%..\src

REM Add source directory to Python path
set PYTHONPATH=%SRC_DIR%;%PYTHONPATH%

REM Try to run the application
python -c "
import sys
import os
from pathlib import Path

# Add source to path
script_dir = Path(r'%SCRIPT_DIR%').resolve()
src_dir = script_dir.parent / 'src'
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

try:
    from virtual_agora.main import main
    main()
except ImportError as e:
    print(f'Error: Could not import Virtual Agora. {e}')
    print('\nPlease ensure Virtual Agora is installed:')
    print('  pip install -e .')
    print('\nOr run from the project directory:')
    print('  python src\\virtual_agora\\main.py')
    sys.exit(1)
except KeyboardInterrupt:
    print('\nOperation cancelled by user.')
    sys.exit(130)
except Exception as e:
    print(f'Fatal error: {e}')
    sys.exit(1)
" %*

endlocal