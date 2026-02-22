import importlib.util
import traceback

print('DEBUG_CALL: loading Main.py as module')
try:
    spec = importlib.util.spec_from_file_location('Main_mod', 'Main.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print('DEBUG_CALL: module exec completed')
    if hasattr(module, 'main'):
        print('DEBUG_CALL: calling main()')
        try:
            module.main()
            print('DEBUG_CALL: main() returned')
        except Exception as e:
            print('DEBUG_CALL: main() exception', type(e).__name__, e)
            traceback.print_exc()
    else:
        print('DEBUG_CALL: no main() found')
except Exception as e:
    print('DEBUG_CALL: load exception', type(e).__name__, e)
    traceback.print_exc()
