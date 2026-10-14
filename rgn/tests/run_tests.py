"""pytest'siz test çalıştırıcı: tests/test_*.py içindeki test_* fonksiyonları.

    python tests/run_tests.py                 # hepsi
    python tests/run_tests.py attention data  # yalnız test_attention.py ve test_data.py
"""
import importlib.util
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]


def main(only):
    from rgn.system import setup_process
    setup_process()
    files = sorted(HERE.glob("test_*.py"))
    if only:
        files = [f for f in files if f.stem.removeprefix("test_") in only]
    passed = failed = skipped = 0
    for f in files:
        spec = importlib.util.spec_from_file_location(f.stem, f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            t0 = time.time()
            try:
                getattr(mod, name)()
                passed += 1
                print("  GEÇTİ   %s.%s (%.1f s)" % (f.stem, name, time.time() - t0), flush=True)
            except Exception as e:
                if type(e).__name__ == "Skip":                    # _util.Skip: ortam gereği (örneğin CUDA yok)
                    skipped += 1
                    print("  ATLANDI %s.%s — %s" % (f.stem, name, e), flush=True)
                    continue
                failed += 1
                print("  KALDI   %s.%s" % (f.stem, name), flush=True)
                traceback.print_exc()
    print("\n%d geçti, %d kaldı, %d atlandı" % (passed, failed, skipped))
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1:]) else 1)
