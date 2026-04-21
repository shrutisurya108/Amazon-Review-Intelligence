"""Phase 1 smoke test — verifies directory structure and logger."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.utils.logger import get_logger

def test_directories_exist():
    dirs = [
        config.DATA_RAW, config.DATA_INTERIM, config.DATA_PROCESSED,
        config.OUTPUTS_FIGURES, config.OUTPUTS_REPORTS, config.OUTPUTS_MODELS,
        config.LOGS_DIR,
    ]
    for d in dirs:
        assert d.exists(), f"Missing directory: {d}"
    print("✓ All directories exist")

def test_logger():
    logger = get_logger("test.scaffold")
    logger.info("Logger is working correctly")
    log_files = list(config.LOGS_DIR.glob("*.log"))
    assert len(log_files) > 0, "No log file was created"
    print(f"✓ Logger OK — log file: {log_files[0].name}")

def test_config_loads():
    assert config.CATEGORY == "Electronics"
    assert config.SAMPLE_SIZE == 50_000
    assert config.LDA_NUM_TOPICS == 8
    print("✓ Config loaded correctly")

if __name__ == "__main__":
    test_directories_exist()
    test_logger()
    test_config_loads()
    print("\n✅ Phase 1 scaffold test passed.")
