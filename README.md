### Automated_Predictive_Maintenance

## useful pytest commands for project
# Run all tests
pytest

# Run with coverage report
pytest tests/test_data_validation.py --cov=src/components --cov-report=term-missing

# Run only one class
pytest tests/test_data_validation.py::TestInitiateDataValidation -v

# Stop on first failure
pytest tests/test_data_validation.py -x