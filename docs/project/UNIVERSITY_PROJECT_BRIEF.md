## Data Acquisition Requirement

The project depends on collecting review data through an existing scraping script named `scrapper.py`.

### Current scraping context
- `scrapper.py` is currently the expected entry point for obtaining the raw review data.
- The scraping approach and code quality must be reviewed before the data pipeline is considered stable.
- The repository must support scraping with Google Chrome.
- The browser driver management approach must be reviewed and documented.

### Driver / browser requirement
The project currently assumes a Chrome-based Selenium workflow.

The agent must inspect `scrapper.py` and determine:
- whether the script currently works with Google Chrome
- how the browser driver is currently managed
- whether the current setup should use:
  - Selenium Manager (built into modern Selenium), or
  - `webdriver-manager`, if there is a specific reason to keep it

The chosen approach must be documented and justified.

### Data milestone requirement
One of the core implementation tasks must be:

1. obtain the raw review data
2. clean and standardize the collected data

This is not optional. The project cannot progress meaningfully into analysis without a reviewed and usable data acquisition + cleaning path.

### Validation expectation
The scraping/data-acquisition part of the project should only be considered complete when:
- `scrapper.py` has been reviewed
- the Chrome driver setup is explicit and reproducible
- raw data can be obtained or the blocker is clearly documented
- the output format is known and documented
- the next cleaning step is defined or implemented